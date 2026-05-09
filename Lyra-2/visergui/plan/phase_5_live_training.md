# Phase 5 — Live training hook

Run optimization in a background thread; the viewer reads the current splat
state and renders it live as training progresses. The lock-protected
`SceneState` from Phase 0 is the seam everything plugs into.

## Prerequisites

Phases 0–2 minimum (3 strongly recommended; 4 optional). The renderer must
already use `SceneState.read()` for every frame. If it doesn't, fix that
before this phase — retrofitting concurrency on top of unsynchronized access
is harder than it sounds.

## Architectural decisions to make first

Two choices to commit to before writing code. Both are reasonable; they have
different tradeoffs.

### Choice 1: thread vs. process

**Thread (recommended starting point):**
- Same Python process; training and viewer share GPU memory directly.
- Splat tensors are visible to both without any IPC.
- Trade: training holds the GIL during Python-side work (most gsplat
  optimization is in CUDA kernels, so this matters less than it sounds —
  but profile if rendering stutters).

**Process (when you need isolation):**
- Training is a subprocess; it writes checkpoints; the viewer reloads them
  on a watcher.
- Trade: latency between training step and what the viewer sees (checkpoint
  write + load). Useful if the training script is something you want to keep
  running standalone, with the viewer as an optional attach-on-top.

Pick thread for the in-house monitoring tool. Process if the training script
needs to also run unattended.

### Choice 2: copy-on-read vs. live-reference

**Live-reference (recommended):** the viewer reads the same tensors training
writes to. Cheap. Risk: a single rasterization call uses tensors mutated
mid-call. Mitigation: training only mutates tensors at well-defined sync
points; the viewer's `SceneState.read()` lock blocks training during render.

**Copy-on-read:** at sync points, training swaps in a copy that the viewer
reads. Renderer never blocks training; training never blocks renderer.
Trade: 2× VRAM for the splat parameters during the swap window.

Pick live-reference. The lock contention is in microseconds; you don't need
the VRAM doubling.

If profiling later shows the lock causes visible stutter, switch to
copy-on-read with a double-buffer. Don't preempt that decision.

## What changes

Add:

- `Trainer` — wraps a gsplat optimization loop. Owns the optimizer, dataset
  iterator, step counter, latest loss.
- `TrainingThread` — runs `Trainer.step()` in a loop until told to stop;
  pauseable.
- New "Training" GUI folder.
- Loss plot in the GUI.

The renderer doesn't change. The point of the Phase 0 design is that this
phase doesn't touch render code.

## Hard requirements

1. **`SceneState` is the only shared mutable state.** Training writes splat
   tensors via `SceneState.write()` (a context manager that takes the same
   lock as `read()`). The renderer reads via `read()`. Nothing else touches
   the tensors.

2. **Training step is atomic from the viewer's perspective.** A single
   gsplat optimization step (forward + backward + optimizer step + any
   densification/pruning) happens entirely under `SceneState.write()`.
   The viewer either sees the state before that step or after, never
   during.

3. **Densification is the dangerous case.** Splat count changes mid-training.
   The renderer must always see a consistent set: `means`, `quats`, `scales`,
   `opacities`, `sh` all of length N, where N is whatever it is. The lock
   protects this; do not skip it because "just adding splats should be safe."

4. **Training never blocks waiting for a render.** It tries to acquire the
   lock; if it gets it, it proceeds; if it can't (renderer holds it),
   it... waits. That's fine. The lock is held for milliseconds at most.
   Don't try to make training non-blocking — you'll just rebuild what
   `RLock` already does.

5. **Pause/resume must be clean.** "Pause" means the training loop sleeps
   between steps; the viewer keeps rendering the current state. "Resume"
   wakes it up. Use a `threading.Event`.

6. **Stop is clean too.** On viewer shutdown, signal stop, join the thread
   with a timeout (~5 seconds), then exit. Don't leave orphan threads.

## SceneState additions

```python
class SceneState:
    def __init__(self):
        self._lock = threading.RLock()
        # ... tensors ...
        self.step = 0
        self.loss_history = []  # list of (step, loss); cap at 10k

    @contextmanager
    def read(self):
        with self._lock:
            yield self

    @contextmanager
    def write(self):
        with self._lock:
            yield self

    def record_step(self, step: int, loss: float):
        # called from training thread, under write() lock
        self.step = step
        self.loss_history.append((step, loss))
        if len(self.loss_history) > 10000:
            self.loss_history = self.loss_history[-10000:]
```

`read` and `write` are the same lock; the names communicate intent. Use an
`RLock` so a misclick (calling `read()` from inside `write()`) doesn't
deadlock.

## Trainer

Concretely depends on the optimization recipe. Sketch:

```python
class Trainer:
    def __init__(self, scene: SceneState, dataset, cfg):
        self.scene = scene
        self.dataset = dataset
        self.cfg = cfg
        self.optimizer = None  # built from scene tensors as parameters
        self._build_optimizer()

    def step(self) -> float:
        with self.scene.write() as s:
            cam, gt_image = next(self.dataset)
            rgb, _, info = rasterization(
                s.means, s.quats, s.scales, s.opacities, s.sh,
                cam.viewmat, cam.K, cam.W, cam.H,
                sh_degree=s.sh_degree,
            )
            loss = (rgb[0] - gt_image).abs().mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # densification / pruning hooks here, if relevant
            s.record_step(s.step + 1, loss.item())
        return loss.item()
```

Note: this `step()` holds the write lock for the entire forward + backward.
That's fine for monitoring; if it causes visible stutter, refactor so only
the parameter swap is locked, and forward/backward run on a snapshot.
Profile before optimizing.

## TrainingThread

```python
class TrainingThread(threading.Thread):
    def __init__(self, trainer: Trainer):
        super().__init__(daemon=True)
        self.trainer = trainer
        self._stop = threading.Event()
        self._paused = threading.Event()  # set = paused
        self._paused.set()  # start paused

    def run(self):
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            try:
                self.trainer.step()
            except Exception as e:
                print(f"[trainer] step failed: {e}")
                self._paused.set()  # auto-pause on exception

    def pause(self):  self._paused.set()
    def resume(self): self._paused.clear()
    def stop(self):   self._stop.set()
```

## "Training" GUI folder

- `gui_train_resume`: button. On click: `thread.resume()`.
- `gui_train_pause`: button. On click: `thread.pause()`.
- `gui_train_status`: read-only text. "running" / "paused" / "stopped".
- `gui_train_step`: read-only number. From `scene.step`.
- `gui_train_loss`: read-only number. Latest loss.
- `gui_train_loss_plot`: plotly line chart via viser's `add_plotly`.
  Updated at ~2 Hz (not every frame), reads `scene.loss_history`.
- `gui_train_splat_count`: read-only number. Updates as densification
  changes count.

Updating the plot every frame is a waste — throttle to 2 Hz with a wall-clock
gate.

## CLI

```
python viewer.py --train <config.yaml> [--scene initial.ply]
```

When `--train` is passed, the viewer starts in training mode; otherwise it's
the static viewer. Both modes share all of Phases 0–4.

If the training config requires a dataset path, intrinsics, etc., put them
in the YAML — don't proliferate CLI flags.

## Verification

- Start with `--train <small synthetic scene>`. Hit "Resume." Watch:
  - Step counter increments.
  - Loss readout drops over time.
  - Splat count changes (if your recipe densifies).
  - Render keeps working at smooth FPS.
  - Loss plot updates.
- Pause: training stops, viewer keeps rendering the current state.
- Resume: training continues from where it paused.
- Close the browser tab during training: training keeps running.
- Reconnect: see the current state.
- Ctrl-C the server: training stops cleanly, no orphan threads.

## Gotchas

- Don't pass `torch.no_grad()` / `inference_mode()` around the rasterization
  call inside `Trainer.step()`. That's the gradient-bearing path. The
  renderer's `inference_mode` is unaffected; the two paths exist concurrently.
- The optimizer holds references to the splat tensors. Densification typically
  rebuilds the optimizer (replacing tensors with new larger ones) — make sure
  this rebuild is inside the `write()` lock, not before/after.
- `loss.item()` does a CUDA sync. That's fine — it's once per step. Don't
  call `.item()` in tight loops elsewhere.
- If you densify and the renderer's reference to a tensor gets stale: it
  shouldn't, because the renderer reads `s.means` (etc.) inside `read()`
  freshly each frame. But if you ever cache a tensor reference outside the
  `read()` block, you'll bug. Don't cache.
- Plotly updates over websocket can be heavy. If the loss plot causes lag,
  reduce update frequency or downsample the history before plotting (e.g.
  show every Nth point if `len(history) > 1000`).

## Done means

- Training and viewer run in the same process, viewer renders the live state.
- Pause/resume/stop all work cleanly.
- Loss decreases as expected for the chosen recipe.
- Densification (if used) doesn't crash or visually corrupt the render.
- Closing and reopening the browser doesn't affect training.
- Ctrl-C exits cleanly.

## Out of scope

- Multi-GPU training
- Distributed training
- Resumable training from checkpoints (write a checkpoint button if needed,
  but full resume support is a separate project)
- Hyperparameter editing live (you can add it, but resist — most hyperparams
  shouldn't change mid-run anyway)
