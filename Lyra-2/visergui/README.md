# visergui — viser-based 3DGS viewer

Splat viewer with point-cloud overlay, depth visualization, adaptive
resolution, and an optional live-training hook. Single Python process;
the frontend is viser's React app served over websocket.

## Environment

Use the `lyra2` conda environment — it has matching versions of viser,
gsplat, torch, plyfile, plotly, and matplotlib:

```bash
PY=/home/kristofe/miniconda3/envs/lyra2/bin/python
```

(or `conda activate lyra2 && PY=python` — same result)

## Three ways to run

### 1. Static viewer (just inspect a .ply)

```bash
$PY visergui/viewer.py path/to/scene.ply --port 8080
```

Optional flags:

- `--points file.ply` (or `.npy`, `.npz`) — extra point-cloud layer.
  Repeatable.
- `--no-flip` — skip the default 180°-about-X flip applied to Inria PLYs.
- `--no-derive-points` — don't auto-create the `splat_centers` layer.
- `--max-points N` — uniform-random subsample point clouds above N points
  (default 1,000,000; `0` disables).
- `--host 0.0.0.0` (default), `--port 8080` (default).

The repo's reference scene:

```bash
$PY visergui/viewer.py ../../world_models/Lyra_Experiments/scene.ply --port 8080
```

### 2. Headless training (no viewer)

`visergui/training.py` is stdlib-only — no viser, gsplat, torch, or plotly
required when you only want to train. Simplest pattern is a plain
main-thread loop:

```python
# user_train.py — fully owned by you
class MyTrainer:
    def step(self) -> float:
        # forward / backward / optimizer / densify — your code
        ...
        return float(loss)

trainer = MyTrainer()
try:
    for i in range(MAX_STEPS):
        loss = trainer.step()
        if i % 100 == 0:
            print(f"step {i}: loss {loss:.4f}")
except KeyboardInterrupt:
    print("interrupted")
```

Run it:

```bash
$PY user_train.py
```

If you want pause/stop control (e.g. for SIGTERM handling), wrap it in
`BackgroundTrainingThread` — but the thread is **not** required without a
GUI:

```python
import signal
from visergui.training import BackgroundTrainingThread

trainer = MyTrainer()
control = BackgroundTrainingThread(trainer.step)
signal.signal(signal.SIGTERM, lambda *_: control.stop())
control.start(); control.resume()
control._thread.join()
```

### 3. Trainer + viewer (live training visualization)

The viser server owns the main thread, so training has to run in a
background thread here:

```python
# user_gui_train.py — fully owned by you
from visergui.viewer import ViewerApp, SceneState
from visergui.training import BackgroundTrainingThread

class MyTrainer:
    def __init__(self, scene: SceneState):
        self.scene = scene
        self._step = 0

    def step(self) -> float:
        with self.scene.write() as s:
            # forward / backward / optimizer / densify — your code
            # mutates s.means / s.quats / s.scales / s.opacities / s.sh
            loss = ...
            self._step += 1
            s.record_step(self._step, float(loss))
            return float(loss)

scene = SceneState()
trainer = MyTrainer(scene)
control = BackgroundTrainingThread(trainer.step)

ViewerApp(
    ply_path="initial.ply",
    scene=scene,
    training_control=control,
).run()
```

Run it:

```bash
$PY user_gui_train.py
```

A "Training" folder appears in the GUI with `resume`/`pause` buttons,
live `step` / `loss` / `splat_count` readouts, and a plotly loss chart
updated at 2 Hz.

## Remote use (SSH tunnel)

```bash
# On the GPU box:
$PY visergui/viewer.py path/to/scene.ply --port 8080

# On your laptop:
ssh -L 8080:localhost:8080 user@gpubox

# In your browser:
# http://localhost:8080
```

## GUI cheatsheet

- **Display** — `splats` ↔ `points`. Switching to `points` clears the
  gsplat backdrop and pushes per-layer point clouds; switching back
  removes them and re-rasterizes.
- **Render** — SH degree pin, RGB/Depth, FOV, near/far for depth
  normalization.
- **Performance** — render-resolution cap, FPS / render_ms readouts,
  adaptive-resolution toggle (renders at lower res during motion, snaps
  to full res ~150 ms after stopping), motion thresholds.
- **Scene** — splat count, live camera pos/look-dir, reset-camera button.
- **Point Clouds** — global size multiplier + per-layer subfolders
  (visible / point_size / color_mode / uniform_color / count).
- **Training** (only when a `training_control` is attached) — pause /
  resume / status / step / loss / splat_count / loss plot.

## Verification

Quick checks after edits:

```bash
# Camera-math unit tests
cd visergui && $PY test_camera.py

# Headless training driver smoke (no viser server)
$PY -c '
from visergui.training import BackgroundTrainingThread, TrainingControl
import time

class Fake:
    def __init__(self): self.n = 0
    def step(self): self.n += 1; return 1.0/self.n

t = Fake()
ctl = BackgroundTrainingThread(t.step)
assert isinstance(ctl, TrainingControl)
ctl.start(); ctl.resume(); time.sleep(0.2); ctl.pause()
assert ctl.status() == "paused"
ctl.stop()
print(f"ok, took {t.n} steps")
'
```

## File layout

```
visergui/
├── viewer.py        # SceneState, Renderer, ViewerApp, GUI, CLI
├── training.py      # TrainingControl Protocol + BackgroundTrainingThread (stdlib only)
├── test_camera.py   # hand-computed viewmat assertions
├── plan/            # phase-by-phase implementation plan documents
└── README.md        # this file
```

## Coupling guarantees

- `training.py` has zero dependencies on `viewer.py`. It's stdlib-only —
  importable from any headless context.
- `viewer.py` imports only the `TrainingControl` Protocol from
  `training.py`. It never imports a concrete trainer.
- Your trainer file is yours: rewrite the loss, optimizer, densification,
  dataset — none of those changes touch `viewer.py` or `training.py`.
