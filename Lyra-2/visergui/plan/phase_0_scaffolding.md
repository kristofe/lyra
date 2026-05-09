# Phase 0 — Scaffolding

Prove the viser render-callback loop works end-to-end with a synthetic image,
before introducing any splat code. The point of this phase is to have a known-good
foundation so that bugs in later phases can't be the GUI/transport layer.

## Stack (do not substitute)

- `viser` (Apache 2.0) — owns the window, camera, GUI, transport
- `torch` (CUDA), `numpy`
- `gsplat` (Apache 2.0) — installed but NOT used yet
- Python 3.10+

Designed with **future threading in mind** (Phase 5 will run training in a
background thread). Even in this phase, structure the scene state so that
swapping it under a lock is a one-line change later.

## Architecture

```
viewer.py
  ├── SceneState    holds splat tensors (None for now); RLock around swaps
  ├── Renderer      takes SceneState + camera + (W,H) -> (H,W,3) uint8
  │                 in Phase 0, returns a synthetic image
  ├── ViewerApp     owns viser server, GUI handles, render callback,
  │                 wires Renderer to viser's per-client camera
  └── main()        parses args, constructs the above, blocks on server
```

`SceneState` exists from day one even though it's empty. Phase 1 will fill it.

## Hard requirements

1. **Single render entry point.** `Renderer.render(scene, camera, W, H) -> np.ndarray`.
   No other code path produces frames. Phase 1 swaps the body; nothing else changes.

2. **Camera object is viser's, not ours.** Do NOT introduce a FlyCamera class.
   Viser owns the camera; we read `camera.position`, `camera.wxyz`, `camera.fov`,
   `camera.aspect` inside the render callback.

3. **Render resolution comes from viser**, not hard-coded. The callback receives
   a target resolution; honor it (Phase 3 will override selectively).

4. **Threading-ready state.** `SceneState` exposes a `read()` context manager
   that takes an `RLock`. The renderer must use it even though contention is zero
   in Phase 0. This is the seam Phase 5 plugs into.

5. **No `.cpu()` / `.numpy()` calls in the synthetic-render path that aren't
   the final return.** The output to viser must be a CPU numpy uint8 array
   `(H, W, 3)`, but everything upstream of the final conversion stays on GPU.
   This discipline matters in Phase 1 when real rasterization lands.

## Synthetic render

Generate an image that proves the camera is being read correctly:

- Encode camera yaw into the hue of a horizontal gradient, OR
- Render a checkerboard whose phase shifts with `camera.position`, OR
- Both.

The test is: move/rotate the camera in the browser, the image must change in a
way that's a function of the new camera state. If it doesn't, the camera plumbing
is wrong and Phase 1 will fail mysteriously.

Do this on the GPU with torch ops, then `.cpu().numpy()` once at the end. Sets
the pattern for Phase 1.

## GUI (throwaway, but wire it properly)

Three controls in a "Debug" folder, just to confirm the GUI loop works:

- Slider: "test_value" 0.0–1.0, drives the brightness of the synthetic image
- Dropdown: "test_mode" ["gradient", "checker"], switches the synthetic pattern
- Button: "log camera", prints current camera pose to stdout for the connected client

GUI handles live on `ViewerApp` as attributes. Don't put them in a dict.

## CLI

```
python viewer.py [--host 0.0.0.0] [--port 8080]
```

Default host `0.0.0.0` so it works on a remote box without thinking about it.

## Remote-box sanity check

Document in a top-of-file comment:

```
# Remote use:
#   On GPU box:  python viewer.py --port 8080
#   On laptop:   ssh -L 8080:localhost:8080 user@gpubox
#   Browser:     http://localhost:8080
```

## Gotchas to preempt

- Viser's `add_dropdown` returns a handle; read `.value`, don't reach into internals.
- `@handle.on_update` is the callback decorator. Keep callbacks fast — they run on
  the websocket thread.
- The render callback may be invoked from a different thread than `main()`.
  Treat all GUI handle reads as concurrent; `.value` reads are atomic, longer
  computations need their own snapshot at callback entry.
- Don't call `server.run()` and then expect to do work after — the call blocks.
  Background work goes in threads spawned before `.run()` (or use the non-blocking
  pattern from viser's docs).

## Done means

- `python viewer.py` starts a server, prints the URL.
- Browser connects, scene shows the synthetic image.
- Moving the camera changes the image in a way clearly tied to camera state.
- The three debug controls all work.
- `SceneState` exists with a lock-protected `read()` method, even though it
  holds nothing yet.
- Top-of-file comment shows the SSH port-forward command.
- Code structure matches the architecture diagram above (4 classes/functions,
  in that order, in one file).

## Out of scope (do not do these now)

- Loading any .ply file
- Calling `gsplat.rasterization`
- Any camera math beyond reading viser's camera object
- Adaptive resolution
- Multiple scenes, bookmarks, screenshots
- Threading anything (the lock is structural, not yet exercised)

## File layout

Single file `viewer.py` for Phase 0. Phase 1 may split if it grows past ~400 lines.
