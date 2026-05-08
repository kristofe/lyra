Build a native-window 3D Gaussian Splatting viewer in Python with first-person
navigation and ImGui controls. Single file is fine; split if it gets past ~600 lines.

## Stack (do not substitute)

- `glfw` for window + input
- `moderngl` for the GL context and fullscreen-quad blit
- `imgui-bundle` (NOT `pyimgui`) with the GLFW backend
- `gsplat` (Apache 2.0) for rasterization — NOT `diff-gaussian-rasterization`
  (research license, blocked for our use)
- `cuda-python` for CUDA↔GL interop
- `torch` (CUDA), `numpy`, `plyfile`

## Architecture
```
viewer.py
  ├── FlyCamera         pose + intrinsics, OpenCV convention output
  ├── CameraController  GLFW input → camera, gated on ImGui capture flags
  ├── PlyLoader         standard 3DGS .ply → torch CUDA tensors
  ├── CudaGLBlitter     registers a GL texture with CUDA, copies a torch
  │                     (H,W,3) uint8 tensor into it via cudaMemcpy2DToArray,
  │                     no host roundtrip
  └── main loop         poll → update camera → rasterize → blit → imgui → swap
```

## Hard requirements

1. **Zero CPU roundtrip on the render path.** The torch tensor coming out of
   `gsplat.rasterization` must reach the GL texture via CUDA-GL interop
   (`cudaGraphicsGLRegisterImage` + `cudaGraphicsSubResourceGetMappedArray` +
   `cudaMemcpy2DToArray` with `cudaMemcpyDeviceToDevice`). If you find yourself
   calling `.cpu()` or `.numpy()` in the per-frame path, stop and fix it.

2. **Coordinate conventions.**
   - gsplat expects OpenCV view matrix (x-right, y-down, z-forward, world→cam).
   - Inria-format 3DGS `.ply` files are in COLMAP convention; in our GL world
     (y-up) they render upside-down. Apply a 180° rotation about X on load,
     and rotate the quaternions correspondingly. Expose a `--no-flip` CLI
     flag in case a given .ply doesn't need it.

3. **Camera.**
   - FPS-style: WASD = planar move along camera forward/right, Space/Ctrl = world up/down.
   - Right-mouse-drag to look; set `glfw.CURSOR_DISABLED` while held.
   - Shift = 5× boost. Scroll wheel = adjust base move speed (multiplicative).
   - Pitch clamped to ±89°, never ±90°.
   - Movement keys polled per-frame in `update(dt)`, not via key callbacks
     (callbacks fire on transitions only — holding W must keep moving).
   - All input handlers gate on `imgui.get_io().want_capture_mouse` /
     `want_capture_keyboard`.

4. **Adaptive resolution.** Track `camera.is_moving`. Render at 0.5× resolution
   while moving, 1.0× when idle. The blit shader scales to window size.

5. **Inference path.** Wrap rasterization in `torch.inference_mode()`. No
   gradient buffers ever allocated.

6. **HiDPI.** Use GLFW framebuffer size (not window size) for GL viewport,
   render resolution, and ImGui font scaling.

## ImGui panel ("Viewer")

- FPS readout (1 / smoothed dt)
- Splat count, current render resolution
- Slider: FOV (30°–110°)
- Slider: move speed (log scale, 0.1–50 m/s)
- Combo: SH degree (0/1/2/3) — passed to `rasterization(sh_degree=...)`
- Combo: render mode — RGB, depth (use `render_mode="RGB+ED"`, visualize
  depth with a turbo colormap, normalize per-frame to [near, far] of visible depths)
- Checkbox: adaptive resolution on/off
- Button: "Reset camera" (back to load-time pose)
- Camera readout: position, yaw°, pitch°

## CLI
```
python viewer.py path/to/scene.ply [--no-flip] [--width 1280] [--height 720]
```

## Gotchas to preempt (don't rediscover these)

- `imgui-bundle` enables docking via `imgui.ConfigFlags_.docking_enable`; turn it on.
- `glfw.swap_interval(1)` for vsync. Don't leave it at 0.
- The fullscreen quad needs `ctx.disable(moderngl.DEPTH_TEST)` before drawing
  or it'll z-fight with itself depending on default state.
- Register the CUDA-GL resource ONCE after texture creation, not per frame.
  Map/unmap each frame.
- If the window resizes, the GL texture and its CUDA registration must both
  be recreated. Handle this in the framebuffer-size callback.
- Standard 3DGS .ply fields: `x,y,z`, `nx,ny,nz` (ignore), `f_dc_0..2` (DC SH),
  `f_rest_*` (rest of SH, layout `(N, 3, (deg+1)^2 - 1)` after reshape, then
  permute to gsplat's `(N, K, 3)`), `opacity` (logit), `scale_0..2` (log),
  `rot_0..3` (wxyz). Apply `sigmoid(opacity)`, `exp(scale)`, normalize quats.

## Done means

- `python viewer.py some_scene.ply` opens a window, scene renders right-side-up.
- WASD + RMB-look navigates without snags or input bleeding into ImGui.
- FPS is materially higher with adaptive resolution on during motion.
- No `.cpu()` / `.numpy()` calls in the per-frame render path (grep proves it).
- Resizing the window doesn't crash and continues rendering correctly.
- Closing the window exits cleanly (no CUDA shutdown errors).

Start by getting a triangle on screen with the GL+ImGui scaffold, then the
CUDA-GL blit with a synthetic torch tensor (e.g. UV gradient), THEN wire in
gsplat. Don't try to do all three at once — debugging interop on top of a
broken rasterization call is miserable.