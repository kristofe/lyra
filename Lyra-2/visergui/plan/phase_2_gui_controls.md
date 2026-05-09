# Phase 2 — GUI controls

Add the inspection and rendering controls. Match the feature set of the native
ImGui plan, expressed in viser's GUI primitives.

## Prerequisites

Phase 1 complete. A scene loads, renders correctly, navigates with viser's
default camera controls. `Renderer.render()` is the single frame entry point.

## What changes

Add GUI handles to `ViewerApp`. Read their `.value` inside `render()` and pass
into the rasterization call. Add a depth-mode visualization path. Nothing else
about the architecture changes.

## Hard requirements

1. **GUI handles are typed attributes on `ViewerApp`**, not stored in a dict.
   Direct attribute access reads better and gives the type checker something
   to work with: `self.gui_sh_degree.value` not `self.gui["sh_degree"].value`.

2. **Reads happen at the top of `render()`**, snapshotted into local variables
   before any work. The render callback runs on viser's websocket thread; GUI
   updates from a different thread are concurrent. Snapshotting avoids tearing
   if a slider moves mid-frame.

3. **Depth mode is a render path, not a post-process.** Use
   `render_mode="RGB+ED"` in gsplat to get expected depth; visualize with a
   torch-side turbo colormap, not a numpy/matplotlib one (stay on GPU until
   the final `.cpu().numpy()`).

4. **Reset camera works through viser's API.** Set `client.camera.position` and
   `client.camera.wxyz` directly. Don't try to maintain a separate camera
   state — viser owns it.

## GUI structure

Three folders, in this order:

### Folder: "Render"

- `gui_sh_degree`: dropdown, options `["0", "1", "2", "3"]`, default = scene's
  loaded sh degree. Cap at the loaded degree (don't offer values higher than
  the .ply supports). Passed to `rasterization(sh_degree=...)`.
- `gui_render_mode`: dropdown, options `["RGB", "Depth"]`, default `"RGB"`.
  Selects between visualizing rgb and depth from the same rasterization call.
- `gui_fov`: slider, 30°–110°, default 60°. **Note:** viser owns the camera
  including FOV. To set FOV, write to `client.camera.fov` (in radians) on
  every connected client. Hook this on `gui_fov.on_update`.
- `gui_near` / `gui_far`: sliders, used only for depth normalization range.
  Defaults: 0.1, 100.0. Log-scale sliders are nicer here than linear.

### Folder: "Performance"

- `gui_max_res`: slider, 256–2048, default 1080. Caps the render resolution
  regardless of what viser requests. Useful for keeping framerate sane on
  4K browser windows.
- `gui_fps_readout`: read-only number display, updated each frame with a
  smoothed FPS (EMA, alpha = 0.1).
- `gui_render_ms_readout`: read-only number display, smoothed render time
  in milliseconds.

(Adaptive resolution toggles are Phase 3. Don't add them here.)

### Folder: "Scene"

- `gui_splat_count_readout`: read-only number, set once at load.
- `gui_camera_readout`: read-only multi-line text, updated each frame with
  position (3 floats) and look-direction (3 floats, derived from `wxyz`).
  Truncate to 3 decimal places.
- `gui_reset_camera`: button. On click, write the load-time pose back to
  every connected client.

## Depth visualization

Stay on GPU:

```python
# inside render(), depth path
rgb, alpha, info = rasterization(
    means, quats, scales, opacities, sh,
    viewmat, K, W, H,
    sh_degree=sh_degree_local,
    render_mode="RGB+ED",
)
# info["depth"] or the relevant key — verify in gsplat's docs for your version
depth = info["depth"][0]                    # (H, W) on GPU
near, far = gui_near_local, gui_far_local
d_norm = ((depth - near) / (far - near)).clamp(0, 1)
img_gpu = turbo_colormap(d_norm)            # (H, W, 3) float on GPU
img_u8  = (img_gpu * 255).to(torch.uint8)
return img_u8.cpu().numpy()
```

Implement `turbo_colormap` as a torch function: a small lookup table
`(256, 3)` on GPU, indexed by `(d_norm * 255).long()`. ~10 lines.

## Reset camera implementation

At Phase 1 load time, capture the initial pose viser puts the camera in
(or compute one from the scene bounding box — both are reasonable). Store
on `ViewerApp` as `self.home_position`, `self.home_wxyz`. Reset button
iterates `server.get_clients()` and writes those values back.

If you want a "frame the scene" button instead of a fixed home pose, compute
the bounding box of `means` once at load, place the camera at
`center + diag * (0, 0.3, 1.0) * 1.5` looking at center. Either is fine;
pick one.

## FPS / timing

Use a per-frame timestamp deque (length ~30). EMA smoothing is fine too.
Time the render call itself separately from the frame interval — the
difference is "how much headroom the GPU has."

```python
t0 = time.perf_counter()
img = ... rasterize ...
t1 = time.perf_counter()
self._render_ms_ema = 0.9 * self._render_ms_ema + 0.1 * (t1 - t0) * 1000
```

Push to the readouts via `.value = ...` (viser handles update the GUI).
**Don't update GUI handles from inside `render()` on every frame** if it causes
chatter — throttle to once every ~10 frames or use `time.perf_counter()` to
gate at 10 Hz. Excessive GUI updates can saturate the websocket.

## Gotchas

- Dropdown values are strings. Cast: `int(self.gui_sh_degree.value)`.
- Some gsplat versions return depth in a different key; check the version
  installed and look at `info`'s keys once at startup.
- `client.camera.fov` setter exists in viser ≥ 0.2; older versions may need
  workarounds. If the version available doesn't support it, hide the FOV
  slider and document the version requirement at the top of the file.
- The "look direction" for the camera readout is the third column of the
  rotation matrix derived from `wxyz`, sign-flipped depending on convention.
  Verify against the camera position by checking that
  `position + look_dir * 1.0` is "in front" when you're staring at a known point.

## Done means

- All controls from the three folders are present and live.
- SH degree changes are visibly different (degree 0 looks flat, degree 3 has
  view-dependent highlights).
- Depth mode shows a sensible turbo-colored depth map; near/far sliders
  rescale the colormap as expected.
- FOV slider changes the rendered view.
- FPS and render-ms readouts update at 10 Hz, not every frame.
- Reset camera returns to a sensible home pose.
- No GUI updates happen from outside the websocket thread without going
  through `.value =` (viser handles the threading internally for that).

## Out of scope

- Adaptive resolution (Phase 3)
- Multiple scenes / scene switching (Phase 4)
- Bookmarks, screenshots, click-to-inspect, crop box (Phase 4)
- Live training (Phase 5)
