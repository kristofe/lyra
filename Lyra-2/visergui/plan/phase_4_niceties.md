# Phase 4 — Niceties

Quality-of-life features that make the viewer pleasant to use day-to-day.
None of these are required for inspection; pick the ones that match your
actual workflow and skip the rest.

## Prerequisites

Phases 0–3 complete. The viewer renders correctly, has GUI controls, and
performs well on heavy scenes.

## Suggested order

Implement these in roughly this order — each one builds on Phase 0–3 only,
so they're independent and you can stop at any point.

1. Multiple scene loading (highest daily-use value)
2. Camera bookmarks
3. Screenshot button
4. Click-to-inspect
5. Crop box

## 1. Multiple scene loading

Load several `.ply` files at startup; switch between them via dropdown.

### Changes

- CLI accepts multiple paths: `python viewer.py scene_a.ply scene_b.ply ...`
  or a glob.
- `SceneState` learns to swap which set of tensors is "active." Two patterns
  work; pick one:
  - **Eager**: load all .ply files into VRAM at startup, dropdown switches a
    pointer. Fast switching, high VRAM cost.
  - **Lazy**: load on dropdown change, free the previous one. Slower switching,
    bounded VRAM.
  - For typical research use (2–5 scenes), eager is fine.
- New "Scene" folder dropdown: `gui_scene_select`, options = list of loaded
  scene names (basename of each path). On change, swap active scene under
  the lock.
- The reset-camera home pose recomputes when the scene changes (or store
  per-scene home poses).

### Hard requirements

- Scene swap happens under `SceneState._lock`. A render in flight on the
  old scene completes; the next render uses the new scene.
- Splat-count readout updates on swap.
- Adaptive resolution motion tracker resets on swap (otherwise the first
  frame after swap may register spurious "moving" from the previous scene's
  last camera state — actually fine, but a clean reset is nicer).

## 2. Camera bookmarks

Save/restore camera poses by name. Useful for comparing the same viewpoint
across scenes or training checkpoints.

### Changes

- Add a "Bookmarks" folder.
- `gui_bookmark_name`: text input.
- `gui_bookmark_save`: button. On click, capture `client.camera.position`
  and `client.camera.wxyz` for the first connected client, store in a dict
  keyed by name, and update a dropdown of saved bookmarks.
- `gui_bookmark_select`: dropdown. On change, write the saved pose to all
  clients.
- `gui_bookmark_delete`: button. Removes the currently selected bookmark.
- Persist bookmarks to a JSON file next to the .ply (or in
  `~/.config/splat_viewer/bookmarks.json`, keyed by .ply path's hash).

### Hard requirements

- Bookmarks survive viewer restart (file persistence).
- Bookmark dropdown updates immediately when one is added or removed —
  viser dropdowns can be reconstructed by replacing `.options`; check the
  installed viser version's API.

## 3. Screenshot button

Render at higher resolution than the live view, save to disk.

### Changes

- "Render" folder gains:
  - `gui_screenshot_scale`: slider, 1.0–4.0, default 2.0.
  - `gui_screenshot_button`: button.
- On click: synchronously call `Renderer.render()` at
  `(target_W * scale, target_H * scale)`, bypass the adaptive-resolution path,
  save PNG to `./screenshots/<scene_name>_<ISO timestamp>.png`.
- Skip the adaptive-resolution clamp; respect only `gui_max_res` (or
  override that too — high-res screenshots are the whole point).

### Hard requirements

- Screenshot uses the exact current camera pose. Read from
  `client.camera`, not from any cached state.
- The PNG path is printed to stdout so it's findable when running over SSH.
- Filename includes scene basename + timestamp; no overwrites.

## 4. Click-to-inspect

Click in the viewport, get info about the splat under the cursor. Surprisingly
useful for debugging splat-quality issues.

### Changes

- Use viser's scene-click event. Viser exposes click rays
  (`event.ray_origin`, `event.ray_direction` in world space).
- For each click: do a nearest-splat-to-ray query.
  - Project all `means` onto the ray: `t = dot(means - ray_origin, ray_direction)`.
  - Compute perpendicular distance: `||means - (ray_origin + t * ray_direction)||`.
  - Mask to `t > 0` (in front of camera).
  - Return the splat with the smallest perpendicular distance, weighted by
    splat scale (larger splats "win" ties).
- Display the result in a "Click" folder: position, scale, opacity, RGB
  (from DC SH), index.

### Performance note

A naïve `(N, 3)` projection per click for N up to a few million is fine
(maybe 50 ms on GPU). Don't overengineer with a BVH unless you measure it
actually being slow at click time.

### Hard requirements

- Click computation runs on GPU (it's a torch operation on existing tensors).
- The result panel updates within ~1 frame of the click.
- The selected splat is visualized somehow — easiest: viser's
  `add_point_cloud` with one point at the splat's position, in a contrasting
  color. Updates on each click.

## 5. Crop box

Interactive 3D box that masks splats outside it. Useful for isolating a
region for inspection or for eventual export.

### Changes

- Add a transform-controls gizmo via viser's `add_transform_controls`.
- A bounding-box mesh visualization tied to the gizmo's current transform.
- `gui_crop_enabled`: checkbox. When on, `Renderer.render()` filters
  `means` (and the corresponding `quats`, `scales`, `opacities`, `sh`) to
  those inside the box before passing to `rasterization()`.
- Filter is a single boolean mask computed each frame:
  ```python
  local = (means - box_center) @ R_box.T  # to box-local coords
  inside = ((local.abs() < box_half_extents).all(dim=-1))
  ```
  Then index every splat tensor with `inside`.

### Hard requirements

- Filtering is GPU-side, no CPU roundtrips.
- Rebuild the boolean mask each frame — the gizmo can move at any time.
- When `gui_crop_enabled` is off, filtering is skipped entirely (don't
  pay for an `inside.all()` you'll discard).
- The filtered count updates the splat-count readout when crop is active.

## What to skip

These came up in earlier brainstorming and aren't worth the complexity:

- **Tour mode / animated camera paths**: Real value is low. If you need a
  presentation, use OBS and capture the live viewer.
- **Multi-user collaborative viewing**: Viser supports multiple clients,
  but synchronizing GUI state across them is fiddly. The default behavior
  (each client has its own camera, shared scene) is good enough.
- **Custom shaders / post-effects**: Out of scope for an inspector.

## Done means

For each feature implemented:

- Multiple scenes: dropdown switches scenes; splat count updates; bookmarks
  still work across scenes (or are scene-local — pick one).
- Bookmarks: persist across restarts.
- Screenshot: PNG written, path printed, resolution actually higher than live.
- Click: clicking somewhere in the scene populates the inspect panel within
  a frame.
- Crop: gizmo is interactive, render shows only splats inside the box, count
  updates.

## Out of scope

- Live training (Phase 5)
