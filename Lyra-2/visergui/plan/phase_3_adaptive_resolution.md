# Phase 3 — Adaptive resolution

Make navigation feel smooth on heavy scenes by rendering at lower resolution
during motion and full resolution at idle. The "SIBR trick" — barely
perceptible during motion, crisp on still frames.

## Prerequisites

Phases 0–2 complete. FPS and render-ms readouts are working (you'll need them
to measure whether this phase is helping).

## First: measure before building

Viser already lowers resolution during interaction by default. **Before writing
any code in this phase, measure on your actual scenes:**

1. Load a scene representative of your target use case (large reconstruction,
   not a toy).
2. Fly through it, watch the FPS readout.
3. Note: is motion noticeably choppier than idle? Is the FPS during motion
   already ≥ 30?

If viser's built-in behavior is already good, skip this phase entirely.
Document what you measured and move on.

If not — large scenes that drag during motion — proceed.

## What changes

Add motion detection to `Renderer` (or a small `MotionTracker` helper).
Modify `render()` to pick a resolution based on motion state. Add GUI
toggles to the Performance folder. No other architecture changes.

## Hard requirements

1. **Motion detection lives in the renderer**, not in the GUI layer. The
   renderer is the only thing that sees every camera state per frame; that's
   the right place to track deltas.

2. **Hysteresis on the idle→full-res transition.** A single still frame
   shouldn't trigger a full-res render — the user might be mid-gesture.
   Require N consecutive still frames (e.g. N=5) before declaring idle.
   The reverse (full-res → moving → low-res) is immediate.

3. **Motion threshold is in world units, not pixels.** Translation delta in
   meters and rotation delta in radians. Pixel-space thresholds are scene-scale
   dependent and break on every new dataset.

4. **Low-res renders use the same aspect ratio.** Scale W and H by the same
   factor. Don't introduce stretch.

## Motion tracking

```python
class MotionTracker:
    def __init__(self, trans_thresh=1e-3, rot_thresh=1e-3, idle_frames=5):
        self.last_pos = None
        self.last_wxyz = None
        self.idle_count = 0
        self.trans_thresh = trans_thresh
        self.rot_thresh = rot_thresh
        self.idle_frames_required = idle_frames

    def update(self, position, wxyz) -> bool:
        """Returns True if camera is currently 'moving'."""
        if self.last_pos is None:
            self.last_pos, self.last_wxyz = position, wxyz
            return False
        d_trans = np.linalg.norm(position - self.last_pos)
        # Quaternion angular distance: 2 * acos(|q1 . q2|)
        dot = abs(float(np.dot(wxyz, self.last_wxyz)))
        d_rot = 2.0 * math.acos(min(1.0, dot))

        moving = d_trans > self.trans_thresh or d_rot > self.rot_thresh
        if moving:
            self.idle_count = 0
        else:
            self.idle_count += 1

        self.last_pos, self.last_wxyz = position, wxyz
        return self.idle_count < self.idle_frames_required
```

Tune `trans_thresh` per scene scale. A reasonable default for reconstructed
indoor scenes is `1e-3` meters; for large outdoor reconstructions it might
be `1e-2`.

## Resolution logic in render()

```python
def render(self, camera, target_W, target_H):
    moving = self.motion.update(camera.position, camera.wxyz)
    target_W = min(target_W, self.gui_max_res.value)
    target_H = min(target_H, self.gui_max_res.value)

    if self.gui_adaptive_res.value and moving:
        scale = self.gui_moving_scale.value
        W = max(64, int(target_W * scale))
        H = max(64, int(target_H * scale))
    else:
        W, H = target_W, target_H

    # ... rasterize at (W, H), return (H, W, 3) uint8 ...
```

Viser handles client-side upscaling — you don't need to upsample on the
server.

## GUI additions to "Performance" folder

- `gui_adaptive_res`: checkbox, default `True`.
- `gui_moving_scale`: slider, 0.25–1.0, default 0.5.
- `gui_idle_frames`: slider, 1–30, default 5. Updates `motion.idle_frames_required`
  on change.
- `gui_motion_state_readout`: read-only text, displays "moving" or "idle"
  based on the last frame's motion result. Useful for debugging the threshold.

## Tuning notes

- If "moving" flickers during slow camera motion: raise `idle_frames_required`
  or raise the thresholds.
- If full-res kicks in during a still hold but the user is still adjusting:
  raise `idle_frames_required`.
- If low-res persists for too long after stopping: lower `idle_frames_required`.
- If the resolution drop during motion is too visible: raise `gui_moving_scale`
  toward 0.75.

## Verification

The fact that motion-detection is helping should show up clearly:

- Toggle `gui_adaptive_res` off, fly fast through the scene. Note FPS.
- Toggle on. FPS during motion should rise materially (often 1.5–2× on heavy
  scenes); FPS at idle should be unchanged.
- Watch `gui_motion_state_readout`: should flip to "moving" the instant the
  camera starts moving and to "idle" within a fraction of a second of stopping.

## Gotchas

- Quaternion sign ambiguity: `q` and `-q` represent the same rotation but
  `dot(q1, q2)` flips sign. Use `abs(dot)` for angular distance, as in the
  code above.
- The first frame after connection has `last_pos = None`. The code above
  handles it by returning `False` and seeding state — verify the first
  frame doesn't trigger spurious "moving."
- If multiple clients connect, motion state is per-renderer (singleton),
  not per-client. That's fine for this viewer's use case (one user at a
  time, typically), but document it. Per-client motion tracking adds
  complexity for marginal benefit.
- `gui_idle_frames` change should update the tracker live. Hook the
  `.on_update` to write `self.motion.idle_frames_required = new_value`.

## Done means

- Adaptive resolution toggle measurably increases motion FPS on a heavy scene.
- Idle frames are crisp; the transition from low-res to full-res after a stop
  is barely perceptible.
- Motion-state readout flips correctly.
- All four new GUI controls work.
- Toggling adaptive res off makes the viewer behave exactly as Phase 2.

## Out of scope

- Per-client motion tracking
- Render-time-based fallback (e.g. "if last frame took >100ms, drop res
  more aggressively") — interesting but not necessary
- Anti-aliased downsampling on the client side (viser handles this)
