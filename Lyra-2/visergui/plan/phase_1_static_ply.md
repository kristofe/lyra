# Phase 1 — Static .ply rendering

Replace Phase 0's synthetic renderer with real `gsplat.rasterization` against a
loaded `.ply` file. No new GUI controls; the goal is a correctly oriented,
stable render of a known-good scene.

## Prerequisites

Phase 0 complete and working. `Renderer.render()` is the only place that
produces frames. `SceneState` exists with a lock-protected `read()`.

## What changes

Add:

- `PlyLoader` — parses standard 3DGS `.ply` into torch CUDA tensors.
- `SceneState` now holds the splat tensors (means, quats, scales, opacities, sh).
- `Renderer.render()` body becomes a real rasterization call.
- A coordinate conversion: viser camera → OpenCV `viewmat` for gsplat.

Do NOT touch:

- The viser server setup, the GUI structure, the SSH port-forward comment.
- The `Renderer` / `SceneState` / `ViewerApp` class boundaries.

## Hard requirements

1. **gsplat, not Inria's `diff-gaussian-rasterization`.** The Inria rasterizer
   is research-license only and is blocked. gsplat is Apache 2.0.

2. **OpenCV view matrix convention.** gsplat expects world→camera in OpenCV
   convention: x-right, y-down, z-forward. Viser gives camera-to-world rotation
   as a wxyz quaternion in its own convention. The conversion is the most
   error-prone piece of this phase — write it in a single, named function
   `viser_camera_to_opencv_viewmat(position, wxyz) -> (4,4) np.ndarray` and
   unit-test it on a hand-computed pose before wiring it to the renderer.

3. **Inria-format .ply files load upside-down by default.** They're in COLMAP
   convention; in viser's y-up world they appear flipped. Apply a 180° rotation
   about X at load time:
   - `means[:, 1:] *= -1` (negate y and z)
   - rotate quaternions by the same X-180 rotation (compose with
     `q_flip = (0, 1, 0, 0)` in wxyz)
   Expose `--no-flip` CLI flag for .ply files that don't need it.

4. **Inference-only rasterization.** Wrap the `rasterization()` call in
   `torch.inference_mode()`. No backward buffers ever allocated.

5. **Output is `(H, W, 3) uint8` numpy on CPU.** Single `.cpu().numpy()` at the
   end of `render()`. Everything upstream stays on GPU.

## .ply field unpacking (standard 3DGS export)

Fields and transforms:

- `x, y, z` → `means` `(N, 3)` float32
- `nx, ny, nz` → ignore
- `opacity` (logit) → `sigmoid` → `opacities` `(N,)` float32
- `scale_0, scale_1, scale_2` (log) → `exp` → `scales` `(N, 3)` float32
- `rot_0..rot_3` (wxyz) → normalize → `quats` `(N, 4)` float32
- `f_dc_0, f_dc_1, f_dc_2` → DC SH coefficients
- `f_rest_*` → higher-order SH coefficients

SH layout for gsplat: `colors` is `(N, K, 3)` where `K = (sh_degree + 1)**2`.
The `.ply` stores rest coefficients in a flattened layout that needs reshaping
and permuting. Standard form:

```
rest = stack(f_rest_*, axis=-1).reshape(N, 3, K-1)  # channels-first
rest = rest.transpose(0, 2, 1)                       # -> (N, K-1, 3)
sh   = concat([dc[:, None, :], rest], axis=1)        # (N, K, 3)
```

Verify by checking that K matches the count of `f_rest_*` fields:
`K - 1 = num_rest_fields / 3`.

## Camera conversion (the math worth getting right)

Viser's `client.camera`:
- `position`: `(3,)` world-space camera position
- `wxyz`: `(4,)` camera-to-world rotation (viser convention)
- `fov`: vertical FOV in radians
- `aspect`: width / height

Procedure:

1. Convert `wxyz` to a 3×3 rotation matrix `R_c2w_viser`.
2. Convert viser's camera convention to OpenCV's. Viser uses y-up,
   z-back (looking down -z, like OpenGL). OpenCV uses y-down, z-forward.
   Multiply by a fixed basis change: flip y and z axes of the camera frame.
   `R_c2w_cv = R_c2w_viser @ diag(1, -1, -1)`.
3. Invert to world→camera: `R_w2c = R_c2w_cv.T`, `t_w2c = -R_w2c @ position`.
4. Assemble the 4×4 viewmat.

Intrinsics for the requested `(W, H)`:
```
fy = 0.5 * H / tan(0.5 * fov)
fx = fy  # square pixels; use aspect only for window shape, not fx vs fy
cx, cy = W * 0.5, H * 0.5
```

**Sanity test before declaring victory:** load a scene that has a clear
ground plane and a clear "front." Sit at `(0, 1, 3)` looking at origin.
The ground is below you and forward motion (W in viser's controls) should
move you toward the scene, not away from it or sideways. If parallax goes
the wrong way during strafe, you have a basis flip.

## SceneState evolution

```python
class SceneState:
    def __init__(self):
        self._lock = threading.RLock()
        self.means = None
        self.quats = None
        self.scales = None
        self.opacities = None
        self.sh = None
        self.sh_degree = 0
        self.num_splats = 0

    @contextmanager
    def read(self):
        with self._lock:
            yield self

    def load_from_ply(self, path, flip_x=True):
        # parse, transform, atomically swap into self under self._lock
        ...
```

The `load_from_ply` writes happen under the lock so Phase 5's training thread
never sees a half-written state.

## CLI

```
python viewer.py path/to/scene.ply [--no-flip] [--host 0.0.0.0] [--port 8080]
```

Phase 0's debug GUI controls are removed in this phase. No new controls added —
that's Phase 2.

## Gotchas to preempt

- gsplat 1.x's unified entry is `gsplat.rasterization(...)`. Older tutorials
  show `project_gaussians` + `rasterize_gaussians` — that's pre-1.0, ignore it.
- `rasterization()` returns `(rgb, alpha, info)` with rgb shape `(B, H, W, 3)`,
  batched on the first axis. We render one camera at a time, so squeeze the
  batch dim.
- Pass `viewmat` as `(1, 4, 4)` float32 CUDA tensor and `K` as `(1, 3, 3)`
  float32 CUDA tensor.
- `sh_degree=3` requires K=16 SH coefficients per splat. If your .ply has fewer,
  pass the matching degree (or pad with zeros). Read the actual count from the
  file and set `sh_degree` accordingly at load.
- Quaternion normalization: do it once at load, not per frame.
- Don't apply the X-flip to `scales` (they're per-axis magnitudes; flipping
  them is incorrect). Means and quats only.

## Done means

- `python viewer.py some_scene.ply` shows the scene in the browser, right-side-up.
- Camera flying with viser's default controls feels correct: forward goes forward,
  strafe parallax is in the right direction.
- Loading a non-Inria .ply with `--no-flip` also renders correctly.
- No `.cpu()` / `.numpy()` calls in `render()` outside the final return.
- `viser_camera_to_opencv_viewmat` has a unit test (separate `test_camera.py`)
  that checks a hand-computed case.
- Closing the browser tab and reopening it works; the server stays up.
- `SceneState` swap on load happens under the lock.

## Out of scope

- GUI controls beyond what viser provides by default
- SH degree switching at runtime (Phase 2)
- Depth rendering (Phase 2)
- Adaptive resolution (Phase 3)
- Multiple .ply files, scene switching (Phase 4)
