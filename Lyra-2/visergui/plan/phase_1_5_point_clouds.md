# Phase 1.5 — Point cloud rendering (gap inspection)

Add native viser point-cloud rendering alongside the splat path. The goal is
**finding gaps in coverage** from video-model point clouds — so the
visualization must preserve holes, not hide them.

This is rendered client-side by Three.js (viser's `add_point_cloud`), not
server-side by gsplat. Different code path from Phase 1, runs in parallel
to it.

## Prerequisites

Phase 1 complete. `SceneState` exists with the `RLock`-protected `read()` /
`write()` pattern. The .ply loader works. `Renderer.render()` is the single
splat-render entry point.

## Why a separate render path

Gap inspection has different requirements from splat rendering:

- **Splat rasterization is integrative**: many overlapping translucent
  Gaussians blend into a smooth surface. Holes get partially filled by
  neighboring splats. This is what you want for view synthesis; it's the
  opposite of what you want for finding gaps.
- **Point clouds are discrete**: each point is one sample. Holes stay holes.
  Coverage is visible.

So: don't try to render points through gsplat with tiny scales. Use the
client-side point primitive, which draws as 1-pixel (or sized) GL points
with no blending.

## What changes

Add to `SceneState`:

- A list of named *point cloud layers*. Each layer has positions, colors,
  optional per-point metadata (confidence, source camera index, etc.),
  and a visible flag.
- Layers are independent of the splat tensors; both can be loaded
  simultaneously and shown together.

Add to `ViewerApp`:

- A "Point Clouds" GUI folder with controls for adding, hiding, recoloring,
  and resizing layers.
- Logic to push point cloud data to viser via `server.scene.add_point_cloud`
  on changes (not every frame).

Do NOT touch:

- `Renderer.render()` — point clouds don't go through it. Viser draws them
  client-side; the splat rasterizer keeps doing what it does.
- The viser camera plumbing.

## Hard requirements

1. **Point clouds are pushed to viser via `server.scene.add_point_cloud`,
   not rasterized.** If you find yourself feeding point cloud positions
   into `gsplat.rasterization`, stop — that's the wrong path for this
   phase.

2. **Updates are event-driven, not per-frame.** `add_point_cloud` is
   *idempotent on name* — calling it again with the same name replaces the
   data. But re-pushing N million points on every frame will saturate the
   websocket. Push only when:
   - A point cloud is loaded
   - Visibility toggles on (push) / off (`server.scene.remove_by_name`)
   - Color mode changes (re-derive colors, push)
   - Point size changes (push, since size is per-pointcloud in viser)

3. **Coexists with the splat scene.** Show point cloud and splats together
   without conflict. Three.js renders them in the browser's local 3D scene;
   the gsplat output is a 2D image background. They're composited
   client-side.

   **Important consequence:** points have correct 3D depth against each
   other and against any client-side Three.js geometry, but they do NOT
   have correct depth against the gsplat-rendered background. The
   server-rasterized image is just a backdrop. For your use case (showing
   point cloud over a splat scene to find gaps), this is usually fine —
   you want the points to be visible *over* the splats, not occluded by
   them. If you ever need true depth integration, that's a deeper change
   (would require pushing depth from gsplat back to the client).

4. **Memory budget on the client side.** Three.js comfortably handles
   ~500k–1M points; beyond that, browser performance degrades fast. Add
   a downsample-on-load option for huge clouds (uniform random subsample
   to a configurable cap, default 1M).

## SceneState additions

```python
@dataclass
class PointCloudLayer:
    name: str
    points: np.ndarray         # (N, 3) float32 on CPU
    colors_rgb: np.ndarray     # (N, 3) uint8, "natural" RGB from source
    metadata: dict             # e.g. {"confidence": (N,), "source_cam": (N,)}
    visible: bool = True
    point_size: float = 0.01   # world units
    color_mode: str = "rgb"    # "rgb" | "axis" | "confidence" | "uniform"
    uniform_color: tuple = (1.0, 0.2, 0.2)

class SceneState:
    # ... existing fields ...
    point_clouds: dict[str, PointCloudLayer] = field(default_factory=dict)

    def add_point_cloud(self, layer: PointCloudLayer):
        with self._lock:
            self.point_clouds[layer.name] = layer

    def remove_point_cloud(self, name: str):
        with self._lock:
            self.point_clouds.pop(name, None)
```

Points/colors live on CPU as numpy. Viser's API takes numpy arrays and
serializes them to the client; there's no benefit to keeping them on GPU
since they don't pass through gsplat.

If you want GPU-side color computation (e.g. confidence-based recoloring
over a million points), do it on GPU and `.cpu().numpy()` once before
pushing. Don't keep two copies.

## Loaders

Support multiple input formats. At minimum:

- **`.ply` point clouds** (different from 3DGS .ply — these have just
  `x,y,z` and optionally `red,green,blue`). Use `plyfile`. If RGB is absent,
  fall back to a uniform color or per-axis coloring.
- **`.npy`** arrays of shape `(N, 3)` for positions, optional companion
  `.colors.npy` for `(N, 3)` uint8.
- **`.npz`** archives with arrays named `points`, `colors`, optional
  `confidence`.

CLI:

```
python viewer.py scene.ply --points raw_video_output.ply --points completed.npz
```

Multiple `--points` accumulates layers. Each gets a default name from its
filename basename.

## Color modes

Implement four. They're how you'll actually inspect gaps.

### `rgb`
Use the layer's `colors_rgb`. Default. What the source provided.

### `axis`
Color by world position: R = normalize(x), G = normalize(y), B = normalize(z),
each scaled to layer bounding box. Useful for spotting structural patterns
without source RGB confusing things.

### `confidence`
If the layer has `metadata["confidence"]` (shape `(N,)` float in `[0, 1]`),
colorize via turbo colormap. Low confidence = red, high = blue. Directly
shows where the model was unsure — likely correlates with where gaps will
appear after thresholding.

### `uniform`
Single color (from `uniform_color`). Use this when overlaying a "raw output"
cloud in red over a "completed" cloud in green, so you can immediately see
what was added by completion vs. what was already there.

Color computation lives in a single function:

```python
def compute_colors(layer: PointCloudLayer) -> np.ndarray:
    """Returns (N, 3) uint8 for viser."""
    if layer.color_mode == "rgb":
        return layer.colors_rgb
    if layer.color_mode == "axis":
        p = layer.points
        mn, mx = p.min(0), p.max(0)
        rng = np.maximum(mx - mn, 1e-6)
        return ((p - mn) / rng * 255).astype(np.uint8)
    if layer.color_mode == "confidence":
        c = layer.metadata.get("confidence")
        if c is None:
            return layer.colors_rgb  # fallback
        return turbo_lut[(c.clip(0, 1) * 255).astype(np.int64)]
    if layer.color_mode == "uniform":
        return np.tile(np.array(layer.uniform_color) * 255, (len(layer.points), 1)).astype(np.uint8)
    raise ValueError(layer.color_mode)
```

`turbo_lut` is a `(256, 3)` uint8 array — same data as the GPU LUT in Phase 2,
just on CPU.

## GUI: "Point Clouds" folder

For each loaded layer, a sub-folder named `layer.name`:

- `gui_visible_<name>`: checkbox.
- `gui_size_<name>`: slider, 0.001–0.1 (world units), log-scale ideal.
  Default 0.01.
- `gui_color_mode_<name>`: dropdown over the four modes.
- `gui_uniform_color_<name>`: color picker (viser's `add_rgb`), only relevant
  when color mode is "uniform" — but always present, easier than dynamic UI.
- `gui_count_<name>`: read-only number, point count after any subsampling.

Plus, at the folder root:

- `gui_global_size_mult`: slider 0.1–10, multiplies all layers' point sizes.
  Useful for quick zoom-fitting.
- `gui_load_points`: file path text input + button to load another layer
  at runtime.

When any control changes, the corresponding `add_point_cloud` call to
viser is re-issued for that layer with new args. Don't touch other layers.

## The actual gap-finding workflow

This is what the viewer enables. Worth documenting in the README so users
remember why this phase exists:

1. Load the splat scene (Phase 1 path).
2. Load the raw video-model point cloud as a layer, color mode = "uniform"
   red.
3. Load the structure-completion output as a second layer, color mode =
   "uniform" green.
4. Toggle layer visibility to see what completion added.
5. Switch to "confidence" mode on the raw layer to see *where* the model was
   unsure — usually correlates with where the completion had to fill in.
6. Use the splat scene as visual context for *what surface* the gaps belong
   to (e.g. "all the gaps are on the back side of the chair" → coverage is
   a function of capture trajectory, not model failure).

Two-layer overlay with uniform colors is the highest-signal workflow for
"what did completion change?" Confidence coloring is the highest-signal for
"where will completion be needed?"

## Gotchas

- `add_point_cloud` is named — calling it twice with the same name replaces
  data. Calling it with a *new* name adds a second cloud. Track names
  carefully to avoid leaking layers.
- `remove_by_name` removes from the scene; the data is also gone from the
  client. Re-adding is a fresh push.
- Three.js point primitives don't have anti-aliasing; very small `point_size`
  values (< 0.001 in typical scene scales) render as flickering single pixels.
  Clamp the slider min sensibly.
- Subsample uniformly with a fixed seed when downsampling on load.
  Reproducibility matters when you're comparing two runs.
- If the source RGB is in BGR order (some pipelines export this way),
  swap channels at load. A startup print of the first 3 RGB values plus
  a visual sanity check on a known scene catches this.
- Point cloud .ply files often have `red, green, blue` as `uchar`. Some have
  them as `float` in [0, 1]. Detect and convert at load.

## Done means

- `python viewer.py scene.ply --points raw.ply --points completed.npz`
  shows all three: splats rendered, raw points as one layer, completed
  points as another.
- Toggling visibility hides/shows each layer independently.
- Color mode swap is instant and visually correct.
- Confidence coloring works on a layer that has confidence metadata.
- The two-layer uniform-color overlay (red raw / green completed) makes
  the "what was added by completion" signal obvious.
- Layers persist correctly across browser disconnect/reconnect.
- 1M points renders interactively (≥30 fps in the browser on your dev
  machine).

## Out of scope

- Rendering points through the gsplat rasterizer (wrong tool for this job)
- True depth integration between server-rasterized splats and client-side
  points (deeper architectural change; not needed for gap inspection)
- Per-point selection / picking (Phase 4 click-to-inspect can be extended
  to point clouds if needed)
- Surface reconstruction visualization (e.g. showing a mesh derived from
  the points) — separate concern, separate phase if needed
