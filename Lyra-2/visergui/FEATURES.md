# Gaussian Splat Training — Feature Report

A high-level summary of what the `visergui/` stack does today: training,
diagnostics, meshing, baking, editing, and an interactive viewer for
Gaussian-splat reconstruction of monocular video.

---

## 1. Core training loop

The trainer fits a Gaussian-splat scene to RGB frames with explicit
control over what the loss is allowed to look at.

- **Per-parameter Adam optimizers** with separate LRs for means, scales,
  quats, opacities, sh0, shN — [splat_trainer.py](splat_trainer.py).
- **Masked L1 photometric loss** weighted by a per-pixel
  sky / confidence mask, plus optional LPIPS.
- **Void loss** on rendered alpha inside `~train_mask`: actively drives
  splats to render nothing in excluded regions, eliminating the
  asymmetric "free growth" at mask boundaries that plain masked-L1
  leaves untouched. Exposed as a `void_weight` slider in the Setup tab
  and a `--void-weight` CLI flag.
- **SH band warm-up** — one degree at a time, ramped over
  `sh_ramp_steps_per_band`.
- **Random per-step frame sampling**.

## 2. Initialization

Splats are seeded directly from the input video instead of starting
random, so the optimizer begins on the manifold.

- **RGBD unprojection** from VIPE intrinsics + DepthAnything-3 depth,
  followed by **voxel downsampling** to thin redundancy 3–10×.
- **Scale-per-texel heuristic** so each splat's init size matches its
  pixel footprint at depth; identity quats, opacity 0.9.
- **Sky / low-confidence masking** applied before unprojection so noisy
  pixels never become splats.

## 3. Densification & pruning

- **gsplat DefaultStrategy** clone / split / prune with mode-specific
  thresholds: aggressive 3DGS schedule (refine@200, densify every 50,
  opacity reset every 1500); 2DGS schedule with opacity reset disabled
  and a `gradient_2dgs` key.
- **Post-hoc pruning** by opacity, max scale, anisotropy, and KNN
  outliers.
- **Live per-axis scale clamp** keyed to the init voxel edge, adjustable
  from the GUI without restarting training.
- **"Prune Splats" GUI action** — pauses training, runs the post-hoc
  prune, reports before/after counts, and leaves training paused for
  inspection before resume.

## 4. 2DGS regularization

When the 2DGS rasterizer is selected:

- **Distortion loss** and **internal normal-consistency** (rendered
  normal vs. splat-surface normal), each gated by independent warmup
  step counts.
- **DA3 depth supervision** — L1 on rendered vs. DepthAnything-3 depth.
- **DA3 normal supervision** — cosine on depth-derived normals.
- **Rasterizer toggle** between `gsplat.rasterization` and
  `rasterization_2dgs` (RGB+ED), available as a CLI flag and as a Setup
  tab dropdown that applies on next Initialize.

## 5. Training diagnostics

Built so training-quality issues can be *seen* before the loss is
changed.

- **Mask diagnostic** ([splat_trainer.py](splat_trainer.py)) — for a
  handful of evenly-spaced frames, renders a 5-column panel:
  GT | GT+mask overlay (red on excluded) | render | render·mask (what
  L1 actually sees) | render·~mask "leak" (what splats are doing
  inside the filtered region). Saved as `{tag}_mask_diag.png`.
- **Splat-stats diagnostic** — histograms of max-axis scale (in voxel
  units, with the live clamp drawn as a reference line), opacity, and
  anisotropy / needle-ness, plus a printed top-10 table of the largest
  splats by index for quick inspection. Saved as
  `{tag}_splat_stats.png`.
- **Per-step loss breakdown** — `step()` publishes a per-term
  `components` dict (`l1`, `void`, `lpips`, `distortion`,
  `normal_consistency`, `depth_sup`, `da3_normal`) carrying each term's
  *weighted* contribution. Rendered compactly by `format_loss_components`
  in the tqdm postfix and in the Train-tab status panel.
- Diagnostics fire automatically at the end of `train_for()` via
  `SplatTrainer.render_diagnostics(tag)`; can also be invoked any time
  after `prepare_and_init`.

## 6. Mesh extraction

[mesher.py](mesher.py) supports two paths into a triangle mesh:

- **TSDF mode** — splat-rendered depth → soft-alpha TSDF fusion →
  marching cubes.
- **DLNR mode** — render a synthetic stereo pair, run DLNR for depth,
  fuse into a TSDF-with-features volume, sample per-vertex color.
- **DLNR depth cache** keyed on splat fingerprint so the `density` and
  `shell_thickness` sliders stay interactive without re-running the
  network.

## 7. Texture baking

Two baking strategies, both produce OBJ + MTL + PNG:

- **Per-vertex-color bake** — decimate → xatlas UV → barycentric splat
  of DLNR colors → gap-fill.
- **Photogrammetric splat-projection bake** — per-texel multi-camera
  weighted average using splat-rendered RGB plus a visibility test.
- Configurable `target_faces`, `tex_size`, and depth tolerance.

## 8. Editing & inpainting

- **Inpainter panel** ([inpainter.py](inpainter.py)) captures the live
  viser render and saves RGB / alpha / depth / camera together as the
  inpainting unit of work.
- **Disocclusion mask** derived from rendered alpha, with optional
  user-drawn rectangle and outpaint pad.
- **Multi-view neighbor selection** by camera proximity feeds reference
  frames to the model.
- Backends: **FLUX-Kontext** and **InstaInpaint**.
- **SAM 2 multi-view mask propagation** ([multiview_mask.py](multiview_mask.py))
  lifts a 2D mask into 3D via depth and re-projects it to every
  neighbor frame.

## 9. Viewer / GUI (viser)

[viewer.py](viewer.py) is the main interactive surface.

- **Tabbed control panel** — Setup, Train, Mesh, Inpaint.
- **Display modes** — splats / point cloud / mesh toggle.
- **Render modes** — RGB, turbo-colored Depth, and **Normals**
  (RGB+ED → `depth_to_normal` → `(n+1)/2` RGB, background blacked out
  via the alpha channel), each with adjustable near/far for depth-style
  outputs.
- **Per-client SH degree pin**.
- **Adaptive resolution** during pan/zoom to keep FPS up.
- **Live FPS + render-ms readouts**, **camera readout + home-pose
  reset**.
- **Live loss visualization** — log-scale multi-trace plot in the Train
  tab showing the total loss plus one trace per active component (warmup-
  gated terms appear with gaps before their start step; click legend
  entries to hide/show), plus a markdown breakdown line beside the
  step/loss readout that mirrors the tqdm postfix.

## 10. Camera handling

- **VIPE pose estimation** pipeline for monocular video → per-frame
  extrinsics + intrinsics.
- **wxyz ↔ matrix / OpenCV ↔ viser** conversions.
- **Fibonacci dome camera generator** for meshing.

## 11. I/O

- **Inria-style PLY save/load** with `f_dc` / `f_rest` SH fields and
  activation inversion.
- **Point-cloud load** for `.ply` / `.npz` / `.npy` with optional color
  and confidence channels.
- **Mesh export** — PLY (TSDF / DLNR) and OBJ+MTL+PNG (textured).
- **Diagnostic snapshots** via matplotlib at init time.

---

A few dozen distinct features across training, init, regularization,
diagnostics, meshing, baking, editing, viewer, and I/O.
