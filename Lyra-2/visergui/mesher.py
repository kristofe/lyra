"""
Mesh Gaussians using fvdb TSDF fusion (soft alpha) + DLNR stereo depth + texture baking.

No viser or trainer imports — can be called from GUI thread or CLI.
Requires fvdb-core 0.4.2+ and fvdb-reality-capture.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from pathlib import Path
from typing import Callable


# Module-level cache for DLNR depth maps. Keyed on a tuple that identifies the
# splat state + camera config + DLNR stereo params. The cached value is a list
# of (rgb_cpu, depth_cpu, weight_cpu, c2w_cpu, K_cpu) tuples — one per camera.
# Density / shell_thickness are NOT part of the key; they only affect the
# downstream TSDF integration which is cheap to rerun.
_DLNR_DEPTH_CACHE: dict[tuple, list] = {}


def generate_mesh(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    device: str = "cuda",
    mode: str = "tsdf",
    n_cams: int = 96,
    H: int = 1024,
    W: int = 1024,
    alpha_thresh: float = 0.5,
    density: float = 0.02,
    shell_thickness: float = 6.0,
    bake_texture: bool = False,
    tex_size: int = 1024,
    out_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    splat_fingerprint: int | None = None,
) -> dict:
    """
    Generate a mesh from Gaussian splats using fvdb TSDF or DLNR.

    Args:
        means, quats, scales, opacities: post-activation tensors from trainer
        sh: full SH tensor (N, K, 3) — DC + higher-order bands
        device: torch device
        mode: "tsdf" (fvdb soft-alpha TSDF) or "dlnr" (stereo depth w/ per-vertex colors)
        n_cams: number of Fibonacci-dome cameras
        H, W: render resolution per camera (1024x1024 default)
        alpha_thresh: drop pixels where alpha < this
        out_path: optional Path to save PLY/OBJ files
        progress_cb: optional callback(cam_idx, total_cams)

    Returns:
        dict with "verts", "faces", "normals" (TSDF) or "normals"=None (DLNR)
        and "colors" (None for TSDF, (V,3) float for DLNR)
    """
    # Scene bounds
    bbmin = means.min(0).values
    bbmax = means.max(0).values
    scene_diag = float((bbmax - bbmin).norm())
    scene_center = 0.5 * (bbmin + bbmax)

    # Build splat model + dome cameras via shared helpers
    model = _build_splat_model(means, quats, scales, opacities, sh)
    c2w, w2c, K = _build_dome_cameras(scene_center, scene_diag, n_cams, W, device)

    if mode == "tsdf":
        return _mesh_tsdf(
            model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
            alpha_thresh, density, shell_thickness, out_path, progress_cb
        )
    elif mode == "dlnr":
        return _mesh_dlnr(
            model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
            density, shell_thickness, bake_texture, tex_size, out_path, progress_cb,
            splat_fingerprint,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _mesh_tsdf(
    model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
    alpha_thresh, density, shell_thickness, out_path, progress_cb
):
    """TSDF fusion with soft alpha weighting (verbatim from notebook)."""
    import fvdb
    from fvdb import Grid

    TRUNCATION_MARGIN = density * scene_diag
    VOX_SIZE = TRUNCATION_MARGIN / shell_thickness
    NEAR = 0.05 * scene_diag
    FAR = 4.0 * scene_diag

    accum_grid = Grid.from_zero_voxels(device=device, voxel_size=VOX_SIZE, origin=0.0)
    tsdf = torch.zeros(accum_grid.num_voxels, device=device, dtype=torch.float16)
    weights = torch.zeros(accum_grid.num_voxels, device=device, dtype=torch.float16)

    for i in range(n_cams):
        feat_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=w2c[i : i + 1],
            projection_matrices=K[i : i + 1],
            image_width=W,
            image_height=H,
            near=0.0,
            far=1e10,
        )
        alpha_i = alpha[0, ..., 0].clamp_min(1e-10)
        depth_i = (feat_depth[0, ..., -1] / alpha_i).to(torch.float16)
        valid = ((depth_i > NEAR) & (depth_i < FAR) & (alpha_i > alpha_thresh)).to(torch.float16)
        weight_img = (alpha_i.to(torch.float16) * valid).contiguous()

        accum_grid, tsdf, weights = accum_grid.integrate_tsdf(
            truncation_distance=TRUNCATION_MARGIN,
            projection_matrix=K[i].to(torch.float16),
            cam_to_world_matrix=c2w[i].to(torch.float16),
            tsdf=tsdf,
            weights=weights,
            depth_image=depth_i,
            weight_image=weight_img,
        )

        if progress_cb:
            progress_cb(i + 1, n_cams)

    # Extract mesh
    grid_b = accum_grid.pruned_grid(weights > 0)
    phi = grid_b.inject_from(accum_grid, tsdf).to(torch.float32)
    vj, fj, nj = grid_b.marching_cubes(field=phi.reshape(-1, 1), level=0.0)

    verts = (vj.jdata if hasattr(vj, "jdata") else vj).cpu().numpy().astype(np.float32)
    faces = (fj.jdata if hasattr(fj, "jdata") else fj).cpu().numpy().astype(np.int32)
    normals = (nj.jdata if hasattr(nj, "jdata") else nj).cpu().numpy().astype(np.float32)

    # Save PLY if requested
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_ply(out_path.with_stem("mesh_tsdf"), verts, faces, normals=normals)

    return {"verts": verts, "faces": faces, "normals": normals, "colors": None}


@torch.no_grad()
def _mesh_dlnr(model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
               density, shell_thickness, bake_texture, tex_size, out_path, progress_cb,
               splat_fingerprint):
    """DLNR stereo depth meshing with per-vertex colors.

    DLNR depth generation is the dominant cost (~10s+); density and
    shell_thickness only affect the TSDF integration that follows. We cache
    the per-camera (rgb, depth, weight) tensors keyed on splat state + camera
    config so re-meshing with different density/shell skips the DLNR pass.
    """
    import sys
    import pathlib
    import tempfile
    import threading
    from fvdb import Grid
    from fvdb_reality_capture.tools._tsdf_from_splats_dlnr import TSDFInputDataset
    from fvdb_reality_capture.foundation_models.dlnr import DLNRModel
    from fvdb_reality_capture.sfm_scene import sfm_cache

    # DLNR stereo params (kept stable so density/shell can change without invalidating)
    BASELINE_FRAC = 0.15
    NEAR_MULT = 1.0
    FAR_MULT = 100.0
    REPRO_THRESH = 10.0
    ALPHA_THRESH = 0.1

    TRUNC = density * scene_diag
    vox = TRUNC / shell_thickness
    print(f"  _mesh_dlnr: scene_diag={scene_diag:.4f}  density={density}  "
          f"TRUNC={TRUNC:.4f}  voxel={vox:.4f}  shell={shell_thickness}",
          file=sys.stderr, flush=True)
    if TRUNC > 0.05:
        print(f"  _mesh_dlnr: WARNING — TRUNC={TRUNC*100:.1f}cm is large; "
              f"colors will be averaged across a wide spatial band. "
              f"Lower 'density' slider for finer color resolution.",
              file=sys.stderr, flush=True)

    image_sizes = torch.tensor([[H, W]] * n_cams, dtype=torch.int32, device=device)

    # ---- 1. DLNR depth pass (cached on splat_fingerprint + camera/stereo config) ----
    cache_key = (
        splat_fingerprint, n_cams, H, W,
        BASELINE_FRAC, NEAR_MULT, FAR_MULT, REPRO_THRESH, ALPHA_THRESH,
    )
    cached = _DLNR_DEPTH_CACHE.get(cache_key) if splat_fingerprint is not None else None

    if cached is None:
        # Patch FileLock to skip signal handling (broken in non-main threads)
        original_enter = sfm_cache.FileLock.__enter__
        original_exit = sfm_cache.FileLock.__exit__

        def patched_enter(self):
            if not hasattr(self, '_lock'):
                self._lock = threading.Lock()
            self._lock.acquire()
            return self

        def patched_exit(self, *args):
            if hasattr(self, '_lock'):
                self._lock.release()

        sfm_cache.FileLock.__enter__ = patched_enter
        sfm_cache.FileLock.__exit__ = patched_exit

        try:
            print(f"  _mesh_dlnr: DLNR cache miss — generating depths for {n_cams} cams",
                  file=sys.stderr, flush=True)
            with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
                dataset = TSDFInputDataset(
                    cache_path=pathlib.Path(tmpdir),
                    model=model,
                    camera_to_world_matrices=c2w,
                    projection_matrices=K,
                    image_sizes=image_sizes,
                    baseline=BASELINE_FRAC,
                    near=NEAR_MULT,
                    far=FAR_MULT,
                    reprojection_threshold=REPRO_THRESH,
                    alpha_threshold=ALPHA_THRESH,
                    dlnr_model=DLNRModel(backbone="middleburry", device=model.device),
                    use_absolute_baseline=False,
                    show_progress=True,
                )
                # Drain dataset entries onto CPU so we can release the tempdir.
                cached = []
                for i in range(len(dataset)):
                    rgb_i, depth_i, weight_i = dataset[i]
                    cached.append((rgb_i.cpu(), depth_i.cpu(), weight_i.cpu()))
        finally:
            sfm_cache.FileLock.__enter__ = original_enter
            sfm_cache.FileLock.__exit__ = original_exit

        if splat_fingerprint is not None:
            _DLNR_DEPTH_CACHE.clear()  # only keep one entry — splat state changes invalidate old
            _DLNR_DEPTH_CACHE[cache_key] = cached
    else:
        print(f"  _mesh_dlnr: DLNR cache HIT — reusing {n_cams} cached depths",
              file=sys.stderr, flush=True)

    # ---- 2. Manual TSDF integration using cached (rgb, depth, weight) tuples ----
    dtype = torch.float16
    feature_dtype = torch.uint8
    accum_grid = Grid.from_dense(dense_dims=1, ijk_min=0, voxel_size=vox, origin=0.0, device=device)
    tsdf = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
    weights = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
    colors_grid = torch.zeros((accum_grid.num_voxels, model.num_channels), device=device, dtype=feature_dtype)

    for i, (rgb_cpu, depth_cpu, weight_cpu) in enumerate(cached):
        cam_to_world_matrix = c2w[i].to(dtype=torch.float32, device=device)
        projection_matrix = K[i].to(dtype=torch.float32, device=device)

        rgb_image = (rgb_cpu * 255).to(feature_dtype).to(device)
        depth_image = depth_cpu.to(dtype).to(device)
        weight_image = weight_cpu.to(dtype).to(device)

        accum_grid, tsdf, weights, colors_grid = accum_grid.integrate_tsdf_with_features(
            TRUNC,
            projection_matrix.to(dtype),
            cam_to_world_matrix.to(dtype),
            tsdf,
            colors_grid,
            weights,
            depth_image,
            rgb_image,
            weight_image,
        )

        # Prune zero-weight voxels each step to bound memory
        new_grid = accum_grid.pruned_grid(weights > 0.0)
        tsdf = new_grid.inject_from(accum_grid, tsdf)
        colors_grid = new_grid.inject_from(accum_grid, colors_grid)
        weights = new_grid.inject_from(accum_grid, weights)
        accum_grid = new_grid

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if progress_cb:
            progress_cb(i + 1, n_cams)

    # Final prune
    new_grid = accum_grid.pruned_grid(weights > 0.0)
    filter_tsdf = new_grid.inject_from(accum_grid, tsdf)
    filter_colors = new_grid.inject_from(accum_grid, colors_grid)

    # ---- 3. Marching cubes + vertex color sampling ----
    mesh_vertices, mesh_faces, _ = new_grid.marching_cubes(filter_tsdf, 0.0)
    mesh_colors = new_grid.sample_trilinear(mesh_vertices, filter_colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    verts = mesh_vertices.cpu().numpy().astype(np.float32)
    faces = mesh_faces.cpu().numpy().astype(np.int32)
    faces = faces[:, ::-1]  # Flip winding order to fix inside-out mesh
    colors = mesh_colors.cpu().numpy().astype(np.float32)

    # Texture bake → (uv, tex_image) or (None, None) if disabled/fails
    uv, tex_image = (None, None)
    if bake_texture:
        try:
            uv, tex_image = _bake_texture_fast(verts, faces, colors, tex_size)
        except Exception as e:
            print(f"Texture baking skipped: {e}")
            import traceback; traceback.print_exc()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_ply(out_path.with_stem("mesh_dlnr"), verts, faces, colors=colors)
        if uv is not None and tex_image is not None:
            _save_textured_obj(out_path.with_stem("mesh_dlnr_tex"), verts, faces, uv, tex_image)

    return {
        "verts": verts, "faces": faces,
        "normals": None, "colors": colors,
        "uv": uv, "tex_image": tex_image,
    }


def clear_dlnr_cache() -> None:
    """Drop any cached DLNR depth maps. Call after manual splat edits."""
    _DLNR_DEPTH_CACHE.clear()


# -----------------------------------------------------------------------------
# Shared helpers — used by generate_mesh AND bake_texture_from_splat_projection
# -----------------------------------------------------------------------------

def _build_splat_model(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
):
    """Invert activations + split SH, return fvdb.GaussianSplat3d."""
    import fvdb
    eps = 1e-6
    opac_clamped = opacities.clamp(eps, 1 - eps)
    logit_opacities = torch.log(opac_clamped / (1 - opac_clamped))
    log_scales = torch.log(scales.clamp_min(eps))
    sh0 = sh[:, :1, :].contiguous()
    shN = sh[:, 1:, :].contiguous()
    return fvdb.GaussianSplat3d.from_tensors(
        means=means, quats=quats,
        log_scales=log_scales, logit_opacities=logit_opacities,
        sh0=sh0, shN=shN,
    )


def _build_dome_cameras(
    scene_center: torch.Tensor,
    scene_diag: float,
    n_cams: int,
    image_size: int,
    device: str,
    fov_deg: float = 60.0,
    elev_min_deg: float = 5.0,
    elev_max_deg: float = 70.0,
    ring_radius_frac: float = 0.8,
):
    """Fibonacci hemisphere of cameras aimed at the scene center.

    Returns (c2w, w2c, K) all torch.float32 on `device`.
    `c2w`: (C, 4, 4); `w2c`: (C, 4, 4); `K`: (C, 3, 3).
    """
    i = torch.arange(n_cams, device=device, dtype=torch.float32)
    golden_ang = np.pi * (3 - np.sqrt(5))
    s_min = float(np.sin(np.deg2rad(elev_min_deg)))
    s_max = float(np.sin(np.deg2rad(elev_max_deg)))
    sin_elev = s_min + (s_max - s_min) * (i + 0.5) / n_cams
    cos_elev = torch.sqrt(torch.clamp(1 - sin_elev**2, min=0))
    az = i * golden_ang
    ring_radius = scene_diag * ring_radius_frac
    offset = torch.stack(
        [cos_elev * torch.cos(az), -sin_elev, cos_elev * torch.sin(az)],
        dim=-1,
    )
    cam_centers = scene_center + ring_radius * offset

    def look_at(eyes, target, world_up=torch.tensor([0., -1., 0.], device=device)):
        fwd = target - eyes
        fwd = fwd / fwd.norm(dim=-1, keepdim=True)
        up_ref = world_up.expand_as(fwd).clone()
        parallel = (fwd * up_ref).sum(-1, keepdim=True).abs() > 0.98
        fallback = torch.tensor([1., 0., 0.], device=fwd.device).expand_as(fwd)
        up_ref = torch.where(parallel, fallback, up_ref)
        right = torch.cross(fwd, up_ref, dim=-1)
        right = right / right.norm(dim=-1, keepdim=True)
        up = torch.cross(right, fwd, dim=-1)
        R_c2w = torch.stack([right, -up, fwd], dim=-1)
        c2w = torch.eye(4, device=device).expand(len(eyes), 4, 4).clone()
        c2w[:, :3, :3] = R_c2w
        c2w[:, :3, 3] = eyes
        return c2w

    c2w = look_at(cam_centers, scene_center.expand_as(cam_centers))
    w2c = torch.linalg.inv(c2w).contiguous()
    fx = fy = 0.5 * image_size / np.tan(np.deg2rad(fov_deg) * 0.5)
    K = (
        torch.tensor(
            [[fx, 0, image_size / 2], [0, fy, image_size / 2], [0, 0, 1]],
            device=device, dtype=torch.float32,
        )
        .expand(n_cams, 3, 3)
        .contiguous()
    )
    return c2w, w2c, K


def _save_ply(path: Path, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None, colors: np.ndarray | None = None):
    """Save mesh to PLY with optional normals and colors."""
    from plyfile import PlyData, PlyElement

    if colors is not None:
        colors_u8 = (colors * 255).astype(np.uint8)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        vertex = np.zeros(len(verts), dtype=dtype)
        vertex["x"], vertex["y"], vertex["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        vertex["red"], vertex["green"], vertex["blue"] = colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2]
    elif normals is not None:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        vertex = np.zeros(len(verts), dtype=dtype)
        vertex["x"], vertex["y"], vertex["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        vertex["nx"], vertex["ny"], vertex["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        vertex = np.zeros(len(verts), dtype=dtype)
        vertex["x"], vertex["y"], vertex["z"] = verts[:, 0], verts[:, 1], verts[:, 2]

    face = np.zeros(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = faces

    PlyData(
        [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")],
        text=False,
    ).write(str(path.with_suffix(".ply")))



def bake_texture_from_splats(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    tex_size: int = 1024,
    target_faces: int = 100_000,
    out_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Bake a textured mesh from DLNR per-vertex colors (matches notebook approach).

    Pipeline (no camera reprojection, no occlusion logic):
    1. Decimate mesh to ~target_faces via quadric decimation. open3d interpolates
       vertex_colors during decimation so colors stay aligned with the new sparser
       vertex set.
    2. UV-unwrap with xatlas.
    3. Barycentrically interpolate vertex colors into a `tex_size`x`tex_size` PNG.
    4. Gap-fill UV holes via scipy distance transform.
    5. Save OBJ + MTL + PNG (V-flipped for OBJ Y-up convention).

    Returns dict with: uv, tex_image (unflipped, for viser), verts, faces, per_vertex_uv=True.
    """
    import sys, time
    import xatlas
    from scipy.ndimage import distance_transform_edt
    from PIL import Image
    from tqdm import tqdm

    def _log(msg):
        print(f"  bake: {msg}", file=sys.stderr, flush=True)
        if progress_cb:
            progress_cb(_log.step, _log.total)
    _log.step = 0
    _log.total = 100

    def _stage(step, msg):
        _log.step = step
        _log(msg)

    t_total = time.perf_counter()
    _stage(1, f"[1/6] input mesh: {len(verts):,} verts, {len(faces):,} faces, target_faces={target_faces:,}, tex_size={tex_size}")

    # 1) Decimate (open3d interpolates vertex_colors during simplification)
    _stage(5, f"[2/6] decimating to ~{target_faces:,} faces (open3d quadric)…")
    t0 = time.perf_counter()
    m = o3d.geometry.TriangleMesh()
    m.vertices      = o3d.utility.Vector3dVector(verts.astype(np.float64))
    m.triangles     = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    n_in = len(faces)
    if n_in > target_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        dec_ratio = len(m.triangles) / n_in
        _log(f"      decimation ratio {dec_ratio:.3f} (kept {dec_ratio*100:.1f}% of faces)")
    else:
        _log(f"      input already under target ({n_in:,} ≤ {target_faces:,}) — skipping decimation")
    V = np.asarray(m.vertices, dtype=np.float32)
    Faces = np.asarray(m.triangles, dtype=np.uint32)
    C = np.asarray(m.vertex_colors, dtype=np.float32)
    _stage(10, f"      done in {time.perf_counter()-t0:.1f}s → {len(Faces):,} faces, {len(V):,} verts")

    # 2) UV unwrap the decimated mesh
    _stage(12, f"[3/6] xatlas UV unwrap of {len(Faces):,} faces (single-threaded CPU, can take 10–60s)…")
    t0 = time.perf_counter()
    vmap, F_uv, uvs = xatlas.parametrize(V, Faces)
    V_uv = V[vmap]
    C_uv = C[vmap]
    uvs = uvs.astype(np.float32)
    # Defensive: normalize if xatlas returned out-of-[0,1] UVs
    uv_min, uv_max = uvs.min(axis=0), uvs.max(axis=0)
    if (uv_min < 0).any() or (uv_max > 1).any():
        _log(f"      WARNING: UVs out of [0,1] (x=[{uv_min[0]:.2f},{uv_max[0]:.2f}] "
             f"y=[{uv_min[1]:.2f},{uv_max[1]:.2f}]) — normalizing")
        uvs = (uvs - uv_min) / np.maximum(uv_max - uv_min, 1e-8)
    _stage(30, f"      done in {time.perf_counter()-t0:.1f}s → "
               f"{len(V):,} → {len(V_uv):,} verts (xatlas added {len(V_uv)-len(V):,} seam-split verts)")

    # 3) Barycentric bake of per-vertex colors
    _stage(32, f"[4/6] barycentric bake into {tex_size}x{tex_size} texture ({len(F_uv):,} triangles)…")
    t0 = time.perf_counter()
    tex  = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    mask = np.zeros((tex_size, tex_size), dtype=bool)
    uv_px = uvs * tex_size

    # Tight per-face inner loop with tqdm — manual stride to surface progress to the GUI too.
    n_faces = len(F_uv)
    update_every = max(1, n_faces // 200)
    pbar = tqdm(total=n_faces, desc="  bake: rasterize", file=sys.stderr,
                bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for fi, tri in enumerate(F_uv):
        p   = uv_px[tri]                       # (3, 2) UV in pixel space
        col = C_uv[tri]                         # (3, 3) RGB per corner
        u_min = max(0, int(np.floor(p[:, 0].min())))
        u_max = min(tex_size - 1, int(np.ceil(p[:, 0].max())))
        v_min = max(0, int(np.floor(p[:, 1].min())))
        v_max = min(tex_size - 1, int(np.ceil(p[:, 1].max())))
        if u_min >= u_max or v_min >= v_max:
            pbar.update(1); continue
        uu, vv = np.meshgrid(np.arange(u_min, u_max + 1),
                             np.arange(v_min, v_max + 1))
        pu, pv = uu + 0.5, vv + 0.5
        denom = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(denom) < 1e-10:
            pbar.update(1); continue
        a = ((p[1, 1] - p[2, 1]) * (pu - p[2, 0]) + (p[2, 0] - p[1, 0]) * (pv - p[2, 1])) / denom
        b = ((p[2, 1] - p[0, 1]) * (pu - p[2, 0]) + (p[0, 0] - p[2, 0]) * (pv - p[2, 1])) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            pbar.update(1); continue
        col_interp = a[..., None] * col[0] + b[..., None] * col[1] + c[..., None] * col[2]
        tex[vv[inside],  uu[inside]]  = col_interp[inside]
        mask[vv[inside], uu[inside]]  = True
        pbar.update(1)
        # Mirror progress to GUI status periodically
        if progress_cb and (fi % update_every == 0):
            _log.step = 32 + int(50 * fi / n_faces)
            progress_cb(_log.step, _log.total)
    pbar.close()
    _stage(82, f"      done in {time.perf_counter()-t0:.1f}s "
               f"(covered {mask.mean()*100:.1f}% of UV space — {mask.sum():,}/{mask.size:,} texels)")

    # 4) Gap fill outside UV islands so edge sampling doesn't bleed black
    _stage(85, f"[5/6] gap-fill {(~mask).sum():,} empty texels (scipy distance transform)…")
    t0 = time.perf_counter()
    _, inds = distance_transform_edt(~mask, return_indices=True)
    tex = tex[inds[0], inds[1]]
    _stage(88, f"      done in {time.perf_counter()-t0:.1f}s")

    # 5) Build images: V-flipped for OBJ disk save, NOT flipped for viser (glTF)
    _stage(90, "      converting to uint8 + building PIL images…")
    tex_u8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    img_for_obj   = Image.fromarray(tex_u8[::-1])        # PNG/OBJ uses Y-up V
    img_for_viser = Image.fromarray(tex_u8)              # glTF uses Y-down V

    # 6) Save OBJ + MTL + PNG if out_path given
    if out_path is not None:
        _stage(92, f"[6/6] saving PNG + OBJ + MTL to {Path(out_path).parent}…")
        t0 = time.perf_counter()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base = out_path.with_stem("mesh_splat_tex")
        png_path = base.with_suffix(".png")
        obj_path = base.with_suffix(".obj")
        mtl_path = base.with_suffix(".mtl")
        img_for_obj.save(str(png_path))
        _log(f"      saved PNG ({png_path.stat().st_size // 1024:,} KB)")
        with open(obj_path, "w") as fh:
            fh.write(f"mtllib {mtl_path.name}\n")
            for vx in V_uv:
                fh.write(f"v {vx[0]:.6f} {vx[1]:.6f} {vx[2]:.6f}\n")
            for uv in uvs:
                fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            fh.write("usemtl mat0\n")
            for tri in F_uv:
                a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                fh.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        with open(mtl_path, "w") as fh:
            fh.write(f"newmtl mat0\nKa 1 1 1\nKd 1 1 1\nmap_Kd {png_path.name}\n")
        _log(f"      saved OBJ ({obj_path.stat().st_size // 1024:,} KB) + MTL in {time.perf_counter()-t0:.1f}s")
        _log(f"      TEXTURE → {png_path.resolve()}")
        _log(f"      MESH    → {obj_path.resolve()}")
    else:
        _stage(92, "[6/6] no out_path given — skipping disk save")

    _stage(100, f"DONE: total bake time {time.perf_counter()-t_total:.1f}s")

    # Return the decimated mesh WITH its decimated vertex colors. Viser displays
    # these directly via add_mesh_trimesh + vertex_colors (no GLB texture-atlas
    # round-trip → no UV scramble bugs). The on-disk OBJ uses the texture atlas
    # we just saved (MeshLab / Blender path).
    return {
        "verts": V,                          # decimated vertices (no seam-split)
        "faces": Faces.astype(np.int32),     # decimated faces (no seam-split)
        "colors": np.clip(C, 0, 1),          # decimated per-vertex colors
        # Texture artifacts still available for users who want to push a textured GLB:
        "uv": uvs,
        "tex_image": img_for_viser,
        "uv_verts": V_uv,                    # xatlas-unwrapped (for textured GLB)
        "uv_faces": F_uv.astype(np.int32),
        "per_vertex_uv": True,
    }


def bake_texture_from_splat_projection(
    verts: np.ndarray,
    faces: np.ndarray,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    tex_size: int = 2048,
    target_faces: int = 100_000,
    n_cams: int = 96,
    image_size: int = 1024,
    depth_tol_frac: float = 0.02,
    device: str = "cuda",
    out_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Photogrammetric bake: per-texel multi-camera weighted average of splat RGB.

    Same shell as bake_texture_from_splats (decimate → xatlas → gap-fill → save)
    but replaces the barycentric vertex-color step with per-texel reprojection
    into all dome cameras, weighted by face-normal alignment AND visibility
    (rendered depth ≈ projected depth).

    No silent fallback — texels not seen by any camera get gap-filled.
    """
    import sys, time
    import xatlas
    from scipy.ndimage import distance_transform_edt
    from PIL import Image
    from tqdm import tqdm

    def _log(msg):
        print(f"  splat-bake: {msg}", file=sys.stderr, flush=True)

    def _stage(step, total, msg):
        _log(msg)
        if progress_cb:
            progress_cb(step, total)

    TOTAL = 100
    t_total = time.perf_counter()
    _stage(1, TOTAL,
           f"[1/7] input: {len(verts):,} verts, {len(faces):,} faces, "
           f"target={target_faces:,}, tex={tex_size}, n_cams={n_cams}, image_size={image_size}")

    # ----- 1) Decimate (preserves vertex_colors via interp, but we don't use colors here) -----
    _stage(3, TOTAL, f"[2/7] decimate to ~{target_faces:,} faces…")
    t0 = time.perf_counter()
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    n_in = len(faces)
    if n_in > target_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    V = np.asarray(m.vertices, dtype=np.float32)
    Faces = np.asarray(m.triangles, dtype=np.uint32)
    _stage(8, TOTAL, f"      done in {time.perf_counter()-t0:.1f}s → "
                     f"{len(Faces):,} faces, {len(V):,} verts")

    # ----- 2) xatlas UV unwrap -----
    _stage(10, TOTAL, f"[3/7] xatlas UV unwrap ({len(Faces):,} faces, CPU)…")
    t0 = time.perf_counter()
    vmap, F_uv, uvs = xatlas.parametrize(V, Faces)
    V_uv = V[vmap]
    uvs = uvs.astype(np.float32)
    uv_min, uv_max = uvs.min(axis=0), uvs.max(axis=0)
    if (uv_min < 0).any() or (uv_max > 1).any():
        uvs = (uvs - uv_min) / np.maximum(uv_max - uv_min, 1e-8)
    _stage(28, TOTAL, f"      done in {time.perf_counter()-t0:.1f}s → "
                      f"{len(V):,} → {len(V_uv):,} verts after seam splits")

    # ----- 3) Build splat model + dome cameras -----
    _stage(30, TOTAL, f"[4/7] build splat model + dome cameras…")
    bbmin = means.min(0).values
    bbmax = means.max(0).values
    scene_diag = float((bbmax - bbmin).norm())
    scene_center = 0.5 * (bbmin + bbmax)
    DEPTH_TOL = depth_tol_frac * scene_diag
    _log(f"      scene_diag={scene_diag:.3f}  depth_tol={DEPTH_TOL:.4f}")
    model = _build_splat_model(means, quats, scales, opacities, sh)
    c2w, w2c, K = _build_dome_cameras(scene_center, scene_diag, n_cams, image_size, device)
    cam_centers = c2w[:, :3, 3]                                       # (C, 3)
    cam_fwd_world = c2w[:, :3, 2]                                     # (C, 3) — camera +Z in world

    # ----- 4) Render dome RGB + depth -----
    _stage(35, TOTAL, f"[5/7] render dome ({n_cams} cams × {image_size}²)…")
    t0 = time.perf_counter()
    cam_rgb = torch.zeros(n_cams, image_size, image_size, 3, dtype=torch.float32, device=device)
    cam_depth = torch.zeros(n_cams, image_size, image_size, dtype=torch.float32, device=device)
    for i in tqdm(range(n_cams), desc="  splat-bake: render", file=sys.stderr):
        feat_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=w2c[i:i+1], projection_matrices=K[i:i+1],
            image_width=image_size, image_height=image_size, near=0.0, far=1e10,
        )
        a = alpha[0, ..., 0].clamp_min(1e-10)
        cam_rgb[i] = (feat_depth[0, ..., :3] / a.unsqueeze(-1)).float()
        cam_depth[i] = (feat_depth[0, ..., -1] / a).float()
    torch.cuda.synchronize()
    _stage(50, TOTAL, f"      done in {time.perf_counter()-t0:.1f}s")

    # ----- 5) Per-face chunked GPU rasterization with multi-camera weighted avg -----
    _stage(52, TOTAL, f"[6/7] per-texel multi-cam projection ({len(F_uv):,} faces)…")
    t0 = time.perf_counter()
    F = len(F_uv)
    V_uv_t = torch.from_numpy(V_uv.astype(np.float32)).to(device)
    F_uv_t = torch.from_numpy(F_uv.astype(np.int64)).to(device)
    tri_uv_t = torch.from_numpy((uvs[F_uv] * (tex_size - 1)).astype(np.float32)).to(device)  # (F,3,2) pixel coords
    tri_verts = V_uv_t[F_uv_t]                                           # (F,3,3) 3D positions per corner

    # Per-face normals (in world frame), normalized
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    tri_normal = torch.cross(e1, e2, dim=-1)
    tri_normal = tri_normal / tri_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    uv_min_t = tri_uv_t.min(dim=1).values
    uv_max_t = tri_uv_t.max(dim=1).values

    tex_t  = torch.zeros((tex_size, tex_size, 3), dtype=torch.float32, device=device)
    mask_t = torch.zeros((tex_size, tex_size), dtype=torch.bool, device=device)

    CHUNK = 1024  # smaller than vertex bake because each texel does C-camera work
    n_chunks = (F + CHUNK - 1) // CHUNK
    pbar = tqdm(range(n_chunks), desc="  splat-bake: chunks", file=sys.stderr)
    for c_idx in pbar:
        s, e = c_idx * CHUNK, min(F, (c_idx + 1) * CHUNK)
        ch_size = e - s
        ch_uv     = tri_uv_t[s:e]                       # (B, 3, 2)
        ch_verts  = tri_verts[s:e]                      # (B, 3, 3)
        ch_normal = tri_normal[s:e]                     # (B, 3)
        ch_min    = uv_min_t[s:e]
        ch_max    = uv_max_t[s:e]

        xmin = ch_min[:, 0].clamp(0, tex_size - 1).floor().long()
        xmax = ch_max[:, 0].clamp(0, tex_size - 1).ceil().long()
        ymin = ch_min[:, 1].clamp(0, tex_size - 1).floor().long()
        ymax = ch_max[:, 1].clamp(0, tex_size - 1).ceil().long()
        bw = (xmax - xmin + 1).clamp_min(0)
        bh = (ymax - ymin + 1).clamp_min(0)
        W_pad = int(bw.max().item()) if ch_size > 0 else 0
        H_pad = int(bh.max().item()) if ch_size > 0 else 0
        if W_pad == 0 or H_pad == 0:
            continue
        pbar.set_postfix(pad=f"{H_pad}x{W_pad}", faces=f"{s}-{e}")

        dx = torch.arange(W_pad, device=device).view(1, 1, W_pad)
        dy = torch.arange(H_pad, device=device).view(1, H_pad, 1)
        px_x = (xmin.view(ch_size, 1, 1) + dx).expand(ch_size, H_pad, W_pad).contiguous()
        px_y = (ymin.view(ch_size, 1, 1) + dy).expand(ch_size, H_pad, W_pad).contiguous()
        valid_pad = (dx < bw.view(ch_size, 1, 1)) & (dy < bh.view(ch_size, 1, 1))

        # UV-space barycentric → (u_b, v_b, w_b)
        uv0 = ch_uv[:, 0].view(ch_size, 1, 1, 2)
        uv1 = ch_uv[:, 1].view(ch_size, 1, 1, 2)
        uv2 = ch_uv[:, 2].view(ch_size, 1, 1, 2)
        p = torch.stack([px_x.float(), px_y.float()], dim=-1)
        v0v1 = uv1 - uv0
        v0v2 = uv2 - uv0
        v0p  = p - uv0
        d00 = (v0v1 * v0v1).sum(-1)
        d01 = (v0v1 * v0v2).sum(-1)
        d11 = (v0v2 * v0v2).sum(-1)
        d20 = (v0p * v0v1).sum(-1)
        d21 = (v0p * v0v2).sum(-1)
        denom = d00 * d11 - d01 * d01
        denom_safe = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        v_b = (d11 * d20 - d01 * d21) / denom_safe
        w_b = (d00 * d21 - d01 * d20) / denom_safe
        u_b = 1 - v_b - w_b
        inside = (u_b >= -1e-5) & (v_b >= -1e-5) & (w_b >= -1e-5) & valid_pad & (denom.abs() >= 1e-12)
        if not inside.any():
            continue

        # 3D position via barycentric of face vertices
        vv0 = ch_verts[:, 0].view(ch_size, 1, 1, 3)
        vv1 = ch_verts[:, 1].view(ch_size, 1, 1, 3)
        vv2 = ch_verts[:, 2].view(ch_size, 1, 1, 3)
        pos3d = u_b.unsqueeze(-1) * vv0 + v_b.unsqueeze(-1) * vv1 + w_b.unsqueeze(-1) * vv2
        # → (B, H_pad, W_pad, 3)

        # Batch cameras to keep per-chunk memory bounded while staying vectorized.
        # Pick cam_batch so each (B, H, W, cam_batch, 3) tensor fits within ~1 GB.
        # (Peak is several such tensors; the 1 GB budget keeps total per-tensor footprint sane.)
        K0 = K[0]                                                                # (3, 3)
        bytes_per_elem = 4 * 3
        cam_batch = max(1, min(n_cams,
            int(1e9 // max(1, ch_size * H_pad * W_pad * bytes_per_elem))))

        inside_f = inside.float()
        wsum = torch.zeros_like(inside_f)                                        # (B, H, W)
        rgbsum = torch.zeros((*inside_f.shape, 3), device=device, dtype=torch.float32)
        normal_b = ch_normal.view(ch_size, 1, 1, 1, 3)                           # broadcast over C
        pos3d_5 = pos3d.unsqueeze(-2)                                            # (B, H, W, 1, 3)
        pos_h = torch.cat([pos3d, torch.ones_like(pos3d[..., :1])], dim=-1)      # (B, H, W, 4)

        for cb_s in range(0, n_cams, cam_batch):
            cb_e = min(n_cams, cb_s + cam_batch)
            Cb = cb_e - cb_s
            w2c_b = w2c[cb_s:cb_e]                                               # (Cb, 4, 4)
            cc_b  = cam_centers[cb_s:cb_e]                                       # (Cb, 3)

            pos_cam = torch.einsum("cij,bhwj->bhwci", w2c_b, pos_h)[..., :3]     # (B, H, W, Cb, 3)
            z = pos_cam[..., 2]                                                  # (B, H, W, Cb)
            ok_z = z > 1e-6
            z_safe = z.clamp_min(1e-6).unsqueeze(-1)
            proj = torch.einsum("ij,bhwcj->bhwci", K0, pos_cam / z_safe)         # (B, H, W, Cb, 3)
            ix = proj[..., 0].round().long()
            iy = proj[..., 1].round().long()
            ok_xy = ok_z & (ix >= 0) & (ix < image_size) & (iy >= 0) & (iy < image_size)
            ix_c = ix.clamp(0, image_size - 1)
            iy_c = iy.clamp(0, image_size - 1)

            cam_idx = torch.arange(cb_s, cb_e, device=device).view(1, 1, 1, Cb).expand_as(ix_c)
            rendered_z = cam_depth[cam_idx, iy_c, ix_c]                          # (B, H, W, Cb)
            visible = (rendered_z - z).abs() < DEPTH_TOL

            view_vec = cc_b.view(1, 1, 1, Cb, 3) - pos3d_5                       # (B, H, W, Cb, 3)
            view_vec_n = view_vec / view_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            n_dot_v = (view_vec_n * normal_b).sum(-1).clamp_min(0)               # (B, H, W, Cb)
            w_b = n_dot_v * visible.float() * ok_xy.float() * inside_f.unsqueeze(-1)
            rgb_b = cam_rgb[cam_idx, iy_c, ix_c]                                 # (B, H, W, Cb, 3)
            wsum  = wsum + w_b.sum(dim=-1)
            rgbsum = rgbsum + (rgb_b * w_b.unsqueeze(-1)).sum(dim=-2)

        valid_texel = wsum > 1e-6
        rgb_avg = torch.zeros_like(rgbsum)
        rgb_avg[valid_texel] = rgbsum[valid_texel] / wsum[valid_texel].unsqueeze(-1)

        # Scatter into texture (first-write-wins across chunks)
        write_mask = inside & valid_texel
        if write_mask.any():
            flat_py = px_y[write_mask].clamp(0, tex_size - 1)
            flat_px = px_x[write_mask].clamp(0, tex_size - 1)
            already = mask_t[flat_py, flat_px]
            to_write = ~already
            if to_write.any():
                rgb_to_write = rgb_avg[write_mask]
                tex_t[flat_py[to_write], flat_px[to_write]] = rgb_to_write[to_write]
                mask_t[flat_py[to_write], flat_px[to_write]] = True

        torch.cuda.synchronize()
        if progress_cb:
            frac = 52 + int(28 * (c_idx + 1) / n_chunks)
            progress_cb(frac, TOTAL)

    tex = tex_t.cpu().numpy()
    mask = mask_t.cpu().numpy()
    cov = mask.mean()
    _stage(82, TOTAL, f"      done in {time.perf_counter()-t0:.1f}s "
                      f"({cov*100:.1f}% of texels covered by visible cameras)")

    # ----- 6) Gap fill + sRGB encode -----
    _stage(85, TOTAL, f"      gap-fill {(~mask).sum():,} unfilled texels…")
    t0 = time.perf_counter()
    if not mask.all():
        _, inds = distance_transform_edt(~mask, return_indices=True)
        tex = tex[inds[0], inds[1]]
    _stage(88, TOTAL, f"      gap fill done in {time.perf_counter()-t0:.1f}s")

    tex_srgb = _linear_to_srgb(tex)
    tex_u8 = (np.clip(tex_srgb, 0, 1) * 255).astype(np.uint8)
    img_for_obj   = Image.fromarray(tex_u8[::-1])
    img_for_viser = Image.fromarray(tex_u8)

    # ----- 7) Save OBJ + MTL + PNG -----
    if out_path is not None:
        _stage(92, TOTAL, f"[7/7] saving OBJ + MTL + PNG…")
        t0 = time.perf_counter()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base = out_path.with_stem("mesh_splat_tex")
        png_path = base.with_suffix(".png")
        obj_path = base.with_suffix(".obj")
        mtl_path = base.with_suffix(".mtl")
        img_for_obj.save(str(png_path))
        with open(obj_path, "w") as fh:
            fh.write(f"mtllib {mtl_path.name}\n")
            for vx in V_uv:
                fh.write(f"v {vx[0]:.6f} {vx[1]:.6f} {vx[2]:.6f}\n")
            for uv in uvs:
                fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            fh.write("usemtl mat0\n")
            for tri in F_uv:
                a, b, cc = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                fh.write(f"f {a}/{a} {b}/{b} {cc}/{cc}\n")
        with open(mtl_path, "w") as fh:
            fh.write(f"newmtl mat0\nKa 1 1 1\nKd 1 1 1\nmap_Kd {png_path.name}\n")
        _log(f"      saved in {time.perf_counter()-t0:.1f}s")
        _log(f"      TEXTURE → {png_path.resolve()}")
        _log(f"      MESH    → {obj_path.resolve()}")

    _stage(100, TOTAL, f"DONE: total bake time {time.perf_counter()-t_total:.1f}s")

    return {
        "verts": V,
        "faces": Faces.astype(np.int32),
        "uv": uvs,
        "tex_image": img_for_viser,
        "uv_verts": V_uv,
        "uv_faces": F_uv.astype(np.int32),
        "per_vertex_uv": True,
    }


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Apply sRGB OETF (linear-to-sRGB encoding)."""
    c = np.clip(c, 0.0, 1.0)
    lo = 12.92 * c
    hi = 1.055 * np.power(np.maximum(c, 1e-12), 1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, lo, hi)


def _save_textured_obj(base_path: Path, verts: np.ndarray, faces: np.ndarray,
                       per_corner_uvs: np.ndarray, image):
    """Save OBJ + MTL + PNG for a textured mesh. per_corner_uvs shape (F, 3, 2)."""
    image.save(str(base_path.with_suffix(".png")))

    # Flatten per-corner UVs to a list, build per-face uv indices (1-based)
    F = faces.shape[0]
    uv_flat = per_corner_uvs.reshape(-1, 2)  # (F*3, 2)
    uv_idx_flat = np.arange(F * 3).reshape(F, 3) + 1  # 1-based

    obj_lines = [f"mtllib {base_path.name}.mtl"]
    for vx in verts:
        obj_lines.append(f"v {vx[0]} {vx[1]} {vx[2]}")
    for uv in uv_flat:
        obj_lines.append(f"vt {uv[0]} {uv[1]}")
    obj_lines.append("usemtl material_0")
    for fi in range(F):
        v0, v1, v2 = faces[fi] + 1
        u0, u1, u2 = uv_idx_flat[fi]
        obj_lines.append(f"f {v0}/{u0} {v1}/{u1} {v2}/{u2}")

    with open(str(base_path.with_suffix(".obj")), "w") as f:
        f.write("\n".join(obj_lines) + "\n")

    mtl_str = (
        "newmtl material_0\n"
        "Ka 1.0 1.0 1.0\n"
        "Kd 1.0 1.0 1.0\n"
        "Ks 0.0 0.0 0.0\n"
        f"map_Kd {base_path.name}.png\n"
    )
    with open(str(base_path.with_suffix(".mtl")), "w") as f:
        f.write(mtl_str)


def _save_textured_obj_simple(base_path: Path, verts: np.ndarray, faces: np.ndarray,
                              uv_per_vertex: np.ndarray, image):
    """Fast OBJ writer for meshes with one UV per vertex (xatlas-unwrapped)."""
    import sys
    print(f"  bake: saving PNG ({image.size[0]}x{image.size[1]})…", file=sys.stderr, flush=True)
    image.save(str(base_path.with_suffix(".png")))

    print(f"  bake: writing OBJ ({len(verts)} verts, {len(faces)} faces)…",
          file=sys.stderr, flush=True)

    # Vectorized string assembly — much faster than per-row f-strings
    v_lines = np.char.add(np.char.add(np.char.add(
        np.array(["v "] * len(verts)),
        np.char.add(np.char.add(
            verts[:, 0].astype("U16"), " "),
            verts[:, 1].astype("U16"))), " "),
        verts[:, 2].astype("U16"))
    vt_lines = np.char.add(np.char.add(
        np.array(["vt "] * len(uv_per_vertex)),
        np.char.add(uv_per_vertex[:, 0].astype("U16"), " ")),
        uv_per_vertex[:, 1].astype("U16"))
    f1 = (faces + 1).astype("U10")
    f_lines = np.array(["f "] * len(faces))
    for col in range(3):
        f_lines = np.char.add(f_lines,
                              np.char.add(np.char.add(f1[:, col], "/"), f1[:, col]))
        if col < 2:
            f_lines = np.char.add(f_lines, " ")

    with open(str(base_path.with_suffix(".obj")), "w") as f:
        f.write(f"mtllib {base_path.name}.mtl\n")
        f.write("\n".join(v_lines))
        f.write("\n")
        f.write("\n".join(vt_lines))
        f.write("\nusemtl material_0\n")
        f.write("\n".join(f_lines))
        f.write("\n")

    mtl_str = (
        "newmtl material_0\n"
        "Ka 1.0 1.0 1.0\n"
        "Kd 1.0 1.0 1.0\n"
        "Ks 0.0 0.0 0.0\n"
        f"map_Kd {base_path.name}.png\n"
    )
    with open(str(base_path.with_suffix(".mtl")), "w") as f:
        f.write(mtl_str)


def load_inria_3dgs_ply(path: str | Path, device: str = "cuda") -> dict:
    """Load Inria 3DGS PLY, invert activations (sh0+shN concatenated)."""
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply["vertex"]
    N = v.count

    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)[:, None, :]
    rest = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if len(rest) % 3 != 0:
        raise ValueError(f"f_rest count {len(rest)} is not a multiple of 3")
    K_rest = len(rest) // 3
    f_rest = np.stack([v[n] for n in rest], -1).reshape(N, 3, K_rest).transpose(0, 2, 1)

    opacity = v["opacity"].astype(np.float32)[:, None]
    scales = np.stack([v[f"scale_{i}"] for i in range(3)], -1)
    rots = np.stack([v[f"rot_{i}"] for i in range(4)], -1)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)

    sh = torch.cat([t(f_dc), t(f_rest)], dim=1)  # (N, K, 3)

    return dict(
        means=t(xyz),
        sh=sh,
        opacities=torch.sigmoid(t(opacity)).squeeze(-1),
        scales=torch.exp(t(scales)),
        quats=t(rots) / t(rots).norm(dim=-1, keepdim=True).clamp_min(1e-8),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mesh Gaussians from a PLY")
    p.add_argument("--ply", type=Path, required=True, help="Input Inria 3DGS PLY")
    p.add_argument("--out", type=Path, default=Path("/tmp/mesh.ply"), help="Output mesh PLY")
    p.add_argument("--mode", type=str, default="tsdf", help="tsdf or dlnr")
    p.add_argument("--n-cams", type=int, default=48, help="Fibonacci dome cameras")
    p.add_argument("--alpha-thresh", type=float, default=0.5, help="Alpha threshold")
    p.add_argument("--device", default="cuda", help="torch device")
    args = p.parse_args()

    print(f"Loading {args.ply}...")
    scene = load_inria_3dgs_ply(args.ply, device=args.device)

    print(f"Meshing with {args.n_cams} cameras ({args.mode})...")
    result = generate_mesh(
        means=scene["means"],
        quats=scene["quats"],
        scales=scene["scales"],
        opacities=scene["opacities"],
        sh=scene["sh"],
        device=args.device,
        mode=args.mode,
        n_cams=args.n_cams,
        alpha_thresh=args.alpha_thresh,
        out_path=args.out,
        progress_cb=lambda i, n: print(f"  {i}/{n}", end="\r"),
    )

    print(f"\nDone: {len(result['verts']):,} vertices, {len(result['faces']):,} faces")
