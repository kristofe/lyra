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
    import fvdb
    from fvdb import Grid

    # Scene bounds
    bbmin = means.min(0).values
    bbmax = means.max(0).values
    scene_diag = float((bbmax - bbmin).norm())
    scene_center = 0.5 * (bbmin + bbmax)

    # Invert activations to pre-activation form for fvdb.GaussianSplat3d
    eps = 1e-6
    opac_clamped = opacities.clamp(eps, 1 - eps)
    logit_opacities = torch.log(opac_clamped / (1 - opac_clamped))
    log_scales = torch.log(scales.clamp_min(eps))

    # Extract DC and higher-order SH bands
    sh0 = sh[:, :1, :].contiguous()
    shN = sh[:, 1:, :].contiguous()

    # Create fvdb model
    model = fvdb.GaussianSplat3d.from_tensors(
        means=means,
        quats=quats,
        log_scales=log_scales,
        logit_opacities=logit_opacities,
        sh0=sh0,
        shN=shN,
    )

    # Fibonacci dome: N cameras on a hemisphere, y-down COLMAP convention
    i = torch.arange(n_cams, device=device, dtype=torch.float32)
    golden_ang = np.pi * (3 - np.sqrt(5))
    elev_min_deg, elev_max_deg = 5.0, 70.0
    s_min = float(np.sin(np.deg2rad(elev_min_deg)))
    s_max = float(np.sin(np.deg2rad(elev_max_deg)))
    sin_elev = s_min + (s_max - s_min) * (i + 0.5) / n_cams
    cos_elev = torch.sqrt(torch.clamp(1 - sin_elev**2, min=0))
    az = i * golden_ang
    ring_radius = scene_diag * 0.8
    offset = torch.stack(
        [
            cos_elev * torch.cos(az),
            -sin_elev,
            cos_elev * torch.sin(az),
        ],
        dim=-1,
    )
    cam_centers = scene_center + ring_radius * offset

    # Poses: camera-to-world matrices
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

    # Intrinsics
    fov = 60.0
    fx = fy = 0.5 * W / np.tan(np.deg2rad(fov) * 0.5)
    K = (
        torch.tensor(
            [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device=device, dtype=torch.float32
        )
        .expand(n_cams, 3, 3)
        .contiguous()
    )

    if mode == "tsdf":
        return _mesh_tsdf(
            model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
            alpha_thresh, density, shell_thickness, out_path, progress_cb
        )
    elif mode == "dlnr":
        return _mesh_dlnr(
            model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
            density, shell_thickness, bake_texture, tex_size, out_path, progress_cb
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


def _mesh_dlnr(model, c2w, w2c, K, device, n_cams, H, W, scene_diag,
               density, shell_thickness, bake_texture, tex_size, out_path, progress_cb):
    """DLNR stereo depth meshing with per-vertex colors."""
    from fvdb_reality_capture.tools import mesh_from_splats_dlnr
    from fvdb_reality_capture.sfm_scene import sfm_cache
    import threading

    # Patch FileLock to not use signals (which don't work in non-main threads or with event loops)
    original_enter = sfm_cache.FileLock.__enter__
    original_exit = sfm_cache.FileLock.__exit__

    def patched_enter(self):
        # Skip signal setup, just use basic locking
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
        TRUNC = density * scene_diag
        image_sizes = torch.tensor([[H, W]] * n_cams, dtype=torch.int32, device=device)

        verts_d, faces_d, colors_d = mesh_from_splats_dlnr(
            model=model,
            camera_to_world_matrices=c2w,
            projection_matrices=K,
            image_sizes=image_sizes,
            truncation_margin=TRUNC,
            grid_shell_thickness=shell_thickness,
            baseline=0.15,
            near=1.0,
            far=100.0,
            disparity_reprojection_threshold=10.0,
            alpha_threshold=0.1,
            image_downsample_factor=1,
            use_absolute_baseline=False,
            show_progress=True,
        )

        verts = verts_d.cpu().numpy().astype(np.float32)
        faces = faces_d.cpu().numpy().astype(np.int32)
        faces = faces[:, ::-1]  # Flip winding order to fix inside-out mesh
        colors = colors_d.cpu().numpy().astype(np.float32)

        # Texture bake → returns (uv, tex_image) or (None, None) if disabled/fails
        uv, tex_image = (None, None)
        if bake_texture:
            try:
                uv, tex_image = _bake_texture_fast(verts, faces, colors, tex_size)
            except Exception as e:
                print(f"Texture baking skipped: {e}")
                import traceback; traceback.print_exc()

        # Save mesh files if requested
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _save_ply(out_path.with_stem("mesh_dlnr"), verts, faces, colors=colors)
            if uv is not None and tex_image is not None:
                _save_textured_obj(out_path.with_stem("mesh_dlnr_tex"), verts, faces, uv, tex_image)

        if progress_cb:
            progress_cb(n_cams, n_cams)

        return {
            "verts": verts, "faces": faces,
            "normals": None, "colors": colors,
            "uv": uv, "tex_image": tex_image,
        }
    finally:
        # Restore original FileLock methods
        sfm_cache.FileLock.__enter__ = original_enter
        sfm_cache.FileLock.__exit__ = original_exit


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
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    verts: np.ndarray,
    faces: np.ndarray,
    tex_size: int = 2048,
    n_cams: int = 96,
    image_size: int = 1024,
    device: str = "cuda",
    out_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Photogrammetric texture bake — re-renders splats and projects each texel
    onto the best-visible camera. Returns dict with uv, tex_image, verts, faces."""
    import fvdb
    import xatlas
    from scipy import ndimage
    from PIL import Image

    # 1) Build splat model (invert activations)
    eps = 1e-6
    opac_clamped = opacities.clamp(eps, 1 - eps)
    logit_opacities = torch.log(opac_clamped / (1 - opac_clamped))
    log_scales = torch.log(scales.clamp_min(eps))
    sh0 = sh[:, :1, :].contiguous()
    shN = sh[:, 1:, :].contiguous()
    model = fvdb.GaussianSplat3d.from_tensors(
        means=means, quats=quats,
        log_scales=log_scales, logit_opacities=logit_opacities,
        sh0=sh0, shN=shN,
    )

    # 2) Build dome cameras (same geometry as generate_mesh)
    bbmin = means.min(0).values
    bbmax = means.max(0).values
    scene_diag = float((bbmax - bbmin).norm())
    scene_center = 0.5 * (bbmin + bbmax)

    i_t = torch.arange(n_cams, device=device, dtype=torch.float32)
    golden_ang = np.pi * (3 - np.sqrt(5))
    elev_min_deg, elev_max_deg = 5.0, 70.0
    s_min = float(np.sin(np.deg2rad(elev_min_deg)))
    s_max = float(np.sin(np.deg2rad(elev_max_deg)))
    sin_elev = s_min + (s_max - s_min) * (i_t + 0.5) / n_cams
    cos_elev = torch.sqrt(torch.clamp(1 - sin_elev**2, min=0))
    az = i_t * golden_ang
    ring_radius = scene_diag * 0.8
    offset = torch.stack([cos_elev * torch.cos(az), -sin_elev, cos_elev * torch.sin(az)], dim=-1)
    cam_centers = scene_center + ring_radius * offset

    def look_at(eyes, target):
        world_up = torch.tensor([0., -1., 0.], device=device)
        fwd = target - eyes
        fwd = fwd / fwd.norm(dim=-1, keepdim=True)
        up_ref = world_up.expand_as(fwd).clone()
        parallel = (fwd * up_ref).sum(-1, keepdim=True).abs() > 0.98
        fallback = torch.tensor([1., 0., 0.], device=device).expand_as(fwd)
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
    fov = 60.0
    fx = fy = 0.5 * image_size / np.tan(np.deg2rad(fov) * 0.5)
    K = torch.tensor([[fx, 0, image_size/2], [0, fy, image_size/2], [0, 0, 1]],
                     device=device, dtype=torch.float32).expand(n_cams, 3, 3).contiguous()

    # 3) Render RGB + depth from each camera (linear RGB)
    import sys, time
    from tqdm import tqdm
    cam_rgb = torch.zeros(n_cams, image_size, image_size, 3, dtype=torch.float32, device=device)
    cam_depth = torch.zeros(n_cams, image_size, image_size, dtype=torch.float32, device=device)
    for i in tqdm(range(n_cams), desc="bake: render cameras", file=sys.stderr):
        feat_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=w2c[i:i+1], projection_matrices=K[i:i+1],
            image_width=image_size, image_height=image_size, near=0.0, far=1e10,
        )
        a = alpha[0, ..., 0].clamp_min(1e-10)
        cam_rgb[i] = (feat_depth[0, ..., :3] / a.unsqueeze(-1)).float()
        cam_depth[i] = (feat_depth[0, ..., -1] / a).float()
        torch.cuda.synchronize()
        if progress_cb:
            progress_cb(i + 1, n_cams + 3)

    # 4) UV unwrap the mesh
    t_uv = time.perf_counter()
    print(f"  bake: starting xatlas UV unwrap for {faces.shape[0]} faces…", file=sys.stderr, flush=True)
    vmap, F_uv, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    uvs = uvs.astype(np.float32)
    print(f"  bake: xatlas done in {time.perf_counter() - t_uv:.1f}s", file=sys.stderr, flush=True)

    # 5) For each face, find best camera (highest face_normal · view_to_face)
    verts_t = torch.from_numpy(verts.astype(np.float32)).to(device)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device)
    tri_verts = verts_t[faces_t]                          # (F, 3, 3)
    tri_centroid = tri_verts.mean(dim=1)                  # (F, 3)
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    tri_normal = torch.cross(e1, e2, dim=-1)
    tri_normal = tri_normal / tri_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    cam_centers_dev = c2w[:, :3, 3]                       # (C, 3)
    # (F, C, 3): vector from face centroid to each camera
    view_vec = cam_centers_dev[None, :, :] - tri_centroid[:, None, :]
    view_vec_n = view_vec / view_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Dot with face normal — higher = camera looks more head-on at the face
    score = (view_vec_n * tri_normal[:, None, :]).sum(-1)  # (F, C)
    best_cam = score.argmax(dim=1).cpu().numpy()           # (F,)
    print(f"  bake: best-camera selection done ({faces.shape[0]} faces)", file=sys.stderr, flush=True)
    if progress_cb:
        progress_cb(n_cams + 1, n_cams + 3)

    # 6) GPU-batched per-texel rasterization
    F = faces.shape[0]
    tri_uv_t = torch.from_numpy((uvs[F_uv] * (tex_size - 1)).astype(np.float32)).to(device)  # (F, 3, 2)
    best_cam_t = torch.from_numpy(best_cam).long().to(device)                                 # (F,)

    # Precompute per-face UV bbox & barycentric basis (vectorized once)
    uv_min = tri_uv_t.min(dim=1).values                  # (F, 2)
    uv_max = tri_uv_t.max(dim=1).values                  # (F, 2)

    tex_t = torch.zeros((tex_size, tex_size, 3), dtype=torch.float32, device=device)
    mask_t = torch.zeros((tex_size, tex_size), dtype=torch.bool, device=device)

    # Process faces in chunks. We pad each chunk's pixel-grid to the max bbox size
    # so the heavy ops are vectorized per chunk.
    CHUNK = 2048
    n_chunks = (F + CHUNK - 1) // CHUNK
    print(f"  bake: rasterizing {F} faces in {n_chunks} chunks of up to {CHUNK}…", file=sys.stderr, flush=True)
    chunk_iter = tqdm(range(n_chunks), desc="bake: rasterize chunks", file=sys.stderr)
    for c in chunk_iter:
        s, e = c * CHUNK, min(F, (c + 1) * CHUNK)
        ch_size = e - s
        ch_uv = tri_uv_t[s:e]                            # (B, 3, 2)
        ch_verts = tri_verts[s:e]                        # (B, 3, 3) GPU tensor
        ch_cam = best_cam_t[s:e]                         # (B,)
        ch_min = uv_min[s:e]                             # (B, 2)
        ch_max = uv_max[s:e]                             # (B, 2)

        # Bbox dims, clipped to texture
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
        mem_est_mb = ch_size * H_pad * W_pad * (3 + 1 + 2 + 3) * 4 / (1024 * 1024)
        chunk_iter.set_postfix(pad=f"{H_pad}x{W_pad}", mem_mb=f"{mem_est_mb:.0f}")

        # Build padded pixel grid per face: (B, H_pad, W_pad, 2) of pixel xy
        dx = torch.arange(W_pad, device=device).view(1, 1, W_pad)
        dy = torch.arange(H_pad, device=device).view(1, H_pad, 1)
        px_x = xmin.view(ch_size, 1, 1) + dx              # (B, 1, W_pad)
        px_y = ymin.view(ch_size, 1, 1) + dy              # (B, H_pad, 1)
        # Broadcast to (B, H_pad, W_pad)
        px_x = px_x.expand(ch_size, H_pad, W_pad).contiguous()
        px_y = px_y.expand(ch_size, H_pad, W_pad).contiguous()
        valid_pad = (dx < bw.view(ch_size, 1, 1)) & (dy < bh.view(ch_size, 1, 1))

        # Barycentric in UV (vectorized over chunk)
        uv0 = ch_uv[:, 0].view(ch_size, 1, 1, 2)
        uv1 = ch_uv[:, 1].view(ch_size, 1, 1, 2)
        uv2 = ch_uv[:, 2].view(ch_size, 1, 1, 2)
        p = torch.stack([px_x.float(), px_y.float()], dim=-1)  # (B, H_pad, W_pad, 2)
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

        # 3D positions via barycentric of face vertices
        v0 = ch_verts[:, 0].view(ch_size, 1, 1, 3)
        v1 = ch_verts[:, 1].view(ch_size, 1, 1, 3)
        v2 = ch_verts[:, 2].view(ch_size, 1, 1, 3)
        pos3d = u_b.unsqueeze(-1) * v0 + v_b.unsqueeze(-1) * v1 + w_b.unsqueeze(-1) * v2

        # Project to each face's best camera
        cam_idx = ch_cam.view(ch_size, 1, 1).expand(ch_size, H_pad, W_pad)
        W_mats = w2c[cam_idx]                            # (B, H, W, 4, 4)
        Ks = K[cam_idx]                                  # (B, H, W, 3, 3)
        ones = torch.ones_like(pos3d[..., :1])
        pos_h = torch.cat([pos3d, ones], dim=-1).unsqueeze(-1)   # (..., 4, 1)
        pos_cam = (W_mats @ pos_h).squeeze(-1)[..., :3]          # (..., 3)
        z = pos_cam[..., 2]
        ok_z = z > 1e-6
        proj = (Ks @ (pos_cam / z.unsqueeze(-1)).unsqueeze(-1)).squeeze(-1)
        ix = proj[..., 0].round().long()
        iy = proj[..., 1].round().long()
        ok_xy = ok_z & (ix >= 0) & (ix < image_size) & (iy >= 0) & (iy < image_size)
        valid = inside & ok_xy
        if not valid.any():
            continue

        # Gather RGB at the projected pixels (per-pixel best camera lookup)
        flat_cam = cam_idx[valid]
        flat_iy = iy[valid].clamp(0, image_size - 1)
        flat_ix = ix[valid].clamp(0, image_size - 1)
        rgb = cam_rgb[flat_cam, flat_iy, flat_ix]        # (M, 3)

        flat_py = px_y[valid].clamp(0, tex_size - 1)
        flat_px = px_x[valid].clamp(0, tex_size - 1)
        tex_t[flat_py, flat_px] = rgb
        mask_t[flat_py, flat_px] = True

        torch.cuda.synchronize()
        if progress_cb:
            progress_cb(n_cams + 1 + c, n_cams + n_chunks + 2)

    tex = tex_t.cpu().numpy()
    mask = mask_t.cpu().numpy()

    total = n_cams + n_chunks + 2

    if progress_cb:
        progress_cb(total - 1, total)

    if not mask.all():
        t_fill = time.perf_counter()
        print(f"  bake: filling UV gaps ({(~mask).sum()} pixels)…", file=sys.stderr, flush=True)
        _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
        for ch in range(3):
            tex[..., ch] = tex[..., ch][indices[0], indices[1]]
        print(f"  bake: gap fill done in {time.perf_counter() - t_fill:.1f}s", file=sys.stderr, flush=True)

    t_enc = time.perf_counter()
    tex_srgb = _linear_to_srgb(tex)
    tex_u8 = np.clip(tex_srgb * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(tex_u8)
    print(f"  bake: sRGB encode + PIL image in {time.perf_counter() - t_enc:.2f}s", file=sys.stderr, flush=True)

    # Return the xatlas-unwrapped mesh directly (one UV per vertex, no duplication)
    # The viewer can build a trimesh with these without 3x vertex inflation.
    unwrapped_verts = verts[vmap].astype(np.float32)
    unwrapped_faces = F_uv.astype(np.int32)

    if out_path is not None:
        t_save = time.perf_counter()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_textured_obj_simple(
            out_path.with_stem("mesh_splat_tex"),
            unwrapped_verts, unwrapped_faces, uvs, img,
        )
        print(f"  bake: OBJ/MTL/PNG save in {time.perf_counter() - t_save:.1f}s", file=sys.stderr, flush=True)

    if progress_cb:
        progress_cb(total, total)

    # NOTE: returns the xatlas-unwrapped mesh (one UV per vertex). The viewer
    # should use these `verts`/`faces`/`uv` directly to avoid 3x vertex inflation.
    return {
        "uv": uvs.astype(np.float32),       # (V_uv, 2)
        "tex_image": img,
        "verts": unwrapped_verts,           # (V_uv, 3)
        "faces": unwrapped_faces,           # (F, 3)
        "per_vertex_uv": True,              # signals new layout to _push_mesh_to_viser
    }


def bake_mesh_texture(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    tex_size: int = 1024,
    out_path: Path | None = None,
) -> dict:
    """Public API: bake per-vertex colors into a texture; optionally write OBJ/MTL/PNG.

    Returns dict with: "uv" (F, 3, 2), "tex_image" (PIL.Image), "verts", "faces".
    """
    uv, tex_image = _bake_texture_fast(verts, faces, colors, tex_size)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_textured_obj(out_path.with_stem("mesh_dlnr_tex"), verts, faces, uv, tex_image)
    return {"uv": uv, "tex_image": tex_image, "verts": verts, "faces": faces}


def _bake_texture_fast(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, tex_size: int = 1024):
    """Vectorized texture baking. Returns (per_corner_uvs, PIL.Image).

    `per_corner_uvs` has shape (F, 3, 2) — UV per face corner — used to build
    the textured trimesh for viser. Image is RGB uint8.
    """
    import xatlas
    from scipy import ndimage
    from PIL import Image

    # UV unwrap (xatlas returns vmap so vertex i in unwrapped mesh maps from verts[vmap[i]])
    vmap, F_uv, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    uvs = uvs.astype(np.float32)  # (V_uv, 2) in [0,1]

    # Colors on the unwrapped vertices (use vmap to look up original-vertex color)
    uv_colors = colors[vmap].astype(np.float32)  # (V_uv, 3)

    # Per-face UV triangle (image-space pixels). Use texel-center sampling.
    # tex_size pixels indexed 0..tex_size-1; UV [0,1] maps via *(tex_size-1).
    tri_uv = uvs[F_uv] * (tex_size - 1)         # (F, 3, 2)
    tri_col = uv_colors[F_uv]                    # (F, 3, 3)

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    mask = np.zeros((tex_size, tex_size), dtype=bool)

    # Vectorized per-face rasterization
    for fi in range(tri_uv.shape[0]):
        uv0, uv1, uv2 = tri_uv[fi]
        c0, c1, c2 = tri_col[fi]

        xmin = max(0, int(np.floor(min(uv0[0], uv1[0], uv2[0]))))
        xmax = min(tex_size - 1, int(np.ceil(max(uv0[0], uv1[0], uv2[0]))))
        ymin = max(0, int(np.floor(min(uv0[1], uv1[1], uv2[1]))))
        ymax = min(tex_size - 1, int(np.ceil(max(uv0[1], uv1[1], uv2[1]))))
        if xmax < xmin or ymax < ymin:
            continue

        # Pixel grid for the bbox
        xs = np.arange(xmin, xmax + 1, dtype=np.float32)
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)  # both (H, W)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N, 2)

        # Barycentric (vectorized)
        v0v1 = uv1 - uv0
        v0v2 = uv2 - uv0
        v0p = pts - uv0
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = v0p @ v0v1
        d21 = v0p @ v0v2
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        inside = (u >= -1e-5) & (v >= -1e-5) & (w >= -1e-5)
        if not inside.any():
            continue

        u, v, w = u[inside], v[inside], w[inside]
        col = (u[:, None] * c0) + (v[:, None] * c1) + (w[:, None] * c2)  # (M, 3)
        px = pts[inside].astype(np.int32)
        tex[px[:, 1], px[:, 0]] = col
        mask[px[:, 1], px[:, 0]] = True

    # Fill UV-island gaps via nearest-neighbor distance transform
    if not mask.all():
        _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
        for ch in range(3):
            tex[..., ch] = tex[..., ch][indices[0], indices[1]]

    # Encode linear → sRGB before saving as PNG.
    # Three.js treats PNG textures as sRGB and decodes to linear during shading,
    # so storing linear values would double-darken them.
    tex_srgb = _linear_to_srgb(tex)
    tex_u8 = np.clip(tex_srgb * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(tex_u8)

    # Per-corner UV array for the trimesh (V_uv-indexed UVs aligned to F_uv)
    per_corner_uvs = uvs[F_uv].astype(np.float32)  # (F, 3, 2)
    return per_corner_uvs, img


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
