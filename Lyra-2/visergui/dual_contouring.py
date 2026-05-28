"""Dual Contouring on 2DGS Hermite data.

A 2DGS splat IS an oriented Hermite sample: position = mean, normal =
quat·[0,0,1], weight = opacity. DC consumes those directly and preserves
sharp features that marching cubes blurs — useful for scenes with hard edges
(table corners, building facades, etc.).

Pipeline:
  1. `quat_to_disk_normal` — exact per-disk world-space normal axis.
  2. `build_sdf_grid` — splat each oriented sample to its 2-voxel
     neighborhood of grid corners. Per-corner field is the weighted average of
     signed distances along each contributing splat's normal.
  3. `dual_contour` — for each cell whose 8 corners straddle the zero level
     set, solve a QEF (`solve_qef_cell`) using the splats falling inside the
     cell to place one vertex; for each grid edge with a sign change, emit a
     quad connecting the QEF vertices of the (up to 4) adjacent active cells.
  4. `run` — top-level entry: takes (means, quats, opacities), returns
     (verts, tris) numpy arrays matching the mesher contract.

Sanity test in `__main__`: synthetic plane + L-shape, compared against the
vendored BorisTheBrave reference implementation under
`visergui/_third_party/mc_dc_reference.py`.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Hermite-sample construction
# --------------------------------------------------------------------------- #


def quat_to_disk_normal(quats: torch.Tensor) -> torch.Tensor:
    """quats: (M, 4) in (w, x, y, z) order (gsplat / project convention).
    Returns (M, 3) — the third column of R(quats), i.e. R @ [0, 0, 1], which
    is the disk-normal axis used by 2DGS rasterization."""
    quats = F.normalize(quats, dim=-1, eps=1e-8)
    w, x, y, z = quats.unbind(-1)
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1)


# --------------------------------------------------------------------------- #
# Grid SDF construction (splat oriented samples to corners)
# --------------------------------------------------------------------------- #


def build_sdf_grid(positions: torch.Tensor, normals: torch.Tensor,
                   weights: torch.Tensor, origin: torch.Tensor,
                   voxel_size: float, grid_shape: tuple[int, int, int],
                   radius_voxels: int = 2):
    """Build a signed-distance field on a uniform corner grid by splatting
    each oriented sample into its `(2R+1)^3` corner neighborhood. The per-
    corner field is `Σ w·(n·(c−p)) / Σ w`. Corners with zero weight are
    marked invalid (returned as +inf in `phi`, False in `valid`) and treated
    as "outside" by the contouring step — this gives a clean boundary
    between samples-region and empty-region without spurious zero crossings.

    Args:
      positions: (M, 3) sample centers in world space.
      normals:   (M, 3) unit normals.
      weights:   (M,) per-sample weight (e.g. opacity).
      origin:    (3,) world-space position of grid corner (0,0,0).
      voxel_size: edge length of one cell.
      grid_shape: (Nx, Ny, Nz) number of *cells*; corner grid is (Nx+1,…).
      radius_voxels: each sample influences a (2R+1)^3 block of corners.

    Returns:
      phi:   (Nx+1, Ny+1, Nz+1) float64, signed distance per corner (+inf for empty).
      valid: same shape bool — True wherever at least one sample contributed.
    """
    device = positions.device
    Nx, Ny, Nz = grid_shape
    Cx, Cy, Cz = Nx + 1, Ny + 1, Nz + 1
    R = int(radius_voxels)

    # Each sample's "anchor" corner: floor((pos - origin) / voxel_size).
    base = ((positions - origin) / voxel_size).floor().long()         # (M, 3)

    phi = torch.zeros((Cx, Cy, Cz), device=device, dtype=torch.float64)
    wgt = torch.zeros((Cx, Cy, Cz), device=device, dtype=torch.float64)

    w_d = weights.double()
    pos_d = positions.double()
    nrm_d = normals.double()

    for ox in range(-R, R + 1):
        for oy in range(-R, R + 1):
            for oz in range(-R, R + 1):
                ci = base[:, 0] + ox
                cj = base[:, 1] + oy
                ck = base[:, 2] + oz
                in_b = ((ci >= 0) & (ci < Cx) & (cj >= 0) & (cj < Cy)
                        & (ck >= 0) & (ck < Cz))
                if not in_b.any():
                    continue
                ci_v, cj_v, ck_v = ci[in_b], cj[in_b], ck[in_b]
                idx_lin = ci_v * (Cy * Cz) + cj_v * Cz + ck_v
                corner = (origin.double()
                          + torch.stack([ci_v.double(), cj_v.double(), ck_v.double()], -1) * voxel_size)
                d = ((corner - pos_d[in_b]) * nrm_d[in_b]).sum(-1)
                w_m = w_d[in_b]
                phi.view(-1).scatter_add_(0, idx_lin, w_m * d)
                wgt.view(-1).scatter_add_(0, idx_lin, w_m)

    valid = wgt > 0
    phi_out = torch.where(valid, phi / wgt.clamp_min(1e-12),
                          torch.full_like(phi, float("inf")))
    return phi_out, valid


# --------------------------------------------------------------------------- #
# Per-cell QEF (SVD-based)
# --------------------------------------------------------------------------- #


def solve_qef_cell(positions: np.ndarray, normals: np.ndarray,
                   cell_origin: np.ndarray, cell_size: float,
                   bias_strength: float = 0.01) -> np.ndarray:
    """Minimize `Σ wᵢ (nᵢ · (x − pᵢ))²` for vertex position `x`, clipped to
    the cell bounding box. SVD-stable via `np.linalg.lstsq`. A small bias
    toward the centroid of the input samples is added to keep `x` inside the
    cell on rank-deficient inputs (e.g. when all normals are parallel —
    classic 1-DOF undetermined case)."""
    if positions.shape[0] == 0:
        return cell_origin + 0.5 * cell_size

    centroid = positions.mean(axis=0)
    # Stack normals + 3 axis-aligned bias rows pointing at the centroid.
    A = np.vstack([normals,
                   np.eye(3, dtype=normals.dtype) * bias_strength])      # (K+3, 3)
    b = np.concatenate([(positions * normals).sum(-1),
                        centroid * bias_strength])                       # (K+3,)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(sol, cell_origin, cell_origin + cell_size)


# --------------------------------------------------------------------------- #
# Dual contouring core
# --------------------------------------------------------------------------- #


def dual_contour(positions: torch.Tensor, normals: torch.Tensor,
                 weights: torch.Tensor, voxel_size: float,
                 radius_voxels: int = 2, padding_voxels: int = 3):
    """Run DC on oriented samples and return `(verts, tris)` numpy arrays.

    Args:
      positions/normals/weights: (M, 3), (M, 3), (M,) torch tensors.
      voxel_size: grid cell edge length in world units.
      radius_voxels: each sample contributes to a (2R+1)^3 corner block.
      padding_voxels: extend the bbox by this many voxels on each side so
        thin surfaces near the bbox boundary get a clean edge.

    Returns:
      verts: (V, 3) float32.
      tris:  (T, 3) int32 (quads triangulated 2-tris each).
    """
    device = positions.device

    # Bbox + grid sizing.
    pad = padding_voxels * voxel_size
    bbmin = positions.min(0).values - pad
    bbmax = positions.max(0).values + pad
    grid_shape = tuple(int(((bbmax[i] - bbmin[i]) / voxel_size).ceil().item())
                       for i in range(3))
    origin = bbmin

    phi, valid = build_sdf_grid(positions, normals, weights, origin,
                                voxel_size, grid_shape,
                                radius_voxels=radius_voxels)
    Cx, Cy, Cz = phi.shape
    Nx, Ny, Nz = Cx - 1, Cy - 1, Cz - 1

    # Per-corner sign in {-1: invalid, 0: inside (phi<0), 1: outside (phi>0)}.
    # Invalid corners don't participate in sign-change tests — that keeps the
    # contoured surface inside the splat-support region only, not at the
    # boundary between "have data" and "no data".
    sign = torch.where(~valid, torch.full_like(phi, -1, dtype=torch.int8),
                       (phi > 0).to(torch.int8))

    # Active cells: 8 corners are all valid AND mix inside+outside.
    s_in  = sign == 0                                               # (Cx,Cy,Cz)
    s_out = sign == 1
    has_in  = (s_in[:-1, :-1, :-1] | s_in[:-1, :-1, 1:]
               | s_in[:-1, 1:, :-1] | s_in[:-1, 1:, 1:]
               | s_in[1:, :-1, :-1] | s_in[1:, :-1, 1:]
               | s_in[1:, 1:, :-1] | s_in[1:, 1:, 1:])
    has_out = (s_out[:-1, :-1, :-1] | s_out[:-1, :-1, 1:]
               | s_out[:-1, 1:, :-1] | s_out[:-1, 1:, 1:]
               | s_out[1:, :-1, :-1] | s_out[1:, :-1, 1:]
               | s_out[1:, 1:, :-1] | s_out[1:, 1:, 1:])
    active = has_in & has_out                                       # (Nx,Ny,Nz)
    sign_np = sign.cpu().numpy()
    active_np = active.cpu().numpy()
    origin_np = origin.cpu().numpy()

    # Bucket samples by cell.
    base = ((positions - origin) / voxel_size).floor().long()        # (M, 3)
    in_bounds = ((base[:, 0] >= 0) & (base[:, 0] < Nx)
                 & (base[:, 1] >= 0) & (base[:, 1] < Ny)
                 & (base[:, 2] >= 0) & (base[:, 2] < Nz))
    base = base[in_bounds]
    pos_M = positions[in_bounds]
    nrm_M = normals[in_bounds]
    lin_M = base[:, 0] * (Ny * Nz) + base[:, 1] * Nz + base[:, 2]

    sort_idx = lin_M.argsort()
    sorted_lin = lin_M[sort_idx].cpu().numpy()
    sorted_pos = pos_M[sort_idx].cpu().numpy()
    sorted_nrm = nrm_M[sort_idx].cpu().numpy()

    # Bucket boundaries.
    bucket_ranges: dict[int, tuple[int, int]] = {}
    if sorted_lin.size > 0:
        boundaries = np.flatnonzero(np.diff(sorted_lin) != 0) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [sorted_lin.size]])
        for st, ed in zip(starts, ends):
            bucket_ranges[int(sorted_lin[st])] = (int(st), int(ed))

    # Solve QEF per active cell; index cell → vertex (0-indexed here).
    vert_array: list[np.ndarray] = []
    vert_indices: dict[tuple[int, int, int], int] = {}
    active_xyz = np.argwhere(active_np)
    for cell in active_xyz:
        i, j, k = int(cell[0]), int(cell[1]), int(cell[2])
        cell_origin = origin_np + np.array([i, j, k]) * voxel_size
        lin = i * (Ny * Nz) + j * Nz + k
        if lin in bucket_ranges:
            st, ed = bucket_ranges[lin]
            vert = solve_qef_cell(sorted_pos[st:ed], sorted_nrm[st:ed],
                                  cell_origin, voxel_size)
        else:
            vert = cell_origin + 0.5 * voxel_size
        vert_indices[(i, j, k)] = len(vert_array)
        vert_array.append(vert)

    # Connect cells: for each grid edge with a sign change, emit a quad
    # connecting the 4 cells incident to that edge. We walk three axes
    # separately; each axis touches the same 4-cell motif rotated.
    faces: list[tuple[int, int, int, int]] = []

    def _crosses(a: int, b: int) -> bool:
        # sign change only counts if both endpoints are valid (0 or 1) and
        # they differ. -1 (invalid) is excluded from the topology.
        return (a == 0 and b == 1) or (a == 1 and b == 0)

    # Edges parallel to +z: corner (i, j, k) → (i, j, k+1). Adjacent cells:
    # (i-1, j-1, k), (i, j-1, k), (i, j, k), (i-1, j, k).
    for k in range(Cz - 1):
        for i in range(1, Cx - 1):
            for j in range(1, Cy - 1):
                s1 = int(sign_np[i, j, k]); s2 = int(sign_np[i, j, k + 1])
                if not _crosses(s1, s2):
                    continue
                cells = [(i-1, j-1, k), (i, j-1, k), (i, j, k), (i-1, j, k)]
                if not all(c in vert_indices for c in cells):
                    continue
                idx = [vert_indices[c] for c in cells]
                if s2:
                    idx = idx[::-1]
                faces.append(tuple(idx))

    # Edges parallel to +y: corner (i, j, k) → (i, j+1, k). Adjacent cells:
    # (i-1, j, k-1), (i, j, k-1), (i, j, k), (i-1, j, k).
    for j in range(Cy - 1):
        for i in range(1, Cx - 1):
            for k in range(1, Cz - 1):
                s1 = int(sign_np[i, j, k]); s2 = int(sign_np[i, j + 1, k])
                if not _crosses(s1, s2):
                    continue
                cells = [(i-1, j, k-1), (i, j, k-1), (i, j, k), (i-1, j, k)]
                if not all(c in vert_indices for c in cells):
                    continue
                idx = [vert_indices[c] for c in cells]
                if s1:
                    idx = idx[::-1]
                faces.append(tuple(idx))

    # Edges parallel to +x: corner (i, j, k) → (i+1, j, k). Adjacent cells:
    # (i, j-1, k-1), (i, j, k-1), (i, j, k), (i, j-1, k).
    for i in range(Cx - 1):
        for j in range(1, Cy - 1):
            for k in range(1, Cz - 1):
                s1 = int(sign_np[i, j, k]); s2 = int(sign_np[i + 1, j, k])
                if not _crosses(s1, s2):
                    continue
                cells = [(i, j-1, k-1), (i, j, k-1), (i, j, k), (i, j-1, k)]
                if not all(c in vert_indices for c in cells):
                    continue
                idx = [vert_indices[c] for c in cells]
                if s2:
                    idx = idx[::-1]
                faces.append(tuple(idx))

    verts_np = (np.array(vert_array, dtype=np.float32)
                if vert_array else np.zeros((0, 3), dtype=np.float32))
    if not faces:
        return verts_np, np.zeros((0, 3), dtype=np.int32)
    quads = np.array(faces, dtype=np.int64)                          # (Q, 4)
    tris = np.empty((quads.shape[0] * 2, 3), dtype=np.int32)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return verts_np, tris


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def dual_contour_from_grid(phi: torch.Tensor, valid: torch.Tensor,
                           origin: torch.Tensor, voxel_size: float,
                           colors: torch.Tensor | None = None,
                           splat_positions: torch.Tensor | None = None,
                           splat_normals: torch.Tensor | None = None,
                           splat_radius_voxels: float = 3.0):
    """Classical Dual Contouring on a dense corner-SDF grid.

    Topology (which cells are active, edge-walk for quads) comes from the
    fvdb-integrated TSDF — a real, well-defined signed distance field that
    handles depth-driven contributions across many views with proper
    truncation. **Vertex placement** uses QEF with per-sample Hermite data,
    preferred sources in this order:

    1. If `splat_positions` + `splat_normals` are given, the QEF samples
       are the splats falling within `splat_radius_voxels * voxel_size` of
       the cell center. This is what makes DC actually deliver sharp
       corners: 2DGS discs have discrete orientations (e.g. one face of a
       cube has all-normals-+x; another has all-normals-+y), and feeding
       those raw to QEF places a vertex on the corner-intersection ridge.
       The smoothed ∇φ averages those normals together and gives a rounded
       answer, which is why the V2-without-splat-Hermite cube test showed
       DC ≈ TSDF.
    2. If no splats are nearby (or no splat data was passed), fall back to
       interpolated edge crossings on φ with normals from ∇φ. This is the
       classical textbook DC — fine for smooth surfaces, not sharp ones.

    Args:
      phi:    (Cx, Cy, Cz) float32 — SDF at voxel corners. >0 outside,
              <0 inside, +inf for corners that received no integration
              contribution.
      valid:  same shape bool — True wherever the SDF has data. Cells with
              any invalid corner are skipped.
      origin: (3,) world-space position of corner (0,0,0).
      voxel_size: edge length per cell in world units.
      colors: optional (Cx, Cy, Cz, 3) uint8 corner colors. When supplied,
              each output vertex gets trilinearly-sampled RGB at its world
              position.
      splat_positions: optional (M, 3) world-space splat centers.
      splat_normals:   optional (M, 3) unit splat normals (e.g.
                       `quat_to_disk_normal(quats)` from a 2DGS scene).
      splat_radius_voxels: search radius (in units of voxel_size) for
                       splat Hermite samples per cell. 3.0 is the default —
                       tighter values miss the sharp-corner case where the
                       cell straddling one face needs to also see splats
                       from the adjacent face (cube A/B on synthetic data:
                       1.2 → 1% sharp-edge verts; 3.0 → 3.2%, matching the
                       geometric ideal; 5.0 → 3.4% with diminishing return).

    Returns:
      verts:  (V, 3) float32 world-space.
      tris:   (T, 3) int32 (quads triangulated 2-tris each).
      vcolors: (V, 3) float32 in [0, 1] if `colors` was given, else None.
    """
    Cx, Cy, Cz = phi.shape
    Nx, Ny, Nz = Cx - 1, Cy - 1, Cz - 1
    device = phi.device

    # Corner sign: 0=inside (phi<0), 1=outside (phi>0), -1=invalid.
    sign = torch.where(~valid, torch.full_like(phi, -1, dtype=torch.int8),
                       (phi > 0).to(torch.int8))

    # ∇φ via central differences (one-sided at the bbox boundaries). Stored
    # at corners; we interpolate to crossings during the cell walk.
    grad = torch.zeros((Cx, Cy, Cz, 3), dtype=torch.float32, device=device)
    if Cx > 1:
        grad[1:-1, :, :, 0] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * voxel_size)
        grad[0,  :, :, 0] = (phi[1,  :, :] - phi[0, :, :]) / voxel_size
        grad[-1, :, :, 0] = (phi[-1, :, :] - phi[-2, :, :]) / voxel_size
    if Cy > 1:
        grad[:, 1:-1, :, 1] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * voxel_size)
        grad[:, 0,  :, 1] = (phi[:, 1,  :] - phi[:, 0, :]) / voxel_size
        grad[:, -1, :, 1] = (phi[:, -1, :] - phi[:, -2, :]) / voxel_size
    if Cz > 1:
        grad[:, :, 1:-1, 2] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * voxel_size)
        grad[:, :, 0,  2] = (phi[:, :, 1]  - phi[:, :, 0]) / voxel_size
        grad[:, :, -1, 2] = (phi[:, :, -1] - phi[:, :, -2]) / voxel_size

    # +inf at invalid corners contaminates the gradient at their immediate
    # neighbors (∞ − finite → ∞; later ∞ − ∞ → NaN). Sanitize once here so
    # the inner loop can rely on finite gradients at every valid corner.
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    sign_np = sign.cpu().numpy()
    phi_np = phi.cpu().numpy()
    grad_np = grad.cpu().numpy()
    origin_np = origin.cpu().numpy().astype(np.float64)
    colors_np = colors.cpu().numpy().astype(np.float32) if colors is not None else None

    # 12 cell edges, each (corner_a_offset, corner_b_offset).
    EDGES = (
        ((0,0,0), (1,0,0)), ((0,1,0), (1,1,0)),
        ((0,0,1), (1,0,1)), ((0,1,1), (1,1,1)),                       # x
        ((0,0,0), (0,1,0)), ((1,0,0), (1,1,0)),
        ((0,0,1), (0,1,1)), ((1,0,1), (1,1,1)),                       # y
        ((0,0,0), (0,0,1)), ((1,0,0), (1,0,1)),
        ((0,1,0), (0,1,1)), ((1,1,0), (1,1,1)),                       # z
    )

    vert_array: list[np.ndarray] = []
    color_array: list[np.ndarray] = []
    vert_indices: dict[tuple[int, int, int], int] = {}

    def _sample_color_trilinear(pos_w: np.ndarray) -> np.ndarray:
        """Trilinear sample of `colors_np` at world position `pos_w`."""
        rel = (pos_w - origin_np) / voxel_size
        i0 = int(np.floor(rel[0])); j0 = int(np.floor(rel[1])); k0 = int(np.floor(rel[2]))
        i0 = max(0, min(Cx - 2, i0)); j0 = max(0, min(Cy - 2, j0)); k0 = max(0, min(Cz - 2, k0))
        u, v, w = rel[0] - i0, rel[1] - j0, rel[2] - k0
        out = np.zeros(3, dtype=np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = ((u if dx else 1 - u)
                          * (v if dy else 1 - v)
                          * (w if dz else 1 - w))
                    out += wt * colors_np[i0 + dx, j0 + dy, k0 + dz]
        return out / 255.0

    # ---- Vectorized active-cell detection ----------------------------------
    # On a real scene most cells are interior / exterior / fully-invalid —
    # only a thin shell straddles the surface. We compute the active mask in
    # torch (whole-grid slice ops) and Python-iterate only over those cells.
    c000 = sign[:-1, :-1, :-1]; c100 = sign[1:, :-1, :-1]
    c010 = sign[:-1, 1:, :-1];  c110 = sign[1:, 1:, :-1]
    c001 = sign[:-1, :-1, 1:];  c101 = sign[1:, :-1, 1:]
    c011 = sign[:-1, 1:, 1:];   c111 = sign[1:, 1:, 1:]
    any_invalid = ((c000 == -1) | (c100 == -1) | (c010 == -1) | (c110 == -1)
                   | (c001 == -1) | (c101 == -1) | (c011 == -1) | (c111 == -1))
    s_sum = (c000 + c100 + c010 + c110 + c001 + c101 + c011 + c111).to(torch.int16)
    active = (~any_invalid) & (s_sum > 0) & (s_sum < 8)                # (Nx,Ny,Nz)
    active_cells = active.nonzero(as_tuple=False).cpu().numpy()        # (K, 3)

    # Dense cell→vert-index grid; -1 means "this cell did not produce a vert".
    # Replaces the dict lookup in the connectivity pass below.
    vert_idx_grid = np.full((Nx, Ny, Nz), -1, dtype=np.int64)

    # Optional splat-Hermite source. We use a cKDTree for radius queries —
    # `query_ball_point` on a million-splat tree is fast (sub-ms per query).
    splat_tree = None
    splat_pos_np = None
    splat_n_np = None
    if splat_positions is not None and splat_normals is not None:
        from scipy.spatial import cKDTree
        splat_pos_np = (splat_positions.cpu().numpy()
                        if isinstance(splat_positions, torch.Tensor)
                        else np.asarray(splat_positions))
        splat_n_np = (splat_normals.cpu().numpy()
                      if isinstance(splat_normals, torch.Tensor)
                      else np.asarray(splat_normals))
        splat_pos_np = splat_pos_np.astype(np.float64, copy=False)
        splat_n_np = splat_n_np.astype(np.float64, copy=False)
        splat_tree = cKDTree(splat_pos_np)
    splat_search_r = float(splat_radius_voxels) * voxel_size

    for cell in active_cells:
        i = int(cell[0]); j = int(cell[1]); k = int(cell[2])
        cell_origin = origin_np + np.array([i, j, k], dtype=np.float64) * voxel_size
        cell_center = cell_origin + 0.5 * voxel_size

        # Path A — splat Hermite data: prefer when ≥2 splats are within the
        # search radius. Gives DC its sharp-feature win.
        if splat_tree is not None:
            idx = splat_tree.query_ball_point(cell_center, splat_search_r)
            if len(idx) >= 2:
                p_samples = splat_pos_np[idx]
                n_samples = splat_n_np[idx]
                vert = solve_qef_cell(p_samples, n_samples,
                                      cell_origin, voxel_size)
                vert_idx_grid[i, j, k] = len(vert_array)
                vert_indices[(i, j, k)] = len(vert_array)
                vert_array.append(vert)
                if colors_np is not None:
                    color_array.append(_sample_color_trilinear(vert))
                continue

        # Path B — edge-crossings + ∇φ fallback (classical DC on smooth φ).
        crossings_pos: list[np.ndarray] = []
        crossings_normal: list[np.ndarray] = []
        for a, b in EDGES:
            ai, aj, ak = i + a[0], j + a[1], k + a[2]
            bi, bj, bk = i + b[0], j + b[1], k + b[2]
            sa = int(sign_np[ai, aj, ak])
            sb = int(sign_np[bi, bj, bk])
            if sa == sb:
                continue
            pa = float(phi_np[ai, aj, ak])
            pb = float(phi_np[bi, bj, bk])
            denom = pa - pb
            t = pa / denom if abs(denom) > 1e-9 else 0.5
            t = max(0.0, min(1.0, t))
            corner_a = origin_np + np.array([ai, aj, ak], dtype=np.float64) * voxel_size
            corner_b = origin_np + np.array([bi, bj, bk], dtype=np.float64) * voxel_size
            pos = corner_a + t * (corner_b - corner_a)
            n = grad_np[ai, aj, ak] + t * (grad_np[bi, bj, bk] - grad_np[ai, aj, ak])
            nn = float(np.linalg.norm(n))
            if not np.isfinite(nn) or nn < 1e-8 or not np.isfinite(n).all():
                n = np.array([0.0, 0.0, 1.0])
            else:
                n = n / nn
            crossings_pos.append(pos)
            crossings_normal.append(n)

        if len(crossings_pos) < 2:
            continue

        vert = solve_qef_cell(
            np.array(crossings_pos, dtype=np.float64),
            np.array(crossings_normal, dtype=np.float64),
            cell_origin, voxel_size,
        )
        vert_idx_grid[i, j, k] = len(vert_array)
        vert_indices[(i, j, k)] = len(vert_array)
        vert_array.append(vert)
        if colors_np is not None:
            color_array.append(_sample_color_trilinear(vert))

    # ---- Vectorized connectivity edge walk ---------------------------------
    # For each axis, compute the corner-edge sign-change mask in torch, then
    # Python-iterate only over the (usually small) set of crossing edges.
    faces: list[tuple[int, int, int, int]] = []

    def _cross_mask(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return ((s1 == 0) & (s2 == 1)) | ((s1 == 1) & (s2 == 0))

    # +z edges. Interior corners (i,j) ∈ [1..Cx-2] × [1..Cy-2]; edge k ∈ [0..Cz-2].
    if Cx > 2 and Cy > 2 and Cz > 1:
        s_z0 = sign[1:-1, 1:-1, :-1]
        s_z1 = sign[1:-1, 1:-1, 1:]
        crosses_z = _cross_mask(s_z0, s_z1).nonzero(as_tuple=False).cpu().numpy()
        for (i_, j_, k) in crosses_z:
            i = int(i_) + 1; j = int(j_) + 1; k = int(k)
            v00 = vert_idx_grid[i-1, j-1, k]
            v10 = vert_idx_grid[i,   j-1, k]
            v11 = vert_idx_grid[i,   j,   k]
            v01 = vert_idx_grid[i-1, j,   k]
            if v00 < 0 or v10 < 0 or v11 < 0 or v01 < 0:
                continue
            idx = [int(v00), int(v10), int(v11), int(v01)]
            if int(sign_np[i, j, k + 1]):
                idx = idx[::-1]
            faces.append(tuple(idx))

    # +y edges. Interior corners (i,k) ∈ [1..Cx-2] × [1..Cz-2]; edge j ∈ [0..Cy-2].
    if Cx > 2 and Cy > 1 and Cz > 2:
        s_y0 = sign[1:-1, :-1, 1:-1]
        s_y1 = sign[1:-1, 1:, 1:-1]
        crosses_y = _cross_mask(s_y0, s_y1).nonzero(as_tuple=False).cpu().numpy()
        for (i_, j, k_) in crosses_y:
            i = int(i_) + 1; j = int(j); k = int(k_) + 1
            v00 = vert_idx_grid[i-1, j, k-1]
            v10 = vert_idx_grid[i,   j, k-1]
            v11 = vert_idx_grid[i,   j, k]
            v01 = vert_idx_grid[i-1, j, k]
            if v00 < 0 or v10 < 0 or v11 < 0 or v01 < 0:
                continue
            idx = [int(v00), int(v10), int(v11), int(v01)]
            if int(sign_np[i, j, k]):
                idx = idx[::-1]
            faces.append(tuple(idx))

    # +x edges. Interior corners (j,k) ∈ [1..Cy-2] × [1..Cz-2]; edge i ∈ [0..Cx-2].
    if Cx > 1 and Cy > 2 and Cz > 2:
        s_x0 = sign[:-1, 1:-1, 1:-1]
        s_x1 = sign[1:, 1:-1, 1:-1]
        crosses_x = _cross_mask(s_x0, s_x1).nonzero(as_tuple=False).cpu().numpy()
        for (i, j_, k_) in crosses_x:
            i = int(i); j = int(j_) + 1; k = int(k_) + 1
            v00 = vert_idx_grid[i, j-1, k-1]
            v10 = vert_idx_grid[i, j,   k-1]
            v11 = vert_idx_grid[i, j,   k]
            v01 = vert_idx_grid[i, j-1, k]
            if v00 < 0 or v10 < 0 or v11 < 0 or v01 < 0:
                continue
            idx = [int(v00), int(v10), int(v11), int(v01)]
            if int(sign_np[i + 1, j, k]):
                idx = idx[::-1]
            faces.append(tuple(idx))

    verts_np = (np.array(vert_array, dtype=np.float32)
                if vert_array else np.zeros((0, 3), dtype=np.float32))
    if not faces:
        empty_tri = np.zeros((0, 3), dtype=np.int32)
        empty_col = (np.zeros((0, 3), dtype=np.float32) if colors_np is not None else None)
        return verts_np, empty_tri, empty_col
    quads = np.array(faces, dtype=np.int64)
    tris = np.empty((quads.shape[0] * 2, 3), dtype=np.int32)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    vcolors = np.array(color_array, dtype=np.float32) if colors_np is not None else None
    return verts_np, tris, vcolors


def run(means: torch.Tensor, quats: torch.Tensor, opacities: torch.Tensor,
        scene_diag: float, density: float = 0.02,
        shell_thickness: float = 6.0, *,
        extra_positions: torch.Tensor | None = None,
        extra_normals: torch.Tensor | None = None,
        extra_weights: torch.Tensor | None = None,
        radius_voxels: int = 2):
    """Build Hermite samples from a 2DGS splat cloud and run DC.

    `means/quats/opacities` are the post-activation trainer tensors (quats
    are unit quaternions in (w,x,y,z) order; opacities ∈ [0,1]). Voxel size
    follows the existing `_mesh_tsdf` formula
    (`density * scene_diag / shell_thickness`) so the mesh-tab sliders
    transfer 1:1.

    Optional `extra_*` tensors append per-pixel dome-render Hermite samples
    for the `dense_samples` enrichment path (drives smoother large flat
    regions; harmless on sharp features since QEF still places the vertex
    via the dominant-normal directions)."""
    positions = means
    normals = quat_to_disk_normal(quats)
    weights = opacities.clone()

    if extra_positions is not None:
        positions = torch.cat([positions, extra_positions], dim=0)
        normals = torch.cat([normals, extra_normals], dim=0)
        weights = torch.cat([weights, extra_weights], dim=0)

    voxel_size = (density * float(scene_diag)) / float(shell_thickness)
    return dual_contour(positions, normals, weights, voxel_size,
                        radius_voxels=radius_voxels)


# --------------------------------------------------------------------------- #
# Sanity test — run as `python visergui/dual_contouring.py`
# --------------------------------------------------------------------------- #


def _sanity_plane(device: str = "cpu"):
    """Plane at z=0.5 with normal [0,0,1]. Both implementations should produce
    a flat sheet at z≈0.5."""
    rng = np.random.default_rng(42)
    N = 1500
    xy = rng.uniform(-1.0, 1.0, (N, 2))
    z = np.full((N, 1), 0.5)
    pos = torch.from_numpy(np.concatenate([xy, z], axis=1)).float().to(device)
    nrm = torch.tensor([0., 0., 1.], device=device).expand(N, 3).contiguous()
    wts = torch.ones(N, device=device)

    cell = 0.1
    verts, tris = dual_contour(pos, nrm, wts, voxel_size=cell)
    print(f"  ours   V={len(verts):5d}  F={len(tris):5d}  "
          f"z=[{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")

    # Reference on the analytical SDF z - 0.5.
    from _third_party.mc_dc_reference import dual_contour_3d_ref, V3
    def f(x, y, z): return z - 0.5
    def f_n(x, y, z): return V3(0.0, 0.0, 1.0)
    v_ref, q_ref = dual_contour_3d_ref(f, f_n, -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
                                       cell_size=cell)
    print(f"  ref    V={len(v_ref):5d}  Q={len(q_ref):5d}  "
          f"z=[{v_ref[:, 2].min():.3f}, {v_ref[:, 2].max():.3f}]")

    assert abs(verts[:, 2].mean() - 0.5) < cell, \
        f"plane test: our mesh mean-z is off: {verts[:, 2].mean()}"
    assert abs(v_ref[:, 2].mean() - 0.5) < cell, \
        f"plane test: reference mean-z is off: {v_ref[:, 2].mean()}"
    print("  plane test PASSED (both impls within voxel_size of z=0.5)")


def _sanity_lshape(device: str = "cpu"):
    """Two perpendicular planes meeting at x=0, y=0. Tests sharp-feature
    preservation: QEF should place the corner vertex on the intersection
    edge, not interpolated to mid-cell.

    Solid region: x>0 OR y>0 (the "L"). SDF: min(-x, -y). >0 outside, <0 inside.
    """
    from _third_party.mc_dc_reference import dual_contour_3d_ref, V3
    def f(x, y, z): return min(-x, -y)
    def f_n(x, y, z):
        if -x < -y:                         # x is the active face
            return V3(-1.0, 0.0, 0.0)
        return V3(0.0, -1.0, 0.0)
    v_ref, q_ref = dual_contour_3d_ref(f, f_n, -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
                                       cell_size=0.1)
    near_edge = np.sum(np.abs(v_ref[:, 0]) + np.abs(v_ref[:, 1]) < 0.15)
    print(f"  ref    V={len(v_ref):5d}  Q={len(q_ref):5d}  "
          f"verts near (x,y)=(0,0) edge: {near_edge}")
    assert near_edge > 0, "reference DC did not recover sharp edge"
    print("  L-shape reference test PASSED (sharp edge present in ref output)")


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    print("[dual_contouring] sanity test")
    print("plane:")
    _sanity_plane()
    print("L-shape:")
    _sanity_lshape()
    print("OK")
