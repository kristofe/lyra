"""Multi-view SAM 2 mask prep for the InstaInpaint backend.

Given:
- A captured view (RGB + per-pixel depth + camera) with a 2D mask in it,
- A set of nearby training views (RGB + cameras),

we produce one SAM 2 mask per neighbor view that selects the same 3D object.

Algorithm (Phase 2 of the InstaInpaint plan):
  1. Compute the centroid of the captured-view mask. Read depth at that
     pixel; back-project to a 3D world point P.
  2. For each neighbor view, project P through the neighbor's intrinsics +
     extrinsics → pixel (u, v).
  3. Run SAM 2 with that pixel as a point prompt on the neighbor RGB →
     binary mask.
  4. Return the per-neighbor masks alongside the captured mask (unchanged).

Cameras are OpenCV-convention c2w (x right, y down, z forward), same as
Lyra-2's `viser_camera_to_opencv_viewmat`.

The SAM 2 predictor is lazy-loaded on first call and cached on the module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


_VENDOR_INSTA = Path(__file__).resolve().parent.parent / "vendor" / "InstaInpaint"
_DEFAULT_SAM_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_DEFAULT_SAM_CKPT = _VENDOR_INSTA / "checkpoints" / "sam2.1_hiera_large.pt"

_PREDICTOR = None


def _get_predictor(device: str = "cuda"):
    global _PREDICTOR
    if _PREDICTOR is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not _DEFAULT_SAM_CKPT.is_file():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found at {_DEFAULT_SAM_CKPT}. "
                f"Download via: "
                f"wget -O {_DEFAULT_SAM_CKPT} 'https://dl.fbaipublicfiles.com/"
                f"segment_anything_2/092824/sam2.1_hiera_large.pt'"
            )
        sam = build_sam2(_DEFAULT_SAM_CFG, str(_DEFAULT_SAM_CKPT), device=device)
        _PREDICTOR = SAM2ImagePredictor(sam)
    return _PREDICTOR


_VIDEO_PREDICTOR = None


def _get_video_predictor(device: str = "cuda"):
    """Lazy-load + cache the SAM 2 *video* predictor (temporal propagation).

    Mirrors `_get_predictor` but builds the video model, used to propagate a
    single object's mask across an ordered sequence of frames (the training
    frames). Reuses the same config + checkpoint as the image predictor.
    """
    global _VIDEO_PREDICTOR
    if _VIDEO_PREDICTOR is None:
        from sam2.build_sam import build_sam2_video_predictor

        if not _DEFAULT_SAM_CKPT.is_file():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found at {_DEFAULT_SAM_CKPT}. "
                f"Download via: "
                f"wget -O {_DEFAULT_SAM_CKPT} 'https://dl.fbaipublicfiles.com/"
                f"segment_anything_2/092824/sam2.1_hiera_large.pt'"
            )
        _VIDEO_PREDICTOR = build_sam2_video_predictor(
            _DEFAULT_SAM_CFG, str(_DEFAULT_SAM_CKPT), device=device
        )
    return _VIDEO_PREDICTOR


def _mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (u, v) pixel centroid of the non-zero region, or None if empty."""
    if mask.dtype != np.bool_:
        m = mask > 0
    else:
        m = mask
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    return int(round(xs.mean())), int(round(ys.mean()))


def _backproject_to_world(
    u: int, v: int, depth: float,
    K: np.ndarray, c2w_cv: np.ndarray,
) -> np.ndarray:
    """Pixel (u, v) + depth → 3D world point under OpenCV convention."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    return (c2w_cv @ p_cam)[:3]


def _project_world_to_pixel(
    p_world: np.ndarray, K: np.ndarray, c2w_cv: np.ndarray,
) -> tuple[float, float, float]:
    """World point → (u, v, z_camera) in OpenCV convention.

    z_camera > 0 means in front of the camera.
    """
    w2c = np.linalg.inv(c2w_cv)
    p_cam = (w2c @ np.append(p_world, 1.0).astype(np.float32))[:3]
    z = float(p_cam[2])
    if z <= 1e-6:
        return float("nan"), float("nan"), z
    u = float(K[0, 0] * p_cam[0] / z + K[0, 2])
    v = float(K[1, 1] * p_cam[1] / z + K[1, 2])
    return u, v, z


def prepare_multiview_masks(
    captured_rgb: np.ndarray,
    captured_mask: np.ndarray,
    captured_depth: np.ndarray,
    captured_K: np.ndarray,
    captured_c2w: np.ndarray,
    neighbor_rgbs: list[np.ndarray],
    neighbor_Ks: list[np.ndarray],
    neighbor_c2ws: list[np.ndarray],
    device: str = "cuda",
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int] | None]]:
    """Back-project the captured-view mask into each neighbor frame via SAM 2.

    Args:
        captured_rgb: (H, W, 3) uint8.
        captured_mask: (H, W) bool/uint8 — non-zero = inpaint region.
        captured_depth: (H, W) float — per-pixel depth in captured view (world units).
        captured_K: (3, 3) — OpenCV intrinsics for the captured view.
        captured_c2w: (4, 4) — OpenCV c2w for the captured view.
        neighbor_rgbs: list of (H_n, W_n, 3) uint8 — one RGB per neighbor view.
        neighbor_Ks: list of (3, 3) — OpenCV intrinsics per neighbor.
        neighbor_c2ws: list of (4, 4) — OpenCV c2w per neighbor.
        device: cuda by default.

    Returns:
        (captured_mask, per_neighbor_masks, per_neighbor_prompt_pixels). Each
        neighbor mask is (H_n, W_n) bool (or None if the 3D point fell behind
        the neighbor's camera). prompt_pixels[i] is the (u, v) used for SAM 2;
        None if the projection failed.
    """
    centroid = _mask_centroid(captured_mask)
    if centroid is None:
        raise ValueError("captured_mask is empty — nothing to back-project.")
    u_c, v_c = centroid
    if not (0 <= v_c < captured_depth.shape[0] and 0 <= u_c < captured_depth.shape[1]):
        raise ValueError(f"Centroid ({u_c}, {v_c}) out of depth bounds.")
    d_c = float(captured_depth[v_c, u_c])
    if not np.isfinite(d_c) or d_c <= 0:
        # Fall back to median of finite positive depths inside the mask region.
        m = (captured_mask > 0) if captured_mask.dtype != np.bool_ else captured_mask
        finite_pos = captured_depth[m & np.isfinite(captured_depth) & (captured_depth > 0)]
        if finite_pos.size == 0:
            raise ValueError("No valid depth values inside the mask region.")
        d_c = float(np.median(finite_pos))

    p_world = _backproject_to_world(
        u_c, v_c, d_c,
        np.asarray(captured_K, dtype=np.float32),
        np.asarray(captured_c2w, dtype=np.float32),
    )

    predictor = _get_predictor(device=device)

    neighbor_masks: list[np.ndarray | None] = []
    prompt_pixels: list[tuple[int, int] | None] = []
    for rgb_n, K_n, c2w_n in zip(neighbor_rgbs, neighbor_Ks, neighbor_c2ws):
        u_n, v_n, z_n = _project_world_to_pixel(
            p_world, np.asarray(K_n, dtype=np.float32),
            np.asarray(c2w_n, dtype=np.float32),
        )
        H_n, W_n = rgb_n.shape[:2]
        if not (np.isfinite(u_n) and np.isfinite(v_n) and 0 <= u_n < W_n and 0 <= v_n < H_n):
            neighbor_masks.append(None)
            prompt_pixels.append(None)
            continue
        predictor.set_image(rgb_n)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[u_n, v_n]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        neighbor_masks.append(masks[best].astype(bool))
        prompt_pixels.append((int(round(u_n)), int(round(v_n))))

    return captured_mask.astype(bool), neighbor_masks, prompt_pixels
