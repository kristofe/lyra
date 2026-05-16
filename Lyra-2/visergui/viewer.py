"""
Phase 2 — splat rendering with full GUI controls + point-cloud toggle.

Loads a 3DGS .ply (Phase 1 path), optionally additional point clouds via
`--points` (Phase 1.5), and exposes the inspection / rendering controls from
Phase 2: SH-degree pinning, RGB/Depth render modes, FOV, near/far for depth
normalization, render-resolution cap, FPS / render-ms readouts, camera
readout, and a reset-camera button. The "Display" toggle still switches
the whole scene between gsplat-rasterized splats and viser-native point
clouds.

Remote use:
  On GPU box:  python viewer.py path/to/scene.ply --port 8080
  On laptop:   ssh -L 8080:localhost:8080 user@gpubox
  Browser:     http://localhost:8080

   python visergui/viewer.py ../../world_models/Lyra_Experiments/scene.ply --port 8080
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import gsplat
import numpy as np
import torch
import viser
from plyfile import PlyData

# Phase 5: only the Protocol type — no concrete trainer is imported here.
# `training.py` itself is stdlib-only so this import stays cheap.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from training import TrainingControl  # noqa: E402


SH_C0 = 0.28209479177387814  # band-0 SH normalization (gsplat / Inria convention)


# --------------------------------------------------------------------------- #
# Camera math
# --------------------------------------------------------------------------- #


def _quat_wxyz_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def viser_camera_to_opencv_viewmat(
    position: np.ndarray | tuple[float, float, float],
    wxyz: np.ndarray | tuple[float, float, float, float],
) -> np.ndarray:
    """World-to-camera 4x4 matrix in OpenCV convention.

    Viser's CameraHandle.wxyz is already the c2w rotation in OpenCV camera-frame
    conventions (look=+Z, up=-Y, right=+X) per viser/_viser.py. No basis change
    is needed — invert to get world-to-camera.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    q = np.asarray(wxyz, dtype=np.float64).reshape(4)
    R_c2w = _quat_wxyz_to_rotmat(q)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R_w2c
    M[:3, 3] = t_w2c
    return M


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (..., 4) wxyz quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → (w, x, y, z) quaternion. Shepperd-style branch
    on the largest diagonal element to avoid sqrt of a negative."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


# --------------------------------------------------------------------------- #
# Color utilities for point clouds
# --------------------------------------------------------------------------- #


_TURBO_LUT: np.ndarray | None = None


def _turbo_lut() -> np.ndarray:
    """(256, 3) uint8 LUT for the 'turbo' colormap. Lazy and cached."""
    global _TURBO_LUT
    if _TURBO_LUT is None:
        from matplotlib import colormaps
        _TURBO_LUT = (
            colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0
        ).astype(np.uint8)
    return _TURBO_LUT


# --------------------------------------------------------------------------- #
# PointCloudLayer + loaders
# --------------------------------------------------------------------------- #


@dataclass
class PointCloudLayer:
    name: str
    points: np.ndarray             # (N, 3) float32, CPU
    colors_rgb: np.ndarray         # (N, 3) uint8
    metadata: dict = field(default_factory=dict)
    visible: bool = True
    point_size: float = 0.0005
    color_mode: str = "rgb"        # "rgb" | "axis" | "confidence" | "uniform"
    uniform_color: tuple[int, int, int] = (255, 51, 51)


def compute_colors(layer: PointCloudLayer) -> np.ndarray:
    mode = layer.color_mode
    if mode == "rgb":
        return layer.colors_rgb
    if mode == "axis":
        p = layer.points
        mn = p.min(axis=0)
        mx = p.max(axis=0)
        rng = np.maximum(mx - mn, 1e-6)
        return ((p - mn) / rng * 255.0).clip(0, 255).astype(np.uint8)
    if mode == "confidence":
        c = layer.metadata.get("confidence")
        if c is None:
            return layer.colors_rgb
        idx = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.int64)
        return _turbo_lut()[idx]
    if mode == "uniform":
        return np.tile(
            np.asarray(layer.uniform_color, dtype=np.uint8)[None, :],
            (len(layer.points), 1),
        )
    raise ValueError(f"unknown color_mode: {mode!r}")


class PointCloudLoader:
    def __init__(self, max_points: int = 1_000_000, seed: int = 42) -> None:
        self.max_points = max_points
        self.seed = seed

    def load(self, path: Path, name: str | None = None) -> PointCloudLayer:
        ext = path.suffix.lower()
        if ext == ".ply":
            points, colors, extras = self._load_ply(path)
        elif ext == ".npz":
            points, colors, extras = self._load_npz(path)
        elif ext == ".npy":
            points, colors, extras = self._load_npy(path)
        else:
            raise ValueError(f"unsupported point cloud format: {ext}")
        if self.max_points and len(points) > self.max_points:
            points, colors, extras = self._downsample(points, colors, extras)
        return PointCloudLayer(
            name=name or path.stem,
            points=points,
            colors_rgb=colors,
            metadata=extras,
        )

    def _downsample(self, points, colors, extras):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(points), size=self.max_points, replace=False)
        idx.sort()
        return (
            points[idx],
            colors[idx],
            {k: (v[idx] if hasattr(v, "__getitem__") else v) for k, v in extras.items()},
        )

    @staticmethod
    def _load_ply(path: Path):
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        points = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        names = set(v.dtype.names)
        if {"red", "green", "blue"}.issubset(names):
            r, g, b = v["red"], v["green"], v["blue"]
            if r.dtype.kind == "f":
                colors = (
                    np.stack([r, g, b], axis=-1).clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
            else:
                colors = np.stack([r, g, b], axis=-1).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 200, dtype=np.uint8)
        return points, colors, {}

    @staticmethod
    def _load_npz(path: Path):
        data = np.load(str(path))
        if "points" not in data.files:
            raise ValueError(f"{path}: .npz must contain a 'points' array")
        points = data["points"].astype(np.float32)
        if "colors" in data.files:
            c = data["colors"]
            colors = c.astype(np.uint8) if c.dtype != np.uint8 else c
        else:
            colors = np.full((len(points), 3), 200, dtype=np.uint8)
        extras: dict = {}
        if "confidence" in data.files:
            extras["confidence"] = data["confidence"].astype(np.float32)
        return points, colors, extras

    @staticmethod
    def _load_npy(path: Path):
        points = np.load(str(path)).astype(np.float32)
        colors_path = path.parent / (path.stem + ".colors.npy")
        if colors_path.exists():
            colors = np.load(str(colors_path)).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 200, dtype=np.uint8)
        return points, colors, {}


def derive_splat_centers_layer(
    scene: "SceneState", max_display_points: int = 200_000,
) -> PointCloudLayer | None:
    with scene.read() as s:
        if s.means is None or s.num_splats == 0:
            return None
        means_np = s.means.detach().cpu().numpy().astype(np.float32)
        dc_sh = s.sh[:, 0, :].detach().cpu().numpy()
        rgb01 = (SH_C0 * dc_sh + 0.5).clip(0.0, 1.0)
        colors_u8 = (rgb01 * 255.0).astype(np.uint8)
    # Cap displayed points: a 1M+ splat scene blows out the websocket on every
    # pump publish and tanks three.js Points performance. A uniform random
    # sample is plenty for orientation/setup work.
    if means_np.shape[0] > max_display_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(means_np.shape[0], size=max_display_points, replace=False)
        means_np = means_np[idx]
        colors_u8 = colors_u8[idx]
    # Default point size scaled to the scene's bbox diagonal so the layer is
    # visible at whatever world scale the splats live in. DA3 / COLMAP scenes
    # can be sub-unit or hundreds of units; a fixed 0.0005 vanishes in either.
    mn = means_np.min(axis=0)
    mx = means_np.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    size = max(diag * 5e-3, 1e-4)
    return PointCloudLayer(
        name="splat_centers",
        points=means_np,
        colors_rgb=colors_u8,
        point_size=size,
    )


# --------------------------------------------------------------------------- #
# Splat .ply loader
# --------------------------------------------------------------------------- #


@dataclass
class LoadedSplats:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh: torch.Tensor
    sh_degree: int


class PlyLoader:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def load(self, path: str | Path, flip_x: bool = True) -> LoadedSplats:
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        prop_names = v.dtype.names

        means_np = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        opacities_np = _sigmoid(v["opacity"].astype(np.float32))
        scales_np = np.exp(
            np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
        )
        quats_np = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        ).astype(np.float32)
        quats_np = quats_np / np.linalg.norm(quats_np, axis=-1, keepdims=True).clip(min=1e-12)

        dc_np = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
        ).astype(np.float32)

        rest_props = sorted(
            (n for n in prop_names if n.startswith("f_rest_")),
            key=lambda s: int(s.split("_")[-1]),
        )
        if rest_props:
            assert len(rest_props) % 3 == 0
            K_minus_1 = len(rest_props) // 3
            K = K_minus_1 + 1
            sh_degree = int(round(math.sqrt(K))) - 1
            assert (sh_degree + 1) ** 2 == K
            rest_np = np.stack([v[n] for n in rest_props], axis=-1).astype(np.float32)
            rest_np = rest_np.reshape(-1, 3, K_minus_1).transpose(0, 2, 1)
            sh_np = np.concatenate([dc_np[:, None, :], rest_np], axis=1)
        else:
            sh_degree = 0
            sh_np = dc_np[:, None, :]

        if flip_x:
            means_np[:, 1:] *= -1.0
            q_flip = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            quats_np = _quat_mul_wxyz(np.broadcast_to(q_flip, quats_np.shape), quats_np)

        device = self.device
        return LoadedSplats(
            means=torch.from_numpy(means_np).to(device),
            quats=torch.from_numpy(quats_np).to(device),
            scales=torch.from_numpy(scales_np).to(device),
            opacities=torch.from_numpy(opacities_np).to(device),
            sh=torch.from_numpy(sh_np).to(device),
            sh_degree=sh_degree,
        )


# --------------------------------------------------------------------------- #
# SceneState
# --------------------------------------------------------------------------- #


class SceneState:
    """Splat tensors + named point-cloud layers + training progress,
    all guarded by an RLock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.means: torch.Tensor | None = None
        self.quats: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.opacities: torch.Tensor | None = None
        self.sh: torch.Tensor | None = None
        self.sh_degree: int = 0
        self.num_splats: int = 0
        self.point_clouds: dict[str, PointCloudLayer] = {}
        # Phase 5: training progress. Trainer writes via record_step under
        # the same lock; the GUI pump reads them under read().
        self.step: int = 0
        self.loss_history: list[tuple[int, float]] = []
        # Bumped by anything that swaps the splat tensors; the GUI pump
        # watches this so live-training publishes trigger a re-render.
        self.splat_version: int = 0

    @contextmanager
    def read(self):
        with self._lock:
            yield self

    @contextmanager
    def write(self):
        with self._lock:
            yield self

    def load_from_ply(self, path, device, flip_x=True):
        loaded = PlyLoader(device).load(path, flip_x=flip_x)
        with self.write():
            self.means = loaded.means
            self.quats = loaded.quats
            self.scales = loaded.scales
            self.opacities = loaded.opacities
            self.sh = loaded.sh
            self.sh_degree = loaded.sh_degree
            self.num_splats = int(loaded.means.shape[0])

    def add_point_cloud(self, layer: PointCloudLayer) -> None:
        with self.write():
            self.point_clouds[layer.name] = layer

    def remove_point_cloud(self, name: str) -> None:
        with self.write():
            self.point_clouds.pop(name, None)

    def record_step(self, step: int, loss: float) -> None:
        """Trainer-side hook. Records progress under the write lock so the
        GUI pump's reader can never observe a partially-mutated history."""
        with self.write():
            self.step = int(step)
            self.loss_history.append((int(step), float(loss)))
            if len(self.loss_history) > 10_000:
                self.loss_history = self.loss_history[-10_000:]


# --------------------------------------------------------------------------- #
# Motion tracking (Phase 3)
# --------------------------------------------------------------------------- #


class MotionTracker:
    """Per-renderer (singleton) camera motion detector with idle-frame hysteresis.

    Thresholds are in world units (meters) for translation and radians for
    rotation. Quaternion sign ambiguity is handled with abs(dot).
    """

    def __init__(
        self,
        trans_thresh: float = 1e-3,
        rot_thresh: float = 1e-3,
        idle_frames: int = 5,
    ) -> None:
        self.last_pos: np.ndarray | None = None
        self.last_wxyz: np.ndarray | None = None
        self.idle_count: int = 0
        self.trans_thresh = trans_thresh
        self.rot_thresh = rot_thresh
        self.idle_frames_required = idle_frames

    def update(self, position: np.ndarray, wxyz: np.ndarray) -> bool:
        """Returns True if the camera is currently considered 'moving'."""
        if self.last_pos is None:
            self.last_pos = position.copy()
            self.last_wxyz = wxyz.copy()
            return False
        d_trans = float(np.linalg.norm(position - self.last_pos))
        dot = abs(float(np.dot(wxyz, self.last_wxyz)))
        d_rot = 2.0 * math.acos(min(1.0, dot))
        moving = d_trans > self.trans_thresh or d_rot > self.rot_thresh
        if moving:
            self.idle_count = 0
        else:
            self.idle_count += 1
        self.last_pos = position.copy()
        self.last_wxyz = wxyz.copy()
        return self.idle_count < self.idle_frames_required


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class Renderer:
    """Single splat-render entry point. Returns (img_uint8_HW3, render_ms).

    `color_mode='RGB'` runs gsplat with render_mode='RGB'.
    `color_mode='Depth'` runs render_mode='RGB+ED' and turbo-colorizes the
    expected-z-depth channel using a GPU LUT, normalized by [near, far].
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self._depth_lut: torch.Tensor | None = None
        self.motion = MotionTracker()

    def _get_depth_lut(self) -> torch.Tensor:
        if self._depth_lut is None:
            self._depth_lut = torch.from_numpy(_turbo_lut()).to(self.device)
        return self._depth_lut

    @torch.inference_mode()
    def render(
        self,
        scene: SceneState,
        camera: viser.CameraHandle,
        width: int,
        height: int,
        *,
        sh_degree: int,
        color_mode: str = "RGB",
        near: float = 0.1,
        far: float = 100.0,
        adaptive_res: bool = False,
        moving_scale: float = 1.0,
        force_full_res: bool = False,
    ) -> tuple[np.ndarray, float, bool]:
        # Update motion tracker on every render so its history stays
        # consistent. force_full_res still tracks motion but bypasses scaling.
        moving = self.motion.update(
            np.asarray(camera.position, dtype=np.float64),
            np.asarray(camera.wxyz, dtype=np.float64),
        )
        if adaptive_res and moving and not force_full_res:
            scale = max(0.05, min(1.0, float(moving_scale)))
            width = max(64, int(round(width * scale)))
            height = max(64, int(round(height * scale)))

        with scene.read() as s:
            if s.num_splats == 0 or s.means is None:
                return np.zeros((height, width, 3), dtype=np.uint8), 0.0, moving

            viewmat_np = viser_camera_to_opencv_viewmat(camera.position, camera.wxyz)
            viewmats = torch.from_numpy(viewmat_np.astype(np.float32))[None].to(self.device)

            fov = float(camera.fov)
            fy = 0.5 * height / math.tan(0.5 * fov)
            fx = fy
            cx, cy = 0.5 * width, 0.5 * height
            K = torch.tensor(
                [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
                dtype=torch.float32,
                device=self.device,
            )

            sh_degree_eff = max(0, min(int(sh_degree), int(s.sh_degree)))
            t0 = time.perf_counter()

            if color_mode == "Depth":
                out, _alpha, _info = gsplat.rasterization(
                    means=s.means,
                    quats=s.quats,
                    scales=s.scales,
                    opacities=s.opacities,
                    colors=s.sh,
                    viewmats=viewmats,
                    Ks=K,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_eff,
                    render_mode="RGB+ED",
                )
                # Last channel is expected depth.
                depth = out[0, :, :, 3]
                span = max(far - near, 1e-6)
                d_norm = ((depth - near) / span).clamp(0.0, 1.0)
                idx = (d_norm * 255.0).to(torch.long)
                img_t = self._get_depth_lut()[idx]  # (H, W, 3) uint8
            else:
                rgb, _alpha, _info = gsplat.rasterization(
                    means=s.means,
                    quats=s.quats,
                    scales=s.scales,
                    opacities=s.opacities,
                    colors=s.sh,
                    viewmats=viewmats,
                    Ks=K,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_eff,
                    render_mode="RGB",
                )
                img_t = rgb[0].clamp(0.0, 1.0).mul(255.0).to(torch.uint8)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            img_np = img_t.cpu().numpy()
            t1 = time.perf_counter()
            return img_np, (t1 - t0) * 1000.0, moving


# --------------------------------------------------------------------------- #
# ViewerApp
# --------------------------------------------------------------------------- #


_VISER_PC_PREFIX = "pc/"


def _viser_pc_name(layer_name: str) -> str:
    return _VISER_PC_PREFIX + layer_name


def _compute_home_pose(means_np: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return (position, look_at, up_direction) for a reset-camera home view."""
    mn = means_np.min(axis=0)
    mx = means_np.max(axis=0)
    center = ((mn + mx) * 0.5).astype(np.float64)
    extent = (mx - mn).astype(np.float64)
    diag = float(np.linalg.norm(extent))
    if diag <= 1e-9:
        diag = 1.0
    # Scene is -Y-up after the flip_x .ply transform (Inria convention with
    # COLMAP-Y-down source), so offset along -Y for elevation and along
    # +X/-Z for a 3/4-ish view.
    offset = np.array([0.6, -0.6, -0.6], dtype=np.float64)
    offset = offset / np.linalg.norm(offset) * diag * 1.2
    pos = center + offset
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(center[0]), float(center[1]), float(center[2])),
        (0.0, -1.0, 0.0),
    )


class ViewerApp:
    """Owns the viser server, the GUI controls, and per-client render dispatch."""

    def __init__(
        self,
        ply_path: Path | None,
        host: str,
        port: int,
        flip_x: bool = True,
        extra_point_paths: Iterable[Path] = (),
        max_points: int = 1_000_000,
        derive_splat_points: bool = True,
        scene: SceneState | None = None,
        training_control: TrainingControl | None = None,
        initializer: "Callable[[dict], None] | None" = None,
        resetter: "Callable[[], None] | None" = None,
        default_init_args: dict | None = None,
        on_scale_mult_change: "Callable[[float], None] | None" = None,
    ) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # Allow the caller to construct SceneState in advance — the trainer
        # needs a reference *before* the .ply finishes loading so it can
        # close over scene.write() inside its step function.
        self.scene = scene if scene is not None else SceneState()
        self.training_control = training_control
        self.renderer = Renderer(device=device_str)
        self._initializer = initializer
        self._resetter = resetter
        self._default_init_args = default_init_args or {}
        self._on_scale_mult_change = on_scale_mult_change

        if ply_path is not None:
            print(f"loading splat .ply {ply_path} on {device_str}...")
            self.scene.load_from_ply(ply_path, device=device, flip_x=flip_x)
        if self.scene.means is None and initializer is None:
            raise RuntimeError(
                "ViewerApp: scene has no splats; pass ply_path, pre-populate "
                "the scene, or provide an `initializer` to set it up at runtime"
            )
        if self.scene.means is not None:
            print(
                f"  {self.scene.num_splats} splats, sh_degree={self.scene.sh_degree}, "
                f"flip_x={flip_x}"
            )

        # Home pose: compute from splat bbox if we have one; otherwise default
        # to a generic origin-looking pose. The init callback re-computes it
        # from the freshly-populated scene.
        if self.scene.means is not None:
            means_np = self.scene.means.detach().cpu().numpy()
            self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
            print(f"  home pose: pos={self.home_position} look_at={self.home_look_at}")
        else:
            self.home_position = (0.0, 0.0, 5.0)
            self.home_look_at = (0.0, 0.0, 0.0)
            self.home_up = (0.0, 1.0, 0.0)

        if derive_splat_points and self.scene.means is not None:
            derived = derive_splat_centers_layer(self.scene)
            if derived is not None:
                self.scene.add_point_cloud(derived)
                print(f"  derived layer 'splat_centers': {len(derived.points)} pts")

        loader = PointCloudLoader(max_points=max_points)
        for p in extra_point_paths:
            print(f"loading point cloud {p}...")
            layer = loader.load(p)
            self.scene.add_point_cloud(layer)
            print(f"  layer '{layer.name}': {len(layer.points)} pts")

        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(dark_mode=True)
        self._handles: dict[str, dict[str, object]] = {}
        self._pushed: set[str] = set()

        # ---- Perf throttling state --------------------------------------- #
        self._fps_ema: float = 0.0
        self._render_ms_ema: float = 0.0
        self._last_frame_t: float = 0.0
        self._last_readout_push: float = 0.0
        self._last_cam_push: float = 0.0
        self._last_motion_push: float = 0.0
        # Per-client settle timers fire a full-res render shortly after motion stops.
        self._settle_timers: dict[int, threading.Timer] = {}
        self._settle_delay_s: float = 0.15

        # Phase 5 pump state.
        self._pump_stop_event: threading.Event = threading.Event()
        self._pump_thread: threading.Thread | None = None

        # Training-camera frustums: cached so toggles can re-publish without
        # re-passing the data, and so we can clear/rebuild on each new init.
        self._train_cam_payload: dict | None = None
        self._train_cam_names: list[str] = []

        self._build_gui()
        # The pump watches splat_version (live training publishes) AND drives
        # the training-control readouts. Start it whenever either reason exists.
        if self.training_control is not None or self._initializer is not None:
            self._start_training_pump()
        self.server.on_client_connect(self._on_client_connect)

    # ---- GUI construction -------------------------------------------- #

    def _build_gui(self) -> None:
        # If training is attached or runtime-init is wired, split the right
        # panel into two tabs so the inspection controls and the training
        # controls don't fight for vertical space. Static viewer keeps the
        # original flat layout.
        if self.training_control is not None or self._initializer is not None:
            tabs = self.server.gui.add_tab_group()
            with tabs.add_tab("Inspect"):
                self._build_inspect_panel()
            with tabs.add_tab("Training"):
                if self._initializer is not None:
                    self._build_init_panel()
                self._build_training_gui(with_folder=False)
        else:
            self._build_inspect_panel()

    def _build_init_panel(self) -> None:
        """Runtime init/reset controls. Only built when `initializer` was passed."""
        d = self._default_init_args
        with self.server.gui.add_folder("Setup"):
            self.gui_init_video = self.server.gui.add_text(
                "video", initial_value=str(d.get("video", "")),
            )
            self.gui_init_max_frames = self.server.gui.add_number(
                "max_frames",
                initial_value=int(d.get("max_frames", 32)),
                min=1, max=1000, step=1,
            )
            self.gui_init_conf_q = self.server.gui.add_slider(
                "confidence_quantile",
                min=0.0, max=1.0, step=0.01,
                initial_value=float(d.get("confidence_quantile", 0.6)),
                hint="Quantile threshold for both point pruning and the loss mask.",
            )
            self.gui_init_remove_sky = self.server.gui.add_checkbox(
                "remove_sky",
                initial_value=bool(d.get("remove_sky", True)),
            )
            self.gui_init_sh_max_deg = self.server.gui.add_number(
                "sh_max_deg",
                initial_value=int(d.get("sh_max_deg", 0)),
                min=0, max=3, step=1,
                hint="0 = L1-only (v1). >0 enables SH-band progression 0→N.",
            )
            self.gui_init_lpips_weight = self.server.gui.add_slider(
                "lpips_weight",
                min=0.0, max=1.0, step=0.01,
                initial_value=float(d.get("lpips_weight", 0.0)),
                hint="LPIPS perceptual loss weight added to L1 (0 disables).",
            )
            self.gui_init_scale_mult = self.server.gui.add_slider(
                "max_scale_voxels",
                min=0.5, max=10.0, step=0.1,
                initial_value=float(d.get("scale_clamp_voxel_mult", 2.0)),
                hint="Cap on each gaussian's per-axis scale, as a multiple of "
                     "the init voxel edge. Live — applies on the next training step.",
            )
            self.gui_init_densify = self.server.gui.add_checkbox(
                "densify",
                initial_value=bool(d.get("use_densify", False)),
                hint="Enable gsplat clone/split/prune (v3). Splat count "
                     "evolves during training.",
            )
            self.gui_init_btn = self.server.gui.add_button("Initialize")
            self.gui_init_reset_btn = self.server.gui.add_button("Reset")
            self.gui_init_status = self.server.gui.add_markdown(
                "**init:** not initialized"
            )
        self.gui_init_btn.on_click(lambda _ev: self._on_init_click())
        self.gui_init_reset_btn.on_click(lambda _ev: self._on_reset_click())
        if self._on_scale_mult_change is not None:
            self.gui_init_scale_mult.on_update(
                lambda _ev: self._on_scale_mult_change(
                    float(self.gui_init_scale_mult.value)
                )
            )

    def _build_inspect_panel(self) -> None:
        # Top-level: Display toggle (splats vs point clouds).
        self.gui_display = self.server.gui.add_dropdown(
            "Display",
            options=("splats", "points"),
            initial_value="splats",
            hint="Switch the whole scene between splat rasterization and point clouds.",
        )
        self.gui_display.on_update(lambda _ev: self._apply_display_mode())

        # Always offer 0..3 — the renderer clamps via min(gui, scene.sh_degree),
        # so picking 3 on a degree-0 scene just renders at 0 (no harm), and
        # live-training mode can ramp sh_degree up without rebuilding the dropdown.
        sh_options = tuple(str(d) for d in range(4))
        with self.server.gui.add_folder("Render"):
            self.gui_sh_degree = self.server.gui.add_dropdown(
                "sh_degree",
                options=sh_options,
                initial_value=str(min(3, max(0, self.scene.sh_degree))),
                hint="SH degree used by the rasterizer. Clamped to the scene's available bands.",
            )
            self.gui_render_mode = self.server.gui.add_dropdown(
                "render_mode", options=("RGB", "Depth"), initial_value="RGB"
            )
            self.gui_fov = self.server.gui.add_slider(
                "fov_deg", min=30.0, max=110.0, step=1.0, initial_value=60.0,
                hint="Vertical field of view in degrees. Pushed to viser's per-client camera.",
            )
            self.gui_near = self.server.gui.add_slider(
                "depth_near", min=0.001, max=10.0, step=0.001, initial_value=0.1,
                hint="Near plane for depth normalization.",
            )
            self.gui_far = self.server.gui.add_slider(
                "depth_far", min=1.0, max=1000.0, step=1.0, initial_value=100.0,
                hint="Far plane for depth normalization.",
            )

        with self.server.gui.add_folder("Performance"):
            self.gui_max_res = self.server.gui.add_slider(
                "max_res", min=256, max=2048, step=16, initial_value=1080,
                hint="Caps the rendered image's longest edge.",
            )
            self.gui_fps_readout = self.server.gui.add_markdown("**fps:** —")
            self.gui_render_ms_readout = self.server.gui.add_markdown("**render ms:** —")
            self.gui_adaptive_res = self.server.gui.add_checkbox(
                "adaptive_res", initial_value=True,
                hint="Render at lower resolution while the camera is moving.",
            )
            self.gui_moving_scale = self.server.gui.add_slider(
                "moving_scale", min=0.25, max=1.0, step=0.05, initial_value=0.5,
                hint="Resolution scale during motion (1.0 disables the drop).",
            )
            self.gui_idle_frames = self.server.gui.add_slider(
                "idle_frames", min=1, max=30, step=1, initial_value=5,
                hint="Consecutive still frames before declaring 'idle'.",
            )
            self.gui_trans_thresh = self.server.gui.add_number(
                "trans_thresh", initial_value=1e-3,
                min=1e-5, max=1.0, step=1e-4,
                hint="World-units (m) translation delta to count as motion.",
            )
            self.gui_rot_thresh = self.server.gui.add_number(
                "rot_thresh", initial_value=1e-3,
                min=1e-5, max=1.0, step=1e-4,
                hint="Radians rotation delta to count as motion.",
            )
            self.gui_motion_state_readout = self.server.gui.add_markdown(
                "**motion:** idle"
            )

        with self.server.gui.add_folder("Scene"):
            self.gui_splat_count_readout = self.server.gui.add_markdown(
                f"**splat_count:** {int(self.scene.num_splats):,}"
            )
            self.gui_camera_pos_readout = self.server.gui.add_markdown(
                "**camera pos:** —"
            )
            self.gui_camera_look_readout = self.server.gui.add_markdown(
                "**look dir:** —"
            )
            self.gui_reset_camera = self.server.gui.add_button("reset camera")

        with self.server.gui.add_folder("Training Cameras"):
            self.gui_show_train_cams = self.server.gui.add_checkbox(
                "show", initial_value=True,
                hint="Render a frustum for every training-frame camera.",
            )
            self.gui_train_cam_scale = self.server.gui.add_slider(
                "scale", min=0.01, max=2.0, step=0.01, initial_value=0.15,
                hint="Frustum size in world units.",
            )
            self.gui_train_cam_images = self.server.gui.add_checkbox(
                "show images", initial_value=True,
                hint="Display each training frame inside its frustum.",
            )
            self.gui_train_cam_count = self.server.gui.add_markdown(
                "**cameras:** 0"
            )
        self.gui_show_train_cams.on_update(lambda _ev: self._republish_train_cams())
        self.gui_train_cam_scale.on_update(lambda _ev: self._republish_train_cams())
        self.gui_train_cam_images.on_update(lambda _ev: self._republish_train_cams())

        # Debug: push a hard-coded magenta point cloud to test whether viser's
        # point rendering works at all in this scene/coord-frame.
        self.gui_debug_test_pc = self.server.gui.add_button("debug: test point cloud")
        self.gui_debug_test_pc.on_click(lambda _ev: self._push_debug_test_pc())

        # Save the folder handle so lazy layer-GUI builds (from the training
        # pump) can nest under it instead of landing at the top level.
        self._point_clouds_folder = self.server.gui.add_folder("Point Clouds")
        with self._point_clouds_folder:
            self.global_size_mult_handle = self.server.gui.add_slider(
                "global_size_mult", min=0.01, max=100.0, step=0.01, initial_value=1.0
            )
            self.global_size_mult_handle.on_update(
                lambda _ev: self._push_all_visible_layers()
            )
            for name in list(self.scene.point_clouds.keys()):
                self._build_layer_gui(name)

        # Wire control callbacks.
        self.gui_render_mode.on_update(lambda _ev: self._render_all_clients())
        self.gui_sh_degree.on_update(lambda _ev: self._render_all_clients())
        self.gui_max_res.on_update(lambda _ev: self._render_all_clients())
        self.gui_near.on_update(lambda _ev: self._maybe_render_depth())
        self.gui_far.on_update(lambda _ev: self._maybe_render_depth())
        self.gui_fov.on_update(lambda _ev: self._on_fov_change())
        self.gui_reset_camera.on_click(lambda _ev: self._on_reset_camera())
        self.gui_idle_frames.on_update(lambda _ev: self._on_idle_frames_change())
        self.gui_trans_thresh.on_update(
            lambda _ev: setattr(
                self.renderer.motion, "trans_thresh", float(self.gui_trans_thresh.value)
            )
        )
        self.gui_rot_thresh.on_update(
            lambda _ev: setattr(
                self.renderer.motion, "rot_thresh", float(self.gui_rot_thresh.value)
            )
        )

    # ---- Display mode (splats vs points) ----------------------------- #

    @property
    def display_mode(self) -> str:
        return str(self.gui_display.value)

    def _apply_display_mode(self) -> None:
        if self.display_mode == "splats":
            for name in list(self.scene.point_clouds.keys()):
                self._remove_pushed(name)
            for client in self.server.get_clients().values():
                self._render_for(client)
        else:
            for client in self.server.get_clients().values():
                client.scene.set_background_image(None)
            self._push_all_visible_layers()

    # ---- Per-layer GUI ----------------------------------------------- #

    def _build_layer_gui(self, name: str) -> None:
        layer = self.scene.point_clouds[name]
        # Range slider relative to the layer's derived size — for DA3 scenes
        # this lands in a useful neighborhood instead of capping at 0.1.
        base = max(float(layer.point_size), 1e-4)
        size_min = max(base * 0.01, 1e-5)
        size_max = base * 100.0
        size_step = max(base * 0.01, 1e-5)
        with self.server.gui.add_folder(name):
            visible = self.server.gui.add_checkbox("visible", initial_value=layer.visible)
            size = self.server.gui.add_slider(
                "point_size", min=size_min, max=size_max, step=size_step,
                initial_value=base,
            )
            color_mode = self.server.gui.add_dropdown(
                "color_mode", options=("rgb", "axis", "confidence", "uniform"),
                initial_value=layer.color_mode,
            )
            uniform_color = self.server.gui.add_rgb(
                "uniform_color", initial_value=layer.uniform_color
            )
            count = self.server.gui.add_markdown(
                f"**count:** {len(layer.points):,}"
            )

        handles = {
            "visible": visible,
            "size": size,
            "color_mode": color_mode,
            "uniform_color": uniform_color,
            "count": count,
        }
        self._handles[name] = handles
        for h in (visible, size, color_mode, uniform_color):
            h.on_update(lambda _ev, n=name: self._push_layer(n))

    def _push_all_visible_layers(self) -> None:
        for name in list(self.scene.point_clouds.keys()):
            self._push_layer(name)

    def _push_debug_test_pc(self) -> None:
        """Hard-coded sanity-check: try a few variants to isolate why
        add_point_cloud isn't rendering. Also drops a known-good `add_box`
        and `add_icosphere` at the same spot as a sentinel — those use
        completely different three.js paths."""
        if self.scene.means is not None and int(self.scene.num_splats) > 0:
            mn = self.scene.means.detach().cpu().numpy().min(axis=0)
            mx = self.scene.means.detach().cpu().numpy().max(axis=0)
            center = ((mn + mx) * 0.5).astype(np.float32)
        else:
            center = np.zeros(3, dtype=np.float32)

        # 1) Sentinel: a 1-unit magenta box at the center. Uses add_box.
        try:
            self.server.scene.add_box(
                name="debug_box",
                dimensions=(1.0, 1.0, 1.0),
                color=(255, 0, 255),
                position=tuple(center.tolist()),
            )
            print(f"[debug] add_box OK at {center.tolist()}")
        except Exception as e:
            print(f"[debug] add_box raised: {e}")

        # 2) Default precision (float16) + uint8 colors — original path.
        rng = np.random.default_rng(0)
        pts = (rng.standard_normal((1000, 3)).astype(np.float32) * 0.5 + center)
        cols = np.tile(np.array([255, 0, 255], dtype=np.uint8), (1000, 1))
        try:
            self.server.scene.add_point_cloud(
                name="debug_pc_default",
                points=pts, colors=cols, point_size=0.3,
            )
            print("[debug] add_point_cloud (default) OK")
        except Exception as e:
            print(f"[debug] add_point_cloud (default) raised: {e}")

        # 3) Same data but precision='float32' explicitly.
        try:
            self.server.scene.add_point_cloud(
                name="debug_pc_f32",
                points=pts, colors=cols, point_size=0.3,
                precision="float32",
            )
            print("[debug] add_point_cloud (precision=float32) OK")
        except Exception as e:
            print(f"[debug] add_point_cloud (precision=float32) raised: {e}")

        # 4) Tiny: just 3 points way bigger to see if size is the only issue.
        big_pts = np.array(
            [center, center + 1.0, center + 2.0], dtype=np.float32,
        )
        big_cols = np.array(
            [[255, 0, 255], [0, 255, 255], [255, 255, 0]], dtype=np.uint8,
        )
        try:
            self.server.scene.add_point_cloud(
                name="debug_pc_huge",
                points=big_pts, colors=big_cols, point_size=2.0,
            )
            print("[debug] add_point_cloud (3 huge points) OK")
        except Exception as e:
            print(f"[debug] add_point_cloud (3 huge points) raised: {e}")

    # ---- Training-camera frustums ------------------------------------ #

    def publish_training_cameras(
        self,
        c2w: np.ndarray,
        K: np.ndarray,
        images: np.ndarray | None,
        H: int,
        W: int,
    ) -> None:
        """Cache the per-frame camera payload and push frustums into the
        scene. Pass an empty c2w (shape (0, ...)) to clear."""
        c2w_np = np.asarray(c2w, dtype=np.float64)
        if c2w_np.shape[0] == 0:
            self._train_cam_payload = None
            self._republish_train_cams()
            return
        K_np = np.asarray(K, dtype=np.float64)
        imgs_np: np.ndarray | None
        if images is None:
            imgs_np = None
        else:
            imgs = np.asarray(images)
            if imgs.dtype != np.uint8:
                imgs_np = np.clip(imgs * 255.0, 0.0, 255.0).astype(np.uint8) \
                    if imgs.dtype.kind == "f" else imgs.astype(np.uint8)
            else:
                imgs_np = imgs
        self._train_cam_payload = {
            "c2w": c2w_np, "K": K_np, "images": imgs_np,
            "H": int(H), "W": int(W),
        }
        self._republish_train_cams()

    def _republish_train_cams(self) -> None:
        for n in self._train_cam_names:
            try:
                self.server.scene.remove_by_name(n)
            except Exception:
                pass
        self._train_cam_names.clear()

        payload = self._train_cam_payload
        try:
            self.gui_train_cam_count.content = (
                f"**cameras:** {int(payload['c2w'].shape[0]) if payload else 0}"
            )
        except Exception:
            pass

        if payload is None or not bool(self.gui_show_train_cams.value):
            return

        c2w = payload["c2w"]
        K = payload["K"]
        imgs = payload["images"]
        H, W = payload["H"], payload["W"]
        scale = float(self.gui_train_cam_scale.value)
        with_images = bool(self.gui_train_cam_images.value) and imgs is not None
        aspect = W / max(H, 1)
        for i in range(c2w.shape[0]):
            R = c2w[i, :3, :3]
            t = c2w[i, :3, 3]
            wxyz = _rotmat_to_wxyz(R)
            fy = float(K[i, 1, 1])
            fov_y = 2.0 * math.atan(0.5 * H / max(fy, 1e-9))
            name = f"train_cams/{i:04d}"
            self.server.scene.add_camera_frustum(
                name=name,
                fov=fov_y,
                aspect=aspect,
                scale=scale,
                color=(255, 153, 51),
                wxyz=tuple(float(x) for x in wxyz),
                position=tuple(float(x) for x in t),
                image=imgs[i] if with_images else None,
            )
            self._train_cam_names.append(name)

    def _refresh_splat_centers_layer(self) -> None:
        """Re-derive the 'splat_centers' point cloud from the live splats.
        Called from the training pump on every splat_version bump so the
        layer reflects the current optimizer state instead of going stale."""
        derived = derive_splat_centers_layer(self.scene)
        if derived is None:
            print("[refresh splat_centers] derive returned None")
            return
        print(
            f"[refresh splat_centers] N={len(derived.points)} "
            f"layer.point_size={derived.point_size:.5f}"
        )
        self.scene.add_point_cloud(derived)
        # First-time appearance: build the per-layer GUI controls under the
        # existing "Point Clouds" folder context. After that we just push.
        if "splat_centers" not in self._handles:
            try:
                with self._point_clouds_folder:
                    self._build_layer_gui("splat_centers")
                print("[refresh splat_centers] built layer GUI")
            except Exception as e:
                print(f"[refresh splat_centers] GUI build failed: {e}")
                return
        if self.display_mode == "points":
            self._push_layer("splat_centers")

    def _push_layer(self, name: str) -> None:
        layer = self.scene.point_clouds.get(name)
        handles = self._handles.get(name)
        if layer is None or handles is None:
            print(f"[push_layer {name}] skip: layer={layer is not None} handles={handles is not None}")
            self._remove_pushed(name)
            return
        if self.display_mode != "points" or not bool(handles["visible"].value):
            print(f"[push_layer {name}] skip: mode={self.display_mode} visible={bool(handles['visible'].value)}")
            self._remove_pushed(name)
            return
        resolved = replace(
            layer,
            visible=True,
            point_size=float(handles["size"].value),
            color_mode=str(handles["color_mode"].value),
            uniform_color=tuple(int(c) for c in handles["uniform_color"].value),
        )
        colors = compute_colors(resolved)
        size = resolved.point_size * float(self.global_size_mult_handle.value)
        viser_name = _viser_pc_name(name)
        try:
            bbox_min = resolved.points.min(axis=0)
            bbox_max = resolved.points.max(axis=0)
            print(
                f"[push_layer {name}] N={len(resolved.points)} "
                f"size={size:.5f} (layer={resolved.point_size:.5f} mult={float(self.global_size_mult_handle.value):.3f}) "
                f"color_mode={resolved.color_mode} "
                f"bbox_min={bbox_min} bbox_max={bbox_max} "
                f"colors[0]={colors[0].tolist()}"
            )
        except Exception as _e:
            print(f"[push_layer {name}] diag print failed: {_e}")
        self.server.scene.add_point_cloud(
            name=viser_name, points=resolved.points, colors=colors, point_size=size,
        )
        self._pushed.add(viser_name)
        print(f"[push_layer {name}] add_point_cloud OK as '{viser_name}'")

    def _remove_pushed(self, name: str) -> None:
        viser_name = _viser_pc_name(name)
        if viser_name in self._pushed:
            self.server.scene.remove_by_name(viser_name)
            self._pushed.discard(viser_name)

    # ---- Render-control callbacks ------------------------------------ #

    def _render_all_clients(self) -> None:
        for client in self.server.get_clients().values():
            self._render_for(client)

    def _maybe_render_depth(self) -> None:
        if str(self.gui_render_mode.value) == "Depth":
            self._render_all_clients()

    def _on_fov_change(self) -> None:
        rad = math.radians(float(self.gui_fov.value))
        for client in self.server.get_clients().values():
            try:
                client.camera.fov = rad
            except Exception:
                pass

    def _on_reset_camera(self) -> None:
        for client in self.server.get_clients().values():
            client.camera.position = self.home_position
            client.camera.look_at = self.home_look_at
            client.camera.up_direction = self.home_up

    # ---- Client lifecycle / rendering -------------------------------- #

    def _on_client_connect(self, client: viser.ClientHandle) -> None:
        client.camera.on_update(lambda _cam: self._render_for(client))
        # Apply current FOV to this client.
        try:
            client.camera.fov = math.radians(float(self.gui_fov.value))
        except Exception:
            pass
        # Seed the orbit-up axis and home pose so a fresh page load yaws
        # around the right axis without needing the user to click reset.
        try:
            client.camera.up_direction = self.home_up
            client.camera.position = self.home_position
            client.camera.look_at = self.home_look_at
        except Exception:
            pass
        if self.display_mode == "splats":
            self._render_for(client)
        else:
            client.scene.set_background_image(None)

    def _render_for(
        self, client: viser.ClientHandle, *, force_full_res: bool = False
    ) -> None:
        if self.display_mode != "splats":
            return
        cam = client.camera
        try:
            W = int(cam.image_width)
            H = int(cam.image_height)
        except (TypeError, ValueError):
            return
        if W <= 0 or H <= 0:
            return

        # Snapshot all GUI-driven state.
        sh_degree = int(self.gui_sh_degree.value)
        color_mode = str(self.gui_render_mode.value)
        near = float(self.gui_near.value)
        far = float(self.gui_far.value)
        max_res = int(self.gui_max_res.value)
        adaptive_res = bool(self.gui_adaptive_res.value)
        moving_scale = float(self.gui_moving_scale.value)

        max_dim = max(W, H)
        if max_dim > max_res:
            scale = max_res / max_dim
            W = max(1, int(round(W * scale)))
            H = max(1, int(round(H * scale)))

        img, render_ms, moving = self.renderer.render(
            self.scene, cam, W, H,
            sh_degree=sh_degree,
            color_mode=color_mode,
            near=near,
            far=far,
            adaptive_res=adaptive_res,
            moving_scale=moving_scale,
            force_full_res=force_full_res,
        )
        client.scene.set_background_image(img)
        self._update_perf_readouts(render_ms)
        self._update_camera_readout(cam)
        self._update_motion_readout(moving)

        # If we just rendered low-res due to motion, schedule a delayed full-res
        # follow-up so the still frame the user lands on is crisp. No camera
        # update will fire while they hold still, so we cannot rely on the
        # idle-counter alone — it only ticks when renders fire.
        if adaptive_res and moving and not force_full_res:
            self._schedule_settle(client)
        else:
            self._cancel_settle(client)

    # ---- Throttled GUI readouts -------------------------------------- #

    def _update_perf_readouts(self, render_ms: float) -> None:
        now = time.perf_counter()
        # EMA of render-ms regardless of throttle (cheap, informative).
        if self._render_ms_ema == 0.0:
            self._render_ms_ema = render_ms
        else:
            self._render_ms_ema = 0.9 * self._render_ms_ema + 0.1 * render_ms
        if self._last_frame_t > 0.0:
            dt = now - self._last_frame_t
            if dt > 1e-9:
                inst = 1.0 / dt
                self._fps_ema = (
                    inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
                )
        self._last_frame_t = now

        if now - self._last_readout_push >= 0.1:  # 10 Hz
            self.gui_fps_readout.content = f"**fps:** {self._fps_ema:.1f}"
            self.gui_render_ms_readout.content = (
                f"**render ms:** {self._render_ms_ema:.2f}"
            )
            self._last_readout_push = now

    def _update_camera_readout(self, cam: viser.CameraHandle) -> None:
        now = time.perf_counter()
        if now - self._last_cam_push < 0.1:  # 10 Hz
            return
        R = _quat_wxyz_to_rotmat(np.asarray(cam.wxyz))
        look = R[:, 2]  # camera +Z = forward in OpenCV = viser convention
        pos = np.asarray(cam.position)
        self.gui_camera_pos_readout.content = (
            f"**camera pos:** {float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f}"
        )
        self.gui_camera_look_readout.content = (
            f"**look dir:** {float(look[0]):.3f}, {float(look[1]):.3f}, {float(look[2]):.3f}"
        )
        self._last_cam_push = now

    def _update_motion_readout(self, moving: bool) -> None:
        now = time.perf_counter()
        if now - self._last_motion_push < 0.1:  # 10 Hz
            return
        self.gui_motion_state_readout.content = (
            f"**motion:** {'moving' if moving else 'idle'}"
        )
        self._last_motion_push = now

    # ---- Adaptive-resolution helpers --------------------------------- #

    def _on_idle_frames_change(self) -> None:
        self.renderer.motion.idle_frames_required = int(self.gui_idle_frames.value)

    def _schedule_settle(self, client: viser.ClientHandle) -> None:
        cid = int(client.client_id)
        old = self._settle_timers.pop(cid, None)
        if old is not None:
            old.cancel()
        timer = threading.Timer(
            self._settle_delay_s, lambda: self._settle_render(cid)
        )
        timer.daemon = True
        self._settle_timers[cid] = timer
        timer.start()

    def _cancel_settle(self, client: viser.ClientHandle) -> None:
        cid = int(client.client_id)
        old = self._settle_timers.pop(cid, None)
        if old is not None:
            old.cancel()

    def _settle_render(self, client_id: int) -> None:
        # Resolve the live ClientHandle; the user may have disconnected.
        client = self.server.get_clients().get(client_id)
        if client is None:
            return
        self._settle_timers.pop(client_id, None)
        self._render_for(client, force_full_res=True)

    # ---- Training (Phase 5) ----------------------------------------- #

    def _build_training_gui(self, *, with_folder: bool = True) -> None:
        # When called from inside a "Training" tab the outer folder would
        # be redundant, so the caller can suppress it.
        from contextlib import nullcontext
        ctx = self.server.gui.add_folder("Training") if with_folder else nullcontext()
        with ctx:
            self.gui_train_resume = self.server.gui.add_button("resume")
            self.gui_train_pause = self.server.gui.add_button("pause")
            self.gui_train_status = self.server.gui.add_markdown("**status:** stopped")
            self.gui_train_step = self.server.gui.add_markdown("**step:** 0")
            self.gui_train_loss = self.server.gui.add_markdown("**loss:** —")
            self.gui_train_splat_count = self.server.gui.add_markdown(
                f"**splats:** {int(self.scene.num_splats):,}"
            )
            self.gui_train_loss_plot = self.server.gui.add_plotly(
                self._make_loss_figure([]), aspect=2.0
            )

        self.gui_train_resume.on_click(lambda _ev: self._on_resume_training())
        self.gui_train_pause.on_click(lambda _ev: self._on_pause_training())

    def _on_resume_training(self) -> None:
        ctl = self.training_control
        if ctl is None:
            return
        ctl.start()   # idempotent on BackgroundTrainingThread
        ctl.resume()

    def _on_pause_training(self) -> None:
        ctl = self.training_control
        if ctl is None:
            return
        ctl.pause()

    def _on_init_click(self) -> None:
        """Run the user-supplied initializer with the current Init-panel
        values. Updates status + recomputes the home pose so 'reset camera'
        targets the new bbox."""
        if self._initializer is None:
            return
        try:
            self.gui_init_btn.disabled = True
            self.gui_init_reset_btn.disabled = True
            self.gui_init_status.content = "**init:** preprocessing video + DA3…"
            opts = dict(
                video=Path(self.gui_init_video.value),
                max_frames=int(self.gui_init_max_frames.value),
                confidence_quantile=float(self.gui_init_conf_q.value),
                remove_sky=bool(self.gui_init_remove_sky.value),
                sh_max_deg=int(self.gui_init_sh_max_deg.value),
                lpips_weight=float(self.gui_init_lpips_weight.value),
                use_densify=bool(self.gui_init_densify.value),
            )
            # viser runs on_click in a thread pool, so blocking here is fine.
            self._initializer(opts)
            # Recompute home pose from the newly-populated scene so the
            # camera-reset button lands somewhere sensible. Push the new
            # up/position/look_at to every connected client — otherwise the
            # orbit controls keep the empty-boot up axis and splats appear
            # upside down until the page is refreshed.
            if self.scene.means is not None:
                means_np = self.scene.means.detach().cpu().numpy()
                self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
                for client in self.server.get_clients().values():
                    try:
                        client.camera.up_direction = self.home_up
                        client.camera.position = self.home_position
                        client.camera.look_at = self.home_look_at
                    except Exception:
                        pass
            self.gui_init_status.content = (
                f"**init:** ready ({self.scene.num_splats:,} splats) — click resume"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_init_status.content = f"**init:** error — {type(e).__name__}: {e}"
        finally:
            self.gui_init_btn.disabled = False
            self.gui_init_reset_btn.disabled = False

    def _on_reset_click(self) -> None:
        """Pause training (caller's responsibility actually pauses; we also
        try via training_control) and ask the user's resetter to tear down."""
        try:
            self.gui_init_reset_btn.disabled = True
            if self.training_control is not None:
                try:
                    self.training_control.pause()
                except Exception:
                    pass
            if self._resetter is not None:
                self._resetter()
            self.gui_init_status.content = "**init:** reset — re-initialize to train"
        except Exception as e:
            self.gui_init_status.content = f"**init:** reset error — {type(e).__name__}: {e}"
        finally:
            self.gui_init_reset_btn.disabled = False

    @staticmethod
    def _make_loss_figure(history: list[tuple[int, float]]):
        """Build a small plotly figure from a (step, loss) list, downsampled
        stride-uniformly to ≤1000 points so websocket pushes stay cheap."""
        import plotly.graph_objects as go
        h = history
        if len(h) > 1000:
            stride = len(h) // 1000
            h = h[::stride]
        xs = [p[0] for p in h]
        ys = [p[1] for p in h]
        fig = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="lines", name="loss")])
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=220,
            xaxis_title="step",
            yaxis_title="loss",
        )
        return fig

    def _start_training_pump(self) -> None:
        self._pump_stop_event.clear()
        t = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread = t
        t.start()

    def _pump_loop(self) -> None:
        last_plot_t = 0.0
        last_splat_version = -1
        ctl = self.training_control
        while not self._pump_stop_event.is_set():
            time.sleep(0.1)  # 10 Hz readouts
            if ctl is None:
                continue
            with self.scene.read() as s:
                step = s.step
                splats = s.num_splats
                history_snapshot = list(s.loss_history)
                splat_version = s.splat_version
            if splat_version != last_splat_version:
                last_splat_version = splat_version
                try:
                    self._render_all_clients()
                except Exception:
                    pass
                try:
                    self._refresh_splat_centers_layer()
                except Exception:
                    pass
            latest_loss = history_snapshot[-1][1] if history_snapshot else 0.0
            try:
                self.gui_train_status.content = f"**status:** {ctl.status()}"
                self.gui_train_step.content = f"**step:** {int(step):,}"
                self.gui_train_loss.content = f"**loss:** {float(latest_loss):.6f}"
                self.gui_train_splat_count.content = f"**splats:** {int(splats):,}"
                self.gui_splat_count_readout.content = (
                    f"**splat_count:** {int(splats):,}"
                )
            except Exception:
                # Server may have shut down; tolerate races on exit.
                continue
            now = time.perf_counter()
            if now - last_plot_t >= 0.5:  # 2 Hz plot
                try:
                    self.gui_train_loss_plot.figure = self._make_loss_figure(history_snapshot)
                except Exception:
                    pass
                last_plot_t = now

    # ---- Run ---------------------------------------------------------- #

    def run(self) -> None:
        host = self.server.get_host()
        port = self.server.get_port()
        print(f"viser server listening on http://{host}:{port}")
        try:
            self.server.sleep_forever()
        finally:
            # Best-effort shutdown; either order is fine since they don't depend on each other.
            self._pump_stop_event.set()
            if self._pump_thread is not None:
                self._pump_thread.join(timeout=2.0)
            if self.training_control is not None:
                try:
                    self.training_control.stop()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 splat viewer with full GUI controls + point-cloud toggle."
    )
    parser.add_argument("ply_path", type=Path)
    parser.add_argument(
        "--points", type=Path, action="append", default=[],
        help="Additional point cloud (.ply / .npy / .npz). Repeatable.",
    )
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--no-derive-points", action="store_true")
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if not args.ply_path.exists():
        raise SystemExit(f"ply not found: {args.ply_path}")
    for p in args.points:
        if not p.exists():
            raise SystemExit(f"point cloud not found: {p}")

    ViewerApp(
        ply_path=args.ply_path,
        host=args.host,
        port=args.port,
        flip_x=not args.no_flip,
        extra_point_paths=args.points,
        max_points=args.max_points,
        derive_splat_points=not args.no_derive_points,
    ).run()


if __name__ == "__main__":
    main()
