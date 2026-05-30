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
from gsplat.utils import depth_to_normal
import numpy as np
import torch
import viser
from plyfile import PlyData

# Phase 5: only the Protocol type — no concrete trainer is imported here.
# `training.py` itself is stdlib-only so this import stays cheap.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from training import TrainingControl  # noqa: E402
from splat_trainer import format_loss_components  # noqa: E402


SH_C0 = 0.28209479177387814  # band-0 SH normalization (gsplat / Inria convention)


# --------------------------------------------------------------------------- #
# Log buffer (stdout/stderr → GUI mirror)
# --------------------------------------------------------------------------- #


class LogBuffer:
    """Tee sys.stdout/sys.stderr to the real terminal *and* remember the
    most recent line, drained by the pump loop into a single-line GUI readout.
    """

    def __init__(self) -> None:
        self._last_line: str = ""
        self._lock = threading.Lock()
        self._dirty = True
        self._orig_stdout = _sys.stdout
        self._orig_stderr = _sys.stderr
        self._partial = ""  # carry-over for writes that don't end in '\n'

    def install(self) -> None:
        _sys.stdout = self  # type: ignore[assignment]
        _sys.stderr = self  # type: ignore[assignment]

    def uninstall(self) -> None:
        _sys.stdout = self._orig_stdout
        _sys.stderr = self._orig_stderr

    def write(self, text: str) -> int:
        try:
            self._orig_stdout.write(text)
        except Exception:
            pass
        if not text:
            return 0
        with self._lock:
            buf = self._partial + text
            last_nl = buf.rfind("\n")
            if last_nl == -1:
                self._partial = buf
                return len(text)
            self._partial = buf[last_nl + 1 :]
            # Newest non-empty complete line wins.
            for line in buf[:last_nl].splitlines():
                if line.strip():
                    self._last_line = line
            self._dirty = True
        return len(text)

    def flush(self) -> None:
        try:
            self._orig_stdout.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    def drain_text(self) -> str | None:
        """Returns the most recent line if dirty since last drain, else None."""
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
            return self._last_line


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
        # Latest per-term loss breakdown published by the trainer
        # (e.g. {"l1": 0.005, "distortion": 4.2, ...}). Empty when the
        # trainer hasn't produced any component info yet.
        self.last_loss_components: dict[str, float] = {}
        # Parallel history of (step, components_dict) — same length cap as
        # `loss_history`. Used to draw per-term traces on the loss plot.
        self.loss_components_history: list[tuple[int, dict[str, float]]] = []
        # Bumped by anything that swaps the splat tensors; the GUI pump
        # watches this so live-training publishes trigger a re-render.
        self.splat_version: int = 0
        # Mesh tensors (populated by mesher.generate_mesh)
        self.mesh_verts: np.ndarray | None = None     # (V, 3) float32
        self.mesh_faces: np.ndarray | None = None     # (F, 3) int32
        self.mesh_normals: np.ndarray | None = None   # (V, 3) float32
        self.mesh_colors: np.ndarray | None = None    # (V, 3) float32 RGB or None

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

    def record_step(self, step: int, loss: float,
                    components: dict[str, float] | None = None) -> None:
        """Trainer-side hook. Records progress under the write lock so the
        GUI pump's reader can never observe a partially-mutated history.
        `components` is the latest per-term loss breakdown (see
        `splat_trainer.format_loss_components`); pass None or {} when no
        breakdown is available."""
        with self.write():
            self.step = int(step)
            self.loss_history.append((int(step), float(loss)))
            if len(self.loss_history) > 10_000:
                self.loss_history = self.loss_history[-10_000:]
            comps = dict(components) if components else {}
            self.last_loss_components = comps
            self.loss_components_history.append((int(step), comps))
            if len(self.loss_components_history) > 10_000:
                self.loss_components_history = self.loss_components_history[-10_000:]


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
    `color_mode='Normals'` reuses the RGB+ED rasterization and converts the
    expected depth to world-space surface normals via depth_to_normal, then
    encodes them as (n+1)/2 -> RGB with empty pixels set to black.
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

            if color_mode in ("Depth", "Normals"):
                out, alpha, _info = gsplat.rasterization(
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
                if color_mode == "Depth":
                    span = max(far - near, 1e-6)
                    d_norm = ((depth - near) / span).clamp(0.0, 1.0)
                    idx = (d_norm * 255.0).to(torch.long)
                    img_t = self._get_depth_lut()[idx]  # (H, W, 3) uint8
                else:
                    # Camera-space normals: identity camtoworlds leaves the
                    # unprojected points (and thus the normals) in the OpenCV
                    # camera frame. Negate so surfaces facing the camera have
                    # n_z = +1 (standard normal-map convention -> light blue).
                    eye = torch.eye(4, dtype=torch.float32, device=self.device)[None]
                    normals = -depth_to_normal(
                        depth[None, ..., None], eye, K, z_depth=True
                    )[0]
                    rgb_n = ((normals + 1.0) * 127.5).clamp(0.0, 255.0)
                    bg = alpha[0, ..., 0] < 0.01
                    rgb_n[bg] = 0.0
                    img_t = rgb_n.to(torch.uint8)
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
        trainer: "object | None" = None,
        save_checkpoint: "Callable[[str], dict] | None" = None,
        load_checkpoint: "Callable[[str], dict] | None" = None,
        append_video: "Callable[[str, int, bool], dict] | None" = None,
        append_frames: "Callable[[str, bool], dict] | None" = None,
        set_sampling: "Callable[[str, float, int], dict] | None" = None,
        set_freeze_mode: "Callable[[str], dict] | None" = None,
        recompute_freeze_mask: "Callable[[], dict] | None" = None,
        on_seed_dedup_mult_change: "Callable[[float], None] | None" = None,
        compute_voxel_overlap: "Callable[[float], dict] | None" = None,
        compute_coverage: "Callable[[str], dict] | None" = None,
        inpaint_preload: bool = True,
        request_video: "Callable[[bytes, str, str, str, dict], object] | None" = None,
        demo_defaults: dict | None = None,
    ) -> None:
        # Install the stdout/stderr mirror before anything in __init__ prints,
        # so the GUI log captures the full boot sequence.
        self._log_buf = LogBuffer()
        self._log_buf.install()

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
        self._save_checkpoint_cb = save_checkpoint
        self._load_checkpoint_cb = load_checkpoint
        self._append_video_cb = append_video
        self._append_frames_cb = append_frames
        self._set_sampling_cb = set_sampling
        self._set_freeze_mode_cb = set_freeze_mode
        self._recompute_freeze_mask_cb = recompute_freeze_mask
        self._on_seed_dedup_mult_change = on_seed_dedup_mult_change
        self._compute_voxel_overlap_cb = compute_voxel_overlap
        self._voxel_overlap_layer_name = "voxel_overlap"
        self._compute_coverage_cb = compute_coverage
        self._coverage_layer_name = "coverage_heatmap"
        self._inpaint_preload = bool(inpaint_preload)
        # Demo tab: synchronous video-generation callback + boot defaults.
        self._request_video_cb = request_video
        self._demo_defaults = demo_defaults or {}
        self._demo_image: "tuple[str, bytes] | None" = None
        # Last frame (PNG bytes) of the most recently generated clip; used to
        # continue the camera move on the next Request when the toggle is on.
        self._demo_last_frame: "tuple[str, bytes] | None" = None
        self._demo_count = 0
        # The first fetched clip — the one that SEEDED the scene. Re-initialize
        # re-runs init on this video so the original cameras come back. Using
        # the most-recently-appended clip instead (the old bug) re-preprocessed
        # only that clip and dropped the first video's cameras. append_video
        # nulls the trainer's _last_video, so this re-init re-preprocesses the
        # seed video cleanly (incremental appends are discarded — the designed
        # clean re-init, see SplatTrainer.append_video / reset docstrings).
        self._demo_init_video: "str | None" = None
        # Serializes the long, structure-mutating trainer handlers (init,
        # reset, append, prune, checkpoint, demo request/reset). viser runs
        # GUI callbacks on a thread pool and `disabled` is only a UI hint, so
        # without this lock a Reset from either tab could free the param
        # tensors while an append/init is mid-flight on another thread.
        self._trainer_op_lock = threading.Lock()

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

        # Mesh generation state (Phase 4+)
        self._mesh_thread: threading.Thread | None = None
        self._mesh_handle = None
        self._mesh_wire_handle = None
        self._trainer_ref = trainer

        self._build_gui()
        # The pump watches splat_version (live training publishes) AND drives
        # the training-control readouts. Start it whenever either reason exists.
        if self.training_control is not None or self._initializer is not None:
            self._start_training_pump()
        self.server.on_client_connect(self._on_client_connect)

    # ---- GUI construction -------------------------------------------- #

    def _build_gui(self) -> None:
        # Pinned at the very top of the panel — outside any tab — so the
        # latest stdout/stderr line is visible regardless of which tab is
        # active. Updated by `_pump_loop` from the LogBuffer.
        self.gui_log = self.server.gui.add_markdown("`(waiting for output...)`")
        # If training is attached or runtime-init is wired, split the right
        # panel into two tabs so the inspection controls and the training
        # controls don't fight for vertical space. Static viewer keeps the
        # original flat layout.
        if self.training_control is not None or self._initializer is not None:
            tabs = self.server.gui.add_tab_group()
            with tabs.add_tab("Render"):
                self._build_inspect_panel()
            with tabs.add_tab("Train"):
                if self._initializer is not None:
                    self._build_init_panel()
                self._build_training_gui(with_folder=False)
            # Demo tab is built AFTER Train so the gui_init_* handles already
            # exist and can be two-way linked to the Demo duplicates. It also
            # drives the initializer for the first video, so it needs both.
            if self._request_video_cb is not None and self._initializer is not None:
                with tabs.add_tab("Demo"):
                    self._build_demo_panel()
            with tabs.add_tab("Mesh"):
                self._build_mesh_panel()
            with tabs.add_tab("Inpaint"):
                try:
                    from inpainter import InpainterPanel
                    self.inpainter = InpainterPanel(
                        server=self.server,
                        trainer_ref=self._trainer_ref,
                        viewer=self,
                        preload=self._inpaint_preload,
                    )
                except Exception as e:
                    import traceback
                    print(f"  inpainter: panel init skipped: {e}", file=__import__("sys").stderr)
                    traceback.print_exc()
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
            self.gui_init_void_weight = self.server.gui.add_slider(
                "void_weight",
                min=0.0, max=2.0, step=0.05,
                initial_value=float(d.get("void_weight", 0.5)),
                hint="Penalty on rendered alpha inside ~train_mask. Drives "
                     "splats to render nothing in filtered regions and "
                     "removes asymmetric growth at mask boundaries (0 disables).",
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
            self.gui_init_mode = self.server.gui.add_dropdown(
                "mode",
                options=("3dgs", "2dgs"),
                initial_value=str(d.get("mode", "3dgs")),
                hint="3dgs = gsplat.rasterization. 2dgs = rasterization_2dgs + "
                     "distortion / normal-consistency / DA3 depth+normal "
                     "supervision (notebook recipe). Applies on next Initialize.",
            )
            self.gui_init_btn = self.server.gui.add_button("Initialize")
            self.gui_init_reset_btn = self.server.gui.add_button("Reset")
            self.gui_init_prune_btn = self.server.gui.add_button("Prune Splats")
            self.gui_init_status = self.server.gui.add_markdown(
                "**init:** not initialized"
            )
        self.gui_init_btn.on_click(lambda _ev: self._on_init_click())
        self.gui_init_reset_btn.on_click(lambda _ev: self._on_reset_click())
        self.gui_init_prune_btn.on_click(lambda _ev: self._on_prune_click())
        if self._on_scale_mult_change is not None:
            self.gui_init_scale_mult.on_update(
                lambda _ev: self._on_scale_mult_change(
                    float(self.gui_init_scale_mult.value)
                )
            )

        self._build_incremental_panel()
        # Mesh + Inpaint panels are now their own tabs (see _build_panel).

    def _build_incremental_panel(self) -> None:
        """Phase-1 incremental-training controls: save/load checkpoint and
        append additional frames from either another video or a directory
        of (frame, camera-json) pairs. All actions pause training first.
        Hidden when the host didn't wire up incremental callbacks (trainer
        re-use without train_and_view.py)."""
        if (self._save_checkpoint_cb is None
                and self._load_checkpoint_cb is None
                and self._append_video_cb is None
                and self._append_frames_cb is None):
            return
        with self.server.gui.add_folder("Incremental", expand_by_default=False):
            self.gui_inc_ckpt_path = self.server.gui.add_text(
                "checkpoint path",
                initial_value="checkpoints/splats_ckpt.ply",
                hint="PLY file written by Save / read by Load. A sidecar "
                     "JSON (same stem, .json) carries step count + voxel.",
            )
            self.gui_inc_save_btn = self.server.gui.add_button(
                "Save checkpoint",
                hint="Pause training, save PLY + sidecar JSON.",
            )
            self.gui_inc_load_btn = self.server.gui.add_button(
                "Load checkpoint",
                hint="Pause training, replace current splats with the PLY. "
                     "Requires Initialize to have run first so data + masks "
                     "are loaded for the source video.",
            )
            self.gui_inc_seed_splats = self.server.gui.add_checkbox(
                "seed new splats from added frames",
                initial_value=True,
                hint="Phase 2: after appending, RGBD-unproject the new "
                     "frames and add splats wherever the existing splat "
                     "distribution doesn't already cover (voxel-deduped "
                     "against existing splats at init.voxel). Off = "
                     "Phase 1 behavior: append only, splat count "
                     "unchanged.",
            )
            self.gui_inc_dedup_multiplier = self.server.gui.add_slider(
                "seed dedup radius (× init voxel)",
                min=1.0, max=6.0, step=0.5,
                initial_value=2.0,
                hint="Dedup-check radius for seed_new_splats, in init-voxel "
                     "units. 1.0 = exact voxel match; 2.0 (default) = a "
                     "candidate is occupied if any existing splat is within "
                     "~2 init voxels (handles DA3 pose noise so new clips "
                     "don't pile splats onto existing geometry). Raise "
                     "(3-4) if duplicates still appear; lower if dedup is "
                     "rejecting genuinely-new regions.",
            )
            self.gui_inc_video_path = self.server.gui.add_text(
                "append video",
                initial_value="",
                hint="MP4 of a second clip of the SAME scene from new "
                     "camera positions. Runs DA3 + appends each frame.",
            )
            self.gui_inc_video_max = self.server.gui.add_number(
                "append max_frames",
                initial_value=32, min=1, max=1000, step=1,
                hint="Frame-stride cap for the appended clip (same meaning "
                     "as Setup→max_frames).",
            )
            self.gui_inc_video_btn = self.server.gui.add_button(
                "Append from video",
            )
            self.gui_inc_frames_dir = self.server.gui.add_text(
                "append frames dir",
                initial_value="",
                hint="Directory of frame_NNNN.png + cam_NNNN.json files. "
                     "Each JSON has {K:3x3, c2w:4x4, depth?:relative .npy path}.",
            )
            self.gui_inc_frames_btn = self.server.gui.add_button(
                "Append from supplied frames",
            )
            self.gui_inc_status = self.server.gui.add_markdown(
                "**incremental:** (no action yet)"
            )

            # Phase 3: epoch-aware frame sampling.
            self.gui_inc_sampling_mode = self.server.gui.add_dropdown(
                "sampling mode",
                options=("uniform", "stratified", "scheduled"),
                initial_value="uniform",
                hint="Per-step frame sampler. Uniform = original behavior "
                     "(every frame equally likely). Stratified = sample "
                     "epoch first using new-frame-weight, then a frame in "
                     "that epoch. Scheduled = bias toward newest epoch "
                     "early, decay to stratified by 'horizon' steps.",
            )
            self.gui_inc_new_frame_weight = self.server.gui.add_slider(
                "new-frame weight",
                min=0.1, max=10.0, step=0.1,
                initial_value=1.0,
                hint="Relative weight of the *latest* epoch vs every other "
                     "epoch. 1.0 = uniform within stratified (no bias). "
                     "2.0 = latest epoch sampled 2x as often per frame as "
                     "each other epoch. Only matters in stratified / "
                     "scheduled modes.",
            )
            self.gui_inc_sampling_horizon = self.server.gui.add_number(
                "schedule horizon (steps)",
                initial_value=1000, min=10, max=100000, step=10,
                hint="Scheduled mode only: number of steps over which the "
                     "sampler interpolates from 'newest epoch only' (step 0) "
                     "to the stratified target (step ≥ horizon).",
            )
            self.gui_inc_sampling_status = self.server.gui.add_markdown(
                "**sampling:** uniform (epoch counts populate after init)"
            )

            # Phase 4: spatial freezing of irrelevant splats.
            self.gui_inc_freeze_mode = self.server.gui.add_dropdown(
                "freeze mode",
                options=("off", "new_frustums"),
                initial_value="off",
                hint="Off (default) = train every splat. new_frustums = "
                     "after clicking Recompute, freeze splats invisible to "
                     "any camera in the LATEST epoch — old-region splats "
                     "stop moving so new-region refinement won't drag them. "
                     "Densify is auto-disabled while a mask is active so "
                     "splat-count can't desync the mask.",
            )
            self.gui_inc_freeze_btn = self.server.gui.add_button(
                "Recompute freeze mask",
                hint="Projects every splat into the latest-epoch cameras. "
                     "Splats visible (in front of camera + inside the image "
                     "rectangle of at least one) stay trainable; others "
                     "freeze. Runs in a fraction of a second even for 3M "
                     "splats. Click after each Append or after switching modes.",
            )
            self.gui_inc_freeze_status = self.server.gui.add_markdown(
                "**freeze:** off (train all splats)"
            )

            # Phase 5: visual debug tools.
            with self.server.gui.add_folder("Debug viz", expand_by_default=False):
                self.gui_inc_voxel_overlap_show = self.server.gui.add_checkbox(
                    "show voxel overlap",
                    initial_value=False,
                    hint="Phase 5.2: classify every splat into its (voxel × "
                         "epoch) bin and render a colored point cloud at "
                         "occupied voxel centers. Red = old-only voxels, "
                         "green = new-only, yellow = shared. Quick read on "
                         "where two clips overlap and where they cover new "
                         "ground.",
                )
                self.gui_inc_voxel_overlap_mult = self.server.gui.add_slider(
                    "voxel overlap multiplier",
                    min=0.5, max=10.0, step=0.5,
                    initial_value=1.0,
                    hint="Voxel-overlap viz cell size, as a multiple of "
                         "init.voxel. Larger = coarser bins (faster, less "
                         "noisy). Smaller = finer bins (more detail, more "
                         "voxels).",
                )
                self.gui_inc_voxel_overlap_btn = self.server.gui.add_button(
                    "Recompute voxel overlap",
                    hint="Re-bin splats into voxels and re-add the colored "
                         "point cloud. Auto-recomputes on toggle ON.",
                )
                self.gui_inc_voxel_overlap_status = self.server.gui.add_markdown(
                    "**voxel overlap:** (off)"
                )

                self.gui_inc_coverage_show = self.server.gui.add_checkbox(
                    "show coverage heatmap",
                    initial_value=False,
                    hint="Phase 5.3: project every splat into the camera "
                         "set below and color by how many see it. Hot red "
                         "= many; cool blue = few or none. Splats at "
                         "count 0 are obvious prune candidates. Auto-"
                         "recomputes on toggle ON.",
                )
                self.gui_inc_coverage_scope = self.server.gui.add_dropdown(
                    "coverage camera scope",
                    options=("all", "latest"),
                    initial_value="latest",
                    hint="'all' = every camera in self.data; 'latest' = "
                         "only cameras with the largest epoch. Use "
                         "'latest' to see what the most recent append "
                         "supervises; 'all' for the full-scene picture.",
                )
                self.gui_inc_coverage_btn = self.server.gui.add_button(
                    "Recompute coverage heatmap",
                )
                self.gui_inc_coverage_status = self.server.gui.add_markdown(
                    "**coverage:** (off)"
                )

        self.gui_inc_save_btn.on_click(lambda _ev: self._on_save_checkpoint_click())
        self.gui_inc_load_btn.on_click(lambda _ev: self._on_load_checkpoint_click())
        self.gui_inc_video_btn.on_click(lambda _ev: self._on_append_video_click())
        self.gui_inc_frames_btn.on_click(lambda _ev: self._on_append_frames_click())
        # Sampling controls auto-apply on change — calls the host's sampling
        # callback with current widget values, which forwards to the trainer.
        self.gui_inc_sampling_mode.on_update(lambda _ev: self._on_sampling_change())
        self.gui_inc_new_frame_weight.on_update(lambda _ev: self._on_sampling_change())
        self.gui_inc_sampling_horizon.on_update(lambda _ev: self._on_sampling_change())

        # Freeze mode auto-applies on change; the mask itself is only
        # (re)computed when the user clicks the button.
        self.gui_inc_freeze_mode.on_update(lambda _ev: self._on_freeze_mode_change())
        self.gui_inc_freeze_btn.on_click(lambda _ev: self._on_recompute_freeze_click())

        # Dedup multiplier auto-applies — next Append picks it up.
        if self._on_seed_dedup_mult_change is not None:
            self.gui_inc_dedup_multiplier.on_update(
                lambda _ev: self._on_seed_dedup_mult_change(
                    float(self.gui_inc_dedup_multiplier.value)
                )
            )

        # Debug viz wiring.
        self.gui_inc_voxel_overlap_show.on_update(
            lambda _ev: self._on_voxel_overlap_toggle()
        )
        self.gui_inc_voxel_overlap_btn.on_click(
            lambda _ev: self._on_voxel_overlap_recompute()
        )
        self.gui_inc_coverage_show.on_update(
            lambda _ev: self._on_coverage_toggle()
        )
        self.gui_inc_coverage_btn.on_click(
            lambda _ev: self._on_coverage_recompute()
        )

    def _set_inc_status(self, msg: str) -> None:
        try:
            self.gui_inc_status.content = msg
        except Exception:
            pass

    def _pause_training_quietly(self) -> None:
        if self.training_control is None:
            return
        try:
            self.training_control.pause()
        except Exception:
            pass

    def _on_save_checkpoint_click(self) -> None:
        if self._save_checkpoint_cb is None:
            self._set_inc_status("**incremental:** save callback missing")
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self._set_inc_status("**incremental:** busy — another operation is running")
            return
        try:
            self.gui_inc_save_btn.disabled = True
            self._pause_training_quietly()
            self._set_inc_status("**incremental:** saving checkpoint…")
            meta = self._save_checkpoint_cb(str(self.gui_inc_ckpt_path.value))
            self._set_inc_status(
                f"**incremental:** saved → {meta['splat_count']:,} splats "
                f"@ step {meta['step_count']} — click resume"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_inc_status(f"**incremental:** save error — {type(e).__name__}: {e}")
        finally:
            self.gui_inc_save_btn.disabled = False
            self._trainer_op_lock.release()

    def _on_load_checkpoint_click(self) -> None:
        if self._load_checkpoint_cb is None:
            self._set_inc_status("**incremental:** load callback missing")
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self._set_inc_status("**incremental:** busy — another operation is running")
            return
        try:
            self.gui_inc_load_btn.disabled = True
            self._pause_training_quietly()
            self._set_inc_status("**incremental:** loading checkpoint…")
            info = self._load_checkpoint_cb(str(self.gui_inc_ckpt_path.value))
            # Recompute the home pose so 'reset camera' targets the loaded bbox.
            if self.scene.means is not None and int(self.scene.num_splats) > 0:
                means_np = self.scene.means.detach().cpu().numpy()
                self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
            self._set_inc_status(
                f"**incremental:** loaded → {info['splat_count']:,} splats "
                f"@ step {info['step_count']} (sh={info['sh_max_deg']}, "
                f"mode={info['mode']}, frames={info['frame_count']}) "
                f"— click resume"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_inc_status(f"**incremental:** load error — {type(e).__name__}: {e}")
        finally:
            self.gui_inc_load_btn.disabled = False
            self._trainer_op_lock.release()

    def _format_append_status(self, action: str, info: dict) -> str:
        parts = [
            f"**incremental:** {action} {info['n_added']} frames",
            f"(epoch {info['epoch']}; skipped {info['skipped']})",
        ]
        seeded = info.get("n_seeded", 0)
        if seeded:
            cand = info.get("n_candidates", 0)
            dedup_drop = max(0, cand - info.get("n_after_dedup", 0))
            parts.append(
                f"+ seeded {seeded} splats (deduped {dedup_drop} vs existing voxels)"
            )
        parts.append("— click resume")
        return " ".join(parts)

    def _on_append_video_click(self) -> None:
        if self._append_video_cb is None:
            self._set_inc_status("**incremental:** append-video callback missing")
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self._set_inc_status("**incremental:** busy — another operation is running")
            return
        try:
            self.gui_inc_video_btn.disabled = True
            self._pause_training_quietly()
            self._set_inc_status("**incremental:** appending video (DA3 may take a minute)…")
            info = self._append_video_cb(
                str(self.gui_inc_video_path.value),
                int(self.gui_inc_video_max.value),
                bool(self.gui_inc_seed_splats.value),
            )
            self._set_inc_status(self._format_append_status("appended", info))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_inc_status(f"**incremental:** append-video error — {type(e).__name__}: {e}")
        finally:
            self.gui_inc_video_btn.disabled = False
            self._trainer_op_lock.release()

    def _on_sampling_change(self) -> None:
        """Phase 3: push the sampling mode + new-frame weight + horizon to
        the trainer whenever any of the three widgets changes. Fires often
        on slider drags, so the callback must be cheap (no model loads,
        no GPU work) — `SplatTrainer.set_sampling` just assigns three
        scalars. The result formats the per-epoch counts for the status."""
        if self._set_sampling_cb is None:
            return
        try:
            info = self._set_sampling_cb(
                str(self.gui_inc_sampling_mode.value),
                float(self.gui_inc_new_frame_weight.value),
                int(self.gui_inc_sampling_horizon.value),
            )
            counts = info.get("epoch_counts", {}) or {}
            count_str = ", ".join(
                f"ep{e}={n}" for e, n in sorted(counts.items())
            ) or "(no frames yet)"
            self.gui_inc_sampling_status.content = (
                f"**sampling:** {info.get('mode', '?')}, "
                f"new_w={info.get('new_frame_weight', 1.0):.2f}, "
                f"horizon={info.get('horizon', 0)}  | "
                f"frames: {count_str}"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self.gui_inc_sampling_status.content = (
                f"**sampling:** error — {type(e).__name__}: {e}"
            )

    def _on_voxel_overlap_toggle(self) -> None:
        """Phase 5.2: ON flips the visibility of the voxel-overlap layer
        and (re)computes if it's not already cached. OFF removes the layer
        from both the registry and viser so display-mode switches don't
        accidentally re-publish stale data."""
        if bool(self.gui_inc_voxel_overlap_show.value):
            self._on_voxel_overlap_recompute()
        else:
            self._teardown_debug_layer(self._voxel_overlap_layer_name)
            self.gui_inc_voxel_overlap_status.content = "**voxel overlap:** (off)"

    def _on_voxel_overlap_recompute(self) -> None:
        if self._compute_voxel_overlap_cb is None:
            self.gui_inc_voxel_overlap_status.content = (
                "**voxel overlap:** callback missing"
            )
            return
        try:
            self.gui_inc_voxel_overlap_btn.disabled = True
            self.gui_inc_voxel_overlap_status.content = "**voxel overlap:** computing…"
            info = self._compute_voxel_overlap_cb(
                float(self.gui_inc_voxel_overlap_mult.value)
            )
            pts = info["points"]
            cols = info["colors"]
            if pts.shape[0] == 0:
                self.gui_inc_voxel_overlap_status.content = (
                    "**voxel overlap:** no splats — initialize + train first"
                )
                return
            # Size each rendered point ≈ voxel cell so the overlay looks
            # like a low-res scene mask, not pinpricks.
            point_size = max(float(info.get("voxel_size", 0.01)) * 0.6, 1e-4)
            layer = PointCloudLayer(
                name=self._voxel_overlap_layer_name,
                points=pts,
                colors_rgb=cols,
                point_size=point_size,
            )
            # Use the canonical layer pipeline so display-mode + per-layer
            # visibility checkbox both work. _push_layer no-ops in modes
            # other than "points" — that matches the existing splat_centers
            # behaviour and avoids covering the splat render with a 3D PC.
            self._add_or_refresh_debug_layer(layer)
            mode_hint = ""
            if self.display_mode != "points":
                mode_hint = " — switch Display to 'points' to see it"
            self.gui_inc_voxel_overlap_status.content = (
                f"**voxel overlap:** voxel={info['voxel_size']:.4f}m, "
                f"{info['n_voxels']:,} voxels — "
                f"old={info['n_old_only']:,}, new={info['n_new_only']:,}, "
                f"shared={info['n_shared']:,}{mode_hint}"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self.gui_inc_voxel_overlap_status.content = (
                f"**voxel overlap:** error — {type(e).__name__}: {e}"
            )
        finally:
            self.gui_inc_voxel_overlap_btn.disabled = False

    def _on_coverage_toggle(self) -> None:
        """Phase 5.3: ON (re)computes + adds the coverage point cloud; OFF
        tears it down (registry + viser handle) so display-mode switches
        don't reanimate it."""
        if bool(self.gui_inc_coverage_show.value):
            self._on_coverage_recompute()
        else:
            self._teardown_debug_layer(self._coverage_layer_name)
            self.gui_inc_coverage_status.content = "**coverage:** (off)"

    def _on_coverage_recompute(self) -> None:
        if self._compute_coverage_cb is None:
            self.gui_inc_coverage_status.content = "**coverage:** callback missing"
            return
        try:
            self.gui_inc_coverage_btn.disabled = True
            self.gui_inc_coverage_status.content = "**coverage:** computing…"
            info = self._compute_coverage_cb(
                str(self.gui_inc_coverage_scope.value)
            )
            pts = info["points"]
            cols = info["colors"]
            if pts.shape[0] == 0:
                self.gui_inc_coverage_status.content = (
                    "**coverage:** no splats — initialize first"
                )
                return
            # Size each rendered point ~ the scene diag's small fraction —
            # the splat density may be high so keep points small.
            mn = pts.min(axis=0); mx = pts.max(axis=0)
            diag = float(np.linalg.norm(mx - mn))
            point_size = max(diag * 2e-3, 1e-4)
            layer = PointCloudLayer(
                name=self._coverage_layer_name,
                points=pts,
                colors_rgb=cols,
                point_size=point_size,
            )
            self._add_or_refresh_debug_layer(layer)
            mode_hint = ""
            if self.display_mode != "points":
                mode_hint = " — switch Display to 'points' to see it"
            self.gui_inc_coverage_status.content = (
                f"**coverage:** scope={info['scope']}, "
                f"cameras={info['n_cameras']}, "
                f"max_count={info['max_count']}, "
                f"unseen={info['n_unseen']:,} of {info['n_total']:,}"
                f"{mode_hint}"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self.gui_inc_coverage_status.content = (
                f"**coverage:** error — {type(e).__name__}: {e}"
            )
        finally:
            self.gui_inc_coverage_btn.disabled = False

    def _add_or_refresh_debug_layer(self, layer: PointCloudLayer) -> None:
        """Phase 5 debug-layer helper: add (or replace) `layer` in the scene
        registry, ensure a per-layer GUI panel exists, and call _push_layer
        so it respects display_mode + the per-layer visibility checkbox.

        Matches the lifecycle `_refresh_splat_centers_layer` uses for the
        live splat-centers debug layer, so display-mode switches clean it
        up correctly via `_apply_display_mode` -> `_remove_pushed`. The
        previous direct `server.scene.add_point_cloud(...)` was the bug
        that caused debug overlays to obstruct splat rendering."""
        self.scene.add_point_cloud(layer)
        name = layer.name
        if name not in self._handles:
            try:
                with self._point_clouds_folder:
                    self._build_layer_gui(name)
            except Exception as e:
                print(f"_add_or_refresh_debug_layer: GUI build failed for {name}: {e}")
        else:
            # Refresh the count readout; layer's contents change on
            # recompute, the existing GUI panel can stay.
            try:
                self._handles[name]["count"].content = (
                    f"**count:** {len(layer.points):,}"
                )
            except Exception:
                pass
        self._push_layer(name)

    def _teardown_debug_layer(self, name: str) -> None:
        """Phase 5 debug-layer helper: remove from the scene registry, the
        viser scene, and the pushed-set so display-mode switches don't
        come back to a stale viser handle."""
        self.scene.remove_point_cloud(name)
        try:
            self._remove_pushed(name)
        except Exception:
            pass

    def _on_freeze_mode_change(self) -> None:
        """Phase 4: flip the freeze policy. Doesn't recompute the mask
        (that's the explicit button) — just updates `trainer._freeze_mode`
        and reflects whether the existing mask is still in effect."""
        if self._set_freeze_mode_cb is None:
            return
        try:
            info = self._set_freeze_mode_cb(str(self.gui_inc_freeze_mode.value))
            self.gui_inc_freeze_status.content = self._format_freeze_status(info)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.gui_inc_freeze_status.content = (
                f"**freeze:** error — {type(e).__name__}: {e}"
            )

    def _on_recompute_freeze_click(self) -> None:
        """Phase 4: rebuild the freeze mask from the current splat positions
        + latest-epoch cameras. Cheap; safe to spam. Pauses training so the
        step() loop can't race the mask-swap."""
        if self._recompute_freeze_mask_cb is None:
            return
        try:
            self.gui_inc_freeze_btn.disabled = True
            self._pause_training_quietly()
            self.gui_inc_freeze_status.content = "**freeze:** computing…"
            info = self._recompute_freeze_mask_cb()
            self.gui_inc_freeze_status.content = self._format_freeze_status(info)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.gui_inc_freeze_status.content = (
                f"**freeze:** error — {type(e).__name__}: {e}"
            )
        finally:
            self.gui_inc_freeze_btn.disabled = False

    def _format_freeze_status(self, info: dict) -> str:
        mode = info.get("mode", "?")
        if mode == "off":
            return "**freeze:** off (train all splats)"
        total = info.get("n_total", 0)
        train = info.get("n_trainable", 0)
        frozen = info.get("n_frozen", 0)
        cams = info.get("n_cameras", 0)
        epoch = info.get("latest_epoch")
        if total == 0 or cams == 0:
            return (
                f"**freeze:** mode={mode} (no mask — initialize + append + "
                f"click Recompute)"
            )
        pct = 100.0 * train / max(total, 1)
        return (
            f"**freeze:** mode={mode}, latest_epoch={epoch}, "
            f"cameras={cams} | trainable={train:,} ({pct:.1f}%), "
            f"frozen={frozen:,} of {total:,}"
        )

    def _on_append_frames_click(self) -> None:
        if self._append_frames_cb is None:
            self._set_inc_status("**incremental:** append-frames callback missing")
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self._set_inc_status("**incremental:** busy — another operation is running")
            return
        try:
            self.gui_inc_frames_btn.disabled = True
            self._pause_training_quietly()
            self._set_inc_status("**incremental:** appending supplied frames…")
            info = self._append_frames_cb(
                str(self.gui_inc_frames_dir.value),
                bool(self.gui_inc_seed_splats.value),
            )
            self._set_inc_status(self._format_append_status("appended", info))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_inc_status(f"**incremental:** append-frames error — {type(e).__name__}: {e}")
        finally:
            self.gui_inc_frames_btn.disabled = False
            self._trainer_op_lock.release()

    def _build_mesh_panel(self) -> None:
        """Mesh generation controls (Phase 4)."""
        with self.server.gui.add_folder("Mesh"):
            self.gui_mesh_mode = self.server.gui.add_dropdown(
                "mode",
                options=["tsdf", "dlnr", "dc"],
                initial_value="dlnr",
                hint="TSDF (soft alpha) / DLNR (stereo depth, colored) / "
                     "DC (Dual Contouring on splat Hermite data — best for "
                     "2DGS scenes with sharp features)",
            )
            self.gui_mesh_ncams = self.server.gui.add_number(
                "cameras",
                initial_value=96,
                min=8, max=192, step=1,
                hint="Fibonacci dome camera count (more = better coverage)",
            )
            self.gui_mesh_density = self.server.gui.add_slider(
                "density",
                initial_value=0.02,
                min=0.005, max=0.05, step=0.001,
                hint="Truncation margin = density * scene_diag (smaller = denser mesh)",
            )
            self.gui_mesh_shell = self.server.gui.add_slider(
                "shell_thickness",
                initial_value=6.0,
                min=3.0, max=10.0, step=0.5,
                hint="voxel_size = truncation_margin / shell (higher = finer voxels)",
            )
            self.gui_mesh_alpha = self.server.gui.add_slider(
                "alpha_thresh",
                initial_value=0.5,
                min=0.1, max=0.9, step=0.1,
                hint="Drop pixels with alpha below this threshold",
            )
            self.gui_mesh_btn = self.server.gui.add_button("Generate Mesh")
            self.gui_mesh_bake_size = self.server.gui.add_dropdown(
                "tex_size",
                options=["512", "1024", "2048", "4096"],
                initial_value="1024",
                hint="Resolution of baked texture (PNG side length)",
            )
            self.gui_mesh_target_faces = self.server.gui.add_number(
                "target_faces",
                initial_value=100_000,
                min=10_000, max=500_000, step=10_000,
                hint="Decimate mesh to this many faces before UV unwrap & bake (notebook default 100k)",
            )
            self.gui_mesh_bake_mode = self.server.gui.add_dropdown(
                "bake_mode",
                options=["vertex_colors", "splat_projection"],
                initial_value="vertex_colors",
                hint="vertex_colors = fast DLNR-color interp; splat_projection = high quality, samples splat RGB per texel from dome cameras",
            )
            self.gui_mesh_bake_btn = self.server.gui.add_button("Bake Texture")
            self.gui_mesh_lighting = self.server.gui.add_checkbox(
                "lighting",
                initial_value=True,
                hint="Toggle scene lighting (on = lit baseColor; off = emissive only)",
            )
            self.gui_mesh_wireframe = self.server.gui.add_checkbox(
                "wireframe",
                initial_value=False,
                hint="Render mesh as wireframe instead of solid",
            )
            self.gui_mesh_status = self.server.gui.add_markdown("_No mesh_")
        self.gui_mesh_btn.on_click(lambda _ev: self._on_mesh_click())
        self.gui_mesh_bake_btn.on_click(lambda _ev: self._on_bake_click())
        self.gui_mesh_wireframe.on_update(lambda _ev: self._on_wireframe_toggle())
        self.gui_mesh_lighting.on_update(lambda _ev: self._on_lighting_toggle())
        # Apply initial lighting state
        self._on_lighting_toggle()

    def _on_mesh_click(self) -> None:
        """Run mesh generation on main thread (required for DLNR signal handling)."""
        if self._trainer_ref is None or self._trainer_ref.data is None:
            self.gui_mesh_status.content = "_Init first_"
            return
        self.gui_mesh_status.content = "_Running…_"
        self._run_mesh()

    def _run_mesh(self) -> None:
        """Generate mesh from current trainer state."""
        try:
            from mesher import generate_mesh
            import torch.nn.functional as F

            t = self._trainer_ref
            if t is None or t.data is None:
                self.gui_mesh_status.content = "_Error: trainer not ready_"
                return

            with torch.no_grad():
                means = t.means_t.detach()
                quats = F.normalize(t.quats_t.detach(), dim=-1)
                scales = torch.exp(t.log_s_t.detach())
                opacities = torch.sigmoid(t.logit_o_t.detach())
                # Get full SH: DC + higher-order bands
                p = t.train.params
                sh = torch.cat([p["sh0"], p["shN"]], dim=1).detach()

            def progress(i, n):
                self.gui_mesh_status.content = f"_Cam {i}/{n}_"

            # Determine output path for mesh files (PLY/OBJ/MTL/PNG)
            # Fall back to a debug dir if trainer didn't set a name.
            name = getattr(t, 'name', None)
            output_root = getattr(t, 'output_root', None) or Path("vipe_outputs")
            if not name:
                import time as _t
                name = f"debug_{int(_t.time())}"
            out_dir = Path(output_root) / name
            out_path = out_dir / "mesh"
            import sys
            print(f"  mesh: output dir = {out_dir.resolve()}", file=sys.stderr, flush=True)

            # Fingerprint = (step_count, num_splats). Changes whenever the splat
            # state advances or its size changes, invalidating the DLNR cache.
            step_count = int(getattr(t, "_step_count", 0))
            fingerprint = hash((step_count, int(means.shape[0])))

            result = generate_mesh(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                sh=sh,
                device=str(means.device),
                mesh_mode=self.gui_mesh_mode.value,
                rasterizer_mode=str(getattr(t, "mode", "3dgs")),
                n_cams=int(self.gui_mesh_ncams.value),
                density=float(self.gui_mesh_density.value),
                shell_thickness=float(self.gui_mesh_shell.value),
                alpha_thresh=float(self.gui_mesh_alpha.value),
                bake_texture=False,
                out_path=out_path,
                progress_cb=progress,
                splat_fingerprint=fingerprint,
            )

            # Persist the raw mesh so we can bake textures later
            self._last_mesh_result = result
            self._last_mesh_out_path = out_path

            # Store in scene under lock
            with self.scene._lock:
                self.scene.mesh_verts = result["verts"]
                self.scene.mesh_faces = result["faces"]
                self.scene.mesh_normals = result["normals"]
                self.scene.mesh_colors = result.get("colors", None)

            # Push to viser (calls add_mesh_simple or updates existing)
            self._push_mesh_to_viser(result)

            v, f = result["verts"].shape[0], result["faces"].shape[0]
            self.gui_mesh_status.content = f"_Done: {v:,}v {f:,}f_"

        except Exception as e:
            import traceback
            self.gui_mesh_status.content = f"_Error: {str(e)}_"
            traceback.print_exc()

    def _on_bake_click(self) -> None:
        """Bake the last mesh into a texture and re-display the textured mesh."""
        try:
            res = getattr(self, "_last_mesh_result", None)
            if res is None:
                self.gui_mesh_status.content = "_Generate a mesh first_"
                return

            tex_size = int(self.gui_mesh_bake_size.value)
            target_faces = int(self.gui_mesh_target_faces.value)
            mode = self.gui_mesh_bake_mode.value
            out_path = getattr(self, "_last_mesh_out_path", None)

            def progress(i, n):
                self.gui_mesh_status.content = f"_Baking ({mode}) {i}/{n}…_"

            if mode == "splat_projection":
                # Photogrammetric: pull splat tensors from the trainer (same as _run_mesh)
                from mesher import bake_texture_from_splat_projection
                import torch.nn.functional as F
                t = self._trainer_ref
                if t is None or t.data is None:
                    self.gui_mesh_status.content = "_splat_projection needs a trainer_"
                    return
                with torch.no_grad():
                    means = t.means_t.detach()
                    quats = F.normalize(t.quats_t.detach(), dim=-1)
                    scales = torch.exp(t.log_s_t.detach())
                    opacities = torch.sigmoid(t.logit_o_t.detach())
                    p = t.train.params
                    sh = torch.cat([p["sh0"], p["shN"]], dim=1).detach()

                self.gui_mesh_status.content = (
                    f"_Baking (splat_projection) → {tex_size}px texture, {target_faces:,} faces…_"
                )
                baked = bake_texture_from_splat_projection(
                    verts=res["verts"], faces=res["faces"],
                    means=means, quats=quats, scales=scales, opacities=opacities, sh=sh,
                    tex_size=tex_size, target_faces=target_faces,
                    n_cams=int(self.gui_mesh_ncams.value),
                    image_size=1024,
                    device=str(means.device),
                    out_path=out_path, progress_cb=progress,
                )
            else:
                # vertex_colors: fast DLNR-color interpolation
                from mesher import bake_texture_from_splats
                if res.get("colors") is None:
                    self.gui_mesh_status.content = "_vertex_colors bake needs a DLNR mesh_"
                    return
                self.gui_mesh_status.content = (
                    f"_Baking (vertex_colors) → {tex_size}px texture, {target_faces:,} faces…_"
                )
                baked = bake_texture_from_splats(
                    verts=res["verts"], faces=res["faces"], colors=res["colors"],
                    tex_size=tex_size, target_faces=target_faces,
                    out_path=out_path, progress_cb=progress,
                )

            import sys, time

            # Try to load the just-saved OBJ via trimesh — it's the same file
            # MeshLab renders correctly. If trimesh loads it cleanly, we hand
            # the resulting Trimesh to viser. If anything goes wrong, fall back
            # to the vertex-colored decimated mesh.
            push_res = None
            if out_path is not None:
                try:
                    import trimesh
                    obj_path = Path(out_path).with_stem("mesh_splat_tex").with_suffix(".obj")
                    if obj_path.exists():
                        loaded = trimesh.load(str(obj_path), process=False, force="mesh")
                        push_res = {"_trimesh_direct": loaded}
                        print(f"  bake: loaded OBJ from disk → trimesh ({len(loaded.vertices)} verts, {len(loaded.faces)} faces)",
                              file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"  bake: OBJ-load fallback failed: {e}", file=sys.stderr, flush=True)
                    push_res = None

            if push_res is None:
                # Fallback: vertex-colored decimated mesh
                push_res = {
                    "verts": baked["verts"],
                    "faces": baked["faces"],
                    "colors": baked["colors"],
                }

            t_push = time.perf_counter()
            print(f"  bake: pushing mesh to viser…", file=sys.stderr, flush=True)
            self._push_mesh_to_viser(push_res)
            print(f"  bake: viser push done in {time.perf_counter() - t_push:.1f}s",
                  file=sys.stderr, flush=True)
            self.gui_mesh_status.content = f"_Baked ({tex_size}px, {target_faces:,} faces)_"

        except Exception as e:
            import traceback
            self.gui_mesh_status.content = f"_Bake error: {str(e)}_"
            traceback.print_exc()

    def _push_mesh_to_viser(self, result: dict) -> None:
        """Add or update mesh in viser scene."""
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
        if self._mesh_wire_handle is not None:
            self._mesh_wire_handle.remove()
            self._mesh_wire_handle = None

        # Shortcut: a pre-built trimesh.Trimesh (e.g., loaded from the saved OBJ)
        # gets pushed straight to viser without any rebuilding.
        direct = result.get("_trimesh_direct", None)
        if direct is not None:
            self._mesh_handle = self.server.scene.add_mesh_trimesh(
                name="/mesh", mesh=direct,
                visible=(self.display_mode == "mesh"),
            )
            return

        verts = result["verts"]
        faces = result["faces"]
        uv = result.get("uv", None)
        tex_image = result.get("tex_image", None)
        colors = result.get("colors", None)

        if uv is not None and tex_image is not None:
            # Textured mesh — emissive material so PBR lighting doesn't darken it.
            # ALWAYS use per-corner (duplicated-vertex) layout. Each face has its
            # own 3 unique vertices with their own UVs. This avoids any
            # per-vertex-UV interpretation differences in trimesh/three.js when
            # neighboring faces would share a vertex (and thus a single UV) at
            # a UV seam.
            import trimesh
            if result.get("per_vertex_uv"):
                # bake returned (verts, faces=F_uv, uv per vertex). Expand to per-corner.
                tri_verts = verts[faces].reshape(-1, 3)
                tri_uv    = uv[faces].reshape(-1, 2)
            else:
                # Legacy per-corner layout: uv already (F, 3, 2)
                tri_verts = verts[faces].reshape(-1, 3)
                tri_uv = uv.reshape(-1, 2)
            tri_faces = np.arange(len(tri_verts)).reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=tri_verts, faces=tri_faces, process=False)
            # Use the simplest possible TextureVisuals: only uv + image, no custom
            # material. trimesh will build a default PBRMaterial with baseColorTexture
            # pointing to `image`. This avoids any double-texture / sampler-mismatch
            # issues we were hitting with the custom PBRMaterial.
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=tri_uv, image=tex_image,
            )
            self._mesh_handle = self.server.scene.add_mesh_trimesh(
                name="/mesh", mesh=mesh,
                visible=(self.display_mode == "mesh"),
            )
        elif colors is not None:
            # DLNR mesh with per-vertex colors (no texture baked yet)
            import trimesh
            rgba = np.column_stack([
                (colors.clip(0, 1) * 255).astype(np.uint8),
                np.full(len(colors), 255, dtype=np.uint8),
            ])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.visual.vertex_colors = rgba
            self._mesh_handle = self.server.scene.add_mesh_trimesh(
                name="/mesh", mesh=mesh,
                visible=(self.display_mode == "mesh"),
            )
        else:
            # TSDF mesh — grey
            self._mesh_handle = self.server.scene.add_mesh_simple(
                name="/mesh",
                vertices=verts, faces=faces,
                color=(200, 200, 200),
                visible=(self.display_mode == "mesh"),
            )

        # Optional wireframe overlay on top of the colored mesh
        if getattr(self, "gui_mesh_wireframe", None) is not None and self.gui_mesh_wireframe.value:
            self._mesh_wire_handle = self.server.scene.add_mesh_simple(
                name="/mesh_wire",
                vertices=verts, faces=faces,
                color=(0, 0, 0),
                wireframe=True,
                visible=(self.display_mode == "mesh"),
            )

    def _on_lighting_toggle(self) -> None:
        """Enable or disable the scene's default lights."""
        try:
            self.server.scene.configure_default_lights(
                enabled=bool(self.gui_mesh_lighting.value),
                cast_shadow=False,
            )
        except Exception as e:
            print(f"lighting toggle failed: {e}")

    def _on_wireframe_toggle(self) -> None:
        """Add/remove wireframe overlay on the currently-displayed mesh."""
        res = getattr(self, "_last_mesh_result", None)
        if res is None or self._mesh_handle is None:
            return

        if self.gui_mesh_wireframe.value:
            if self._mesh_wire_handle is not None:
                self._mesh_wire_handle.remove()
            self._mesh_wire_handle = self.server.scene.add_mesh_simple(
                name="/mesh_wire",
                vertices=res["verts"], faces=res["faces"],
                color=(0, 0, 0),
                wireframe=True,
                visible=(self.display_mode == "mesh"),
            )
        else:
            if self._mesh_wire_handle is not None:
                self._mesh_wire_handle.remove()
                self._mesh_wire_handle = None

    def _build_inspect_panel(self) -> None:
        # Top-level: Display toggle (splats vs point clouds vs mesh).
        self.gui_display = self.server.gui.add_dropdown(
            "Display",
            options=("splats", "points", "mesh"),
            initial_value="splats",
            hint="Switch the whole scene between splat rasterization, point clouds, and mesh.",
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
                "render_mode", options=("RGB", "Depth", "Normals"), initial_value="RGB"
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
                "adaptive_res", initial_value=False,
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
                "show", initial_value=False,
                hint="Render a frustum for every training-frame camera.",
            )
            self.gui_train_cam_scale = self.server.gui.add_slider(
                "scale", min=0.01, max=2.0, step=0.01, initial_value=0.15,
                hint="Frustum size in world units.",
            )
            self.gui_train_cam_images = self.server.gui.add_checkbox(
                "show images", initial_value=False,
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
        mode = self.display_mode
        if mode == "mesh":
            # Hide splats background, hide point clouds, show mesh
            for client in self.server.get_clients().values():
                client.scene.set_background_image(None)
            for name in list(self.scene.point_clouds.keys()):
                self._remove_pushed(name)
            if self._mesh_handle is not None:
                self._mesh_handle.visible = True
            if self._mesh_wire_handle is not None:
                self._mesh_wire_handle.visible = True
        else:
            # Hide mesh in non-mesh modes
            if self._mesh_handle is not None:
                self._mesh_handle.visible = False
            if self._mesh_wire_handle is not None:
                self._mesh_wire_handle.visible = False
            if mode == "splats":
                for name in list(self.scene.point_clouds.keys()):
                    self._remove_pushed(name)
                for client in self.server.get_clients().values():
                    self._render_for(client)
            else:  # "points"
                for client in self.server.get_clients().values():
                    client.scene.set_background_image(None)
                self._push_all_visible_layers()

    # ---- Per-layer GUI ----------------------------------------------- #

    def _build_layer_gui(self, name: str) -> None:
        layer = self.scene.point_clouds[name]
        # Range slider relative to the layer's derived size — for DA3 scenes
        # this lands in a useful neighborhood instead of capping at 0.1.
        base = max(float(layer.point_size), 1e-4)
        size_min = max(base * 0.1, 1e-5)
        size_max = base * 2.0
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
        colors: np.ndarray | None = None,
    ) -> None:
        """Cache the per-frame camera payload and push frustums into the
        scene. Pass an empty c2w (shape (0, ...)) to clear.

        `colors` is an optional (N, 3) uint8 array, one RGB per camera.
        When provided, each frustum is drawn in its assigned color; when
        None, all frustums fall back to the default orange. Used by
        Phase 5.1 to tint frustums by their `frame_epoch`."""
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
        cols_np: np.ndarray | None
        if colors is None:
            cols_np = None
        else:
            cols_arr = np.asarray(colors)
            if cols_arr.shape != (c2w_np.shape[0], 3):
                raise ValueError(
                    f"colors shape must be ({c2w_np.shape[0]}, 3); "
                    f"got {tuple(cols_arr.shape)}"
                )
            cols_np = cols_arr.astype(np.uint8)
        self._train_cam_payload = {
            "c2w": c2w_np, "K": K_np, "images": imgs_np,
            "H": int(H), "W": int(W),
            "colors": cols_np,
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
        cols = payload.get("colors")
        H, W = payload["H"], payload["W"]
        scale = float(self.gui_train_cam_scale.value)
        with_images = bool(self.gui_train_cam_images.value) and imgs is not None
        aspect = W / max(H, 1)
        default_color = (255, 153, 51)
        for i in range(c2w.shape[0]):
            R = c2w[i, :3, :3]
            t = c2w[i, :3, 3]
            wxyz = _rotmat_to_wxyz(R)
            fy = float(K[i, 1, 1])
            fov_y = 2.0 * math.atan(0.5 * H / max(fy, 1e-9))
            name = f"train_cams/{i:04d}"
            cam_color = (
                tuple(int(x) for x in cols[i])
                if cols is not None else default_color
            )
            self.server.scene.add_camera_frustum(
                name=name,
                fov=fov_y,
                aspect=aspect,
                scale=scale,
                color=cam_color,
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
        size = resolved.point_size
        viser_name = _viser_pc_name(name)
        try:
            bbox_min = resolved.points.min(axis=0)
            bbox_max = resolved.points.max(axis=0)
            print(
                f"[push_layer {name}] N={len(resolved.points)} "
                f"size={size:.5f} color_mode={resolved.color_mode} "
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
            self.gui_train_loss_breakdown = self.server.gui.add_markdown(
                "`breakdown: —`"
            )
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

    # ---- Demo tab ----------------------------------------------------- #

    def _snap_home_to_scene(self) -> None:
        """Recompute the home pose from the current splats and push it to
        every connected client so 'reset camera' / the orbit up-axis track
        the new bbox. No-op when the scene is empty. Shared by Init and Demo
        so re-init from either path reorients identically."""
        if self.scene.means is None:
            return
        means_np = self.scene.means.detach().cpu().numpy()
        self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
        for client in self.server.get_clients().values():
            try:
                client.camera.up_direction = self.home_up
                client.camera.position = self.home_position
                client.camera.look_at = self.home_look_at
            except Exception:
                pass

    def _link_widgets(self, a, b) -> None:
        """Two-way bind two viser input handles so editing either updates the
        other. Setting a handle's `.value` runs the partner's on_update
        callbacks synchronously on the same thread, so without a guard the
        echo would recurse; we kill it with a re-entrancy flag. The flag is
        *per pair* (closed over here), not a shared instance attribute —
        viser dispatches callbacks on a 32-worker thread pool, so a single
        shared flag would let a concurrent edit of a different pair see the
        flag set and silently drop its sync. Any *other* callbacks attached
        to the partner (e.g. the Train tab's live `_on_scale_mult_change` /
        `_on_seed_dedup_mult_change`) still fire, so the demo duplicates get
        those side effects for free without re-wiring them here."""
        if a is None or b is None:
            return
        syncing = {"v": False}  # per-pair guard, isolated from other pairs

        def _make(src, dst):
            def _cb(_ev):
                if syncing["v"]:
                    return
                syncing["v"] = True
                try:
                    dst.value = src.value
                except Exception:
                    pass
                finally:
                    syncing["v"] = False
            return _cb

        a.on_update(_make(a, b))
        b.on_update(_make(b, a))

    def _build_demo_panel(self) -> None:
        """Incremental-first capture: upload an image + prompt, hit Request
        video, and the generation server's clip is auto-run through the same
        pose+init pipeline (first video) or an incremental append (every video
        after). Settings duplicate the Train tab's and stay two-way synced.
        Built after the Train tab so the gui_init_* handles exist to link."""
        d = self._demo_defaults
        # Resolution presets for the dropdown — queried from the server's
        # /resolutions endpoint, with a static fallback if it isn't up yet.
        import video_api
        res_options, res_default = video_api.fetch_resolutions(
            str(d.get("server_url", ""))
        )
        default_res = str(d.get("resolution", res_default))
        if default_res not in res_options:
            res_options = (default_res, *res_options)
        # Camera-motion options — queried from the server's /trajectories
        # endpoint, with a static fallback if it isn't up yet.
        traj_options, dir_options, traj_default, dir_default = video_api.fetch_trajectories(
            str(d.get("server_url", ""))
        )
        default_traj = str(d.get("trajectory", traj_default))
        if default_traj not in traj_options:
            traj_options = (default_traj, *traj_options)
        default_dir = str(d.get("direction", dir_default))
        if default_dir not in dir_options:
            dir_options = (default_dir, *dir_options)
        with self.server.gui.add_folder("Generate"):
            self.gui_demo_server = self.server.gui.add_text(
                "server URL",
                initial_value=str(d.get("server_url", "")),
                hint="POST endpoint of the video-generation server. The image "
                     "+ prompt go up as multipart/form-data; the response "
                     "carries the video (~30 s). Live — read on each request.",
            )
            self.gui_demo_prompt = self.server.gui.add_text(
                "prompt",
                initial_value=str(d.get("prompt", "")),
                hint="Text prompt sent alongside the image.",
            )
            self.gui_demo_image_btn = self.server.gui.add_upload_button(
                "image", mime_type="image/*",
                hint="Conditioning frame. Uploaded from your browser and "
                     "forwarded to the server with the prompt.",
            )
            self.gui_demo_image_status = self.server.gui.add_markdown(
                "**image:** none uploaded"
            )
            self.gui_demo_request_btn = self.server.gui.add_button(
                "Request video",
                hint="Send image + prompt, wait for the video, then auto-run "
                     "pose + init (first video) or an incremental append "
                     "(subsequent videos).",
            )
            self.gui_demo_status = self.server.gui.add_markdown(
                "**demo:** upload an image + a prompt, then Request video"
            )
            self.gui_demo_count = self.server.gui.add_markdown(
                "**videos processed:** 0"
            )

        with self.server.gui.add_folder("Lyra2 camera"):
            # One camera move per Request. These map 1:1 onto demo_server's
            # /generate form fields (resolution, trajectory, direction,
            # num_frames, strength).
            self.gui_demo_resolution = self.server.gui.add_dropdown(
                "resolution",
                options=res_options,
                initial_value=default_res,
                hint="Output video resolution. Smaller = faster + less VRAM. "
                     "Presets are fetched from the server's /resolutions "
                     "endpoint (480p is the model's native size). The server "
                     "also accepts a raw 'H,W' if you edit a preset in.",
            )
            self.gui_demo_trajectory = self.server.gui.add_dropdown(
                "trajectory",
                options=traj_options,
                initial_value=default_traj,
                hint="Camera motion for this clip (one move per Request). "
                     "Options come from the server's /trajectories endpoint, "
                     "e.g. horizontal_zoom (dolly in), horizontal (strafe), "
                     "orbit_horizontal, spiral, dolly_zoom, rotate_spot, back, "
                     "original (locked-off).",
            )
            self.gui_demo_direction = self.server.gui.add_dropdown(
                "direction",
                options=dir_options,
                initial_value=default_dir,
                hint="Direction of the move (left/right/up/down). Meaning "
                     "depends on the trajectory — e.g. for horizontal_zoom, "
                     "right = forward (in), left = backward (out).",
            )
            self.gui_demo_num_frames = self.server.gui.add_number(
                "num_frames",
                initial_value=int(d.get("num_frames", 81)),
                min=81, max=801, step=80,
                hint="Frames in the clip. Must be 1 + 80k (81, 161, 241, …) to "
                     "align with AR chunk boundaries. More frames spread the "
                     "same motion over a slower, smoother move.",
            )
            self.gui_demo_strength = self.server.gui.add_slider(
                "strength",
                min=0.0, max=3.0, step=0.1,
                initial_value=float(d.get("strength", 0.5)),
                hint="Magnitude of the move (distance for dolly/strafe, angle "
                     "for orbits). A built-in collision check can cap forward "
                     "motion, so larger values don't always push further.",
            )
            self.gui_demo_continue = self.server.gui.add_checkbox(
                "continue from last clip",
                initial_value=False,
                hint="On: the NEXT Request seeds from the LAST FRAME of the "
                     "previously generated clip, so the camera keeps moving "
                     "instead of restarting from the uploaded image. Off: every "
                     "Request restarts from the uploaded image (repeating the "
                     "same move yields the same clip). Note: each clip re-grounds "
                     "depth, so continuity is visual and drifts over many hops.",
            )

        with self.server.gui.add_folder("Settings (synced with Train)"):
            # Initial values are pulled from the already-built Train widgets so
            # the two start in sync; _link_widgets keeps them that way.
            self.gui_demo_max_frames = self.server.gui.add_number(
                "max_frames",
                initial_value=int(d.get("max_frames", self.gui_init_max_frames.value)),
                min=1, max=1000, step=1,
            )
            self.gui_demo_conf_q = self.server.gui.add_slider(
                "confidence_quantile", min=0.0, max=1.0, step=0.01,
                initial_value=float(self.gui_init_conf_q.value),
            )
            self.gui_demo_remove_sky = self.server.gui.add_checkbox(
                "remove_sky", initial_value=bool(self.gui_init_remove_sky.value),
            )
            self.gui_demo_sh_max_deg = self.server.gui.add_number(
                "sh_max_deg", initial_value=int(self.gui_init_sh_max_deg.value),
                min=0, max=3, step=1,
            )
            self.gui_demo_lpips_weight = self.server.gui.add_slider(
                "lpips_weight", min=0.0, max=1.0, step=0.01,
                initial_value=float(self.gui_init_lpips_weight.value),
            )
            self.gui_demo_void_weight = self.server.gui.add_slider(
                "void_weight", min=0.0, max=2.0, step=0.05,
                initial_value=float(self.gui_init_void_weight.value),
            )
            self.gui_demo_scale_mult = self.server.gui.add_slider(
                "max_scale_voxels", min=0.5, max=10.0, step=0.1,
                initial_value=float(self.gui_init_scale_mult.value),
            )
            self.gui_demo_densify = self.server.gui.add_checkbox(
                "densify", initial_value=bool(self.gui_init_densify.value),
            )
            self.gui_demo_mode = self.server.gui.add_dropdown(
                "mode", options=("3dgs", "2dgs"),
                initial_value=str(self.gui_init_mode.value),
            )
            # The seed-dedup slider lives in the Train tab's Incremental folder
            # and only exists when incremental callbacks were wired.
            self.gui_demo_dedup = None
            if getattr(self, "gui_inc_dedup_multiplier", None) is not None:
                self.gui_demo_dedup = self.server.gui.add_slider(
                    "seed dedup radius (× init voxel)",
                    min=1.0, max=6.0, step=0.5,
                    initial_value=float(self.gui_inc_dedup_multiplier.value),
                    hint="Dedup-check radius for new-video splats, in init-"
                         "voxel units (same control as Train→Incremental).",
                )

        with self.server.gui.add_folder("Incremental"):
            self.gui_demo_freeze = self.server.gui.add_checkbox(
                "freeze old splats on new video",
                initial_value=True,
                hint="After appending a new video, freeze splats not visible "
                     "to the new cameras (set freeze mode = new_frustums + "
                     "recompute) so only the new region refines. Off = train "
                     "every splat.",
            )
            self.gui_demo_autotrain = self.server.gui.add_checkbox(
                "auto-train after each video",
                initial_value=False,
                hint="Off (default): a loaded video is processed into splats "
                     "but training stays paused until you click Train. On: "
                     "resume optimization automatically once a video is "
                     "processed.",
            )

        with self.server.gui.add_folder("Training"):
            self.gui_demo_train_btn = self.server.gui.add_button("Train")
            self.gui_demo_pause_btn = self.server.gui.add_button("Pause")
            self.gui_demo_reinit_btn = self.server.gui.add_button(
                "Re-initialize",
                hint="Erase the current splats and re-initialize from the "
                     "first fetched video (the seed clip), so its cameras "
                     "come back. Incremental appends are discarded — Request "
                     "more videos to rebuild them. Useful after changing a "
                     "slider to see its effect on a clean init. Does not fetch "
                     "a new video and does not auto-train.",
            )
            self.gui_demo_reset_btn = self.server.gui.add_button("Reset")
            self.gui_demo_train_status = self.server.gui.add_markdown(
                "**status:** stopped"
            )
            self.gui_demo_step = self.server.gui.add_markdown("**step:** 0")
            self.gui_demo_splat_count = self.server.gui.add_markdown(
                f"**splats:** {int(self.scene.num_splats):,}"
            )

        # Action wiring.
        self.gui_demo_image_btn.on_upload(lambda _ev: self._on_demo_image_upload())
        self.gui_demo_request_btn.on_click(lambda _ev: self._on_demo_request_click())
        self.gui_demo_train_btn.on_click(lambda _ev: self._on_resume_training())
        self.gui_demo_pause_btn.on_click(lambda _ev: self._on_pause_training())
        self.gui_demo_reinit_btn.on_click(lambda _ev: self._on_demo_reinit_click())
        self.gui_demo_reset_btn.on_click(lambda _ev: self._on_demo_reset_click())

        # Two-way sync each duplicate with its Train-tab twin.
        self._link_widgets(self.gui_demo_max_frames, self.gui_init_max_frames)
        self._link_widgets(self.gui_demo_conf_q, self.gui_init_conf_q)
        self._link_widgets(self.gui_demo_remove_sky, self.gui_init_remove_sky)
        self._link_widgets(self.gui_demo_sh_max_deg, self.gui_init_sh_max_deg)
        self._link_widgets(self.gui_demo_lpips_weight, self.gui_init_lpips_weight)
        self._link_widgets(self.gui_demo_void_weight, self.gui_init_void_weight)
        self._link_widgets(self.gui_demo_scale_mult, self.gui_init_scale_mult)
        self._link_widgets(self.gui_demo_densify, self.gui_init_densify)
        self._link_widgets(self.gui_demo_mode, self.gui_init_mode)
        if self.gui_demo_dedup is not None:
            self._link_widgets(self.gui_demo_dedup, self.gui_inc_dedup_multiplier)

    def _on_demo_image_upload(self) -> None:
        f = self.gui_demo_image_btn.value
        content = getattr(f, "content", None)
        if not content:
            return
        self._demo_image = (getattr(f, "name", "image.png"), content)
        # A fresh upload starts a new scene — drop any carried-over last frame so
        # "continue from last clip" doesn't seed from the previous image's clip.
        self._demo_last_frame = None
        self.gui_demo_image_status.content = (
            f"**image:** {self._demo_image[0]} ({len(content) / 1024.0:,.0f} KB)"
        )

    def _collect_demo_init_opts(self, video_path: str) -> dict:
        """Build the initializer opts dict from the (synced) Train-tab widgets.
        Same keys/shape as _on_init_click so the existing initializer closure
        handles it unchanged."""
        return dict(
            video=Path(video_path),
            max_frames=int(self.gui_init_max_frames.value),
            confidence_quantile=float(self.gui_init_conf_q.value),
            remove_sky=bool(self.gui_init_remove_sky.value),
            sh_max_deg=int(self.gui_init_sh_max_deg.value),
            lpips_weight=float(self.gui_init_lpips_weight.value),
            void_weight=float(self.gui_init_void_weight.value),
            use_densify=bool(self.gui_init_densify.value),
            mode=str(self.gui_init_mode.value),
        )

    def _collect_demo_gen_opts(self) -> dict:
        """Lyra2 single-trajectory options forwarded to the generation server
        (maps onto demo_server's /generate form fields)."""
        return dict(
            resolution=str(self.gui_demo_resolution.value),
            trajectory=str(self.gui_demo_trajectory.value),
            direction=str(self.gui_demo_direction.value),
            num_frames=int(self.gui_demo_num_frames.value),
            strength=float(self.gui_demo_strength.value),
        )

    def _extract_last_frame(self, video_path: str) -> "tuple[str, bytes] | None":
        """Read the final frame of an mp4 and return (name, PNG bytes).

        Used to continue the camera move on the next Request. Returns None on any
        failure so a read hiccup never breaks the generate flow.
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            frame = None
            ret, f = cap.read()
            while ret:
                frame = f
                ret, f = cap.read()  # walk to the genuine last decodable frame
            cap.release()
            if frame is None:
                return None
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                return None
            return ("last_frame.png", buf.tobytes())
        except Exception:
            return None

    def _on_demo_request_click(self) -> None:
        """Fetch one video from the generation server and turn it into splats:
        prepare_and_init for the very first video, append_video (incremental)
        for every one after. Runs in viser's thread pool, so the ~30 s server
        call + DA3 may block here."""
        if self._request_video_cb is None:
            self.gui_demo_status.content = "**demo:** no generation server wired"
            return
        if self._demo_image is None:
            self.gui_demo_status.content = "**demo:** upload an image first"
            return
        # Prompt is optional — the server falls back to a generic caption.
        prompt = str(self.gui_demo_prompt.value).strip()
        # Serialize against any other trainer-mutating handler (a Reset from
        # either tab, Initialize, Append…) so they can't free/replace the
        # param tensors while we're mid append/init on this pool thread.
        if not self._trainer_op_lock.acquire(blocking=False):
            self.gui_demo_status.content = "**demo:** busy — another operation is running"
            return
        try:
            self.gui_demo_request_btn.disabled = True
            self.gui_demo_reset_btn.disabled = True
            self.gui_demo_reinit_btn.disabled = True
            # Pause while we generate + run DA3 so the optimizer isn't racing
            # the param-tensor surgery that append_video does.
            self._pause_training_quietly()
            # Use the same conservative "is there a finished scene" signal the
            # rest of the app uses. `trainer.train` is set mid-init (before
            # _initialized flips True), so a prior failed init could otherwise
            # mis-route the next request into an incremental append.
            had_splats = bool(getattr(self._trainer_ref, "_initialized", False))
            # Seed image: continue from the previous clip's last frame when the
            # toggle is on and we have one, else the uploaded image. This makes
            # the camera actually progress across Requests instead of restarting.
            continue_on = bool(getattr(self, "gui_demo_continue", None)
                               and self.gui_demo_continue.value)
            if continue_on and self._demo_last_frame is not None:
                name, image_bytes = self._demo_last_frame
                self.gui_demo_status.content = "**demo:** requesting video (continuing)…"
            else:
                name, image_bytes = self._demo_image
                self.gui_demo_status.content = "**demo:** requesting video (~30 s)…"
            video_path = str(self._request_video_cb(
                image_bytes, name, prompt, str(self.gui_demo_server.value),
                self._collect_demo_gen_opts(),
            ))
            # Capture this clip's final frame so the next Request can continue
            # from it (used only when "continue from last clip" is on).
            lf = self._extract_last_frame(video_path)
            if lf is not None:
                self._demo_last_frame = lf

            if not had_splats:
                self.gui_demo_status.content = "**demo:** first video — pose + init (DA3)…"
                self._initializer(self._collect_demo_init_opts(video_path))
                # Remember the seed video so Re-initialize re-inits from it
                # (not from a later appended clip).
                self._demo_init_video = video_path
                self._snap_home_to_scene()
            elif self._append_video_cb is None:
                self.gui_demo_status.content = "**demo:** append-video callback missing"
                return
            else:
                self.gui_demo_status.content = (
                    "**demo:** new video — incremental append (DA3)…"
                )
                self._append_video_cb(
                    video_path, int(self.gui_init_max_frames.value), True,
                )
                # Freeze toggle drives the trainer in BOTH directions so
                # unchecking it actually unfreezes (otherwise a stale
                # new_frustums mask keeps freezing old splats forever).
                freeze_on = bool(self.gui_demo_freeze.value)
                if self._set_freeze_mode_cb is not None:
                    self._set_freeze_mode_cb("new_frustums" if freeze_on else "off")
                    fm = getattr(self, "gui_inc_freeze_mode", None)
                    if fm is not None:
                        try:
                            fm.value = "new_frustums" if freeze_on else "off"
                        except Exception:
                            pass
                if freeze_on and self._recompute_freeze_mask_cb is not None:
                    self._recompute_freeze_mask_cb()

            auto = bool(self.gui_demo_autotrain.value)
            if auto:
                self._on_resume_training()
            self._demo_count += 1
            self.gui_demo_count.content = f"**videos processed:** {self._demo_count}"
            self.gui_demo_status.content = (
                f"**demo:** ready ({self.scene.num_splats:,} splats) — "
                + ("training" if auto else "click Train")
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_demo_status.content = f"**demo:** error — {type(e).__name__}: {e}"
        finally:
            self.gui_demo_request_btn.disabled = False
            self.gui_demo_reset_btn.disabled = False
            self.gui_demo_reinit_btn.disabled = False
            self._trainer_op_lock.release()

    def _on_demo_reset_click(self) -> None:
        """Erase everything (same teardown as the Train-tab Reset), then clear
        the demo counter. Keeps the uploaded image + prompt so the user can
        immediately re-generate."""
        if not self._trainer_op_lock.acquire(blocking=False):
            self.gui_demo_status.content = "**demo:** busy — wait for the current operation"
            return
        try:
            self.gui_demo_reset_btn.disabled = True
            if self.training_control is not None:
                try:
                    self.training_control.pause()
                except Exception:
                    pass
            if self._resetter is not None:
                self._resetter()
            self._demo_count = 0
            # Forget the seed so Re-initialize has nothing to rebuild until a
            # new video is fetched (Reset is a clean start-over).
            self._demo_init_video = None
            # Drop the carried-over last frame so "continue from last clip"
            # starts fresh after a Reset.
            self._demo_last_frame = None
            self.gui_demo_count.content = "**videos processed:** 0"
            self.gui_demo_splat_count.content = "**splats:** 0"
            self.gui_demo_status.content = (
                "**demo:** reset — upload + Request video to start over"
            )
        except Exception as e:
            self.gui_demo_status.content = f"**demo:** reset error — {type(e).__name__}: {e}"
        finally:
            self.gui_demo_reset_btn.disabled = False
            self._trainer_op_lock.release()

    def _on_demo_reinit_click(self) -> None:
        """Erase the current splats and re-initialize from the FIRST fetched
        video (the one that seeded the scene), so the original cameras come
        back. Incremental appends are discarded — that's the designed clean
        re-init (append_video nulls the trainer's _last_video so this
        re-preprocesses the seed clip fresh rather than seeding from the
        appended frames still in trainer.data). Does NOT fetch a new clip and
        does NOT auto-train, so you can tweak a slider and see its effect on a
        clean init; Request more videos to rebuild the incremental scene."""
        if self._demo_init_video is None:
            self.gui_demo_status.content = (
                "**demo:** no video yet — click Request video first"
            )
            return
        if self._initializer is None:
            self.gui_demo_status.content = "**demo:** initializer not wired"
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self.gui_demo_status.content = "**demo:** busy — another operation is running"
            return
        try:
            self.gui_demo_reinit_btn.disabled = True
            self.gui_demo_reset_btn.disabled = True
            self.gui_demo_request_btn.disabled = True
            if self.training_control is not None:
                try:
                    self.training_control.pause()
                except Exception:
                    pass
            # Tear down current splats, then re-init on the seed video with the
            # current slider values. After any append the trainer's
            # _last_video is None, so prepare_and_init re-preprocesses the seed
            # clip fresh → trainer.data + cameras are exactly the first video's.
            self.gui_demo_status.content = "**demo:** re-initializing — clearing splats…"
            if self._resetter is not None:
                self._resetter()
            self.gui_demo_status.content = "**demo:** re-initializing — pose + init (DA3)…"
            self._initializer(self._collect_demo_init_opts(self._demo_init_video))
            self._snap_home_to_scene()
            # Back to a single-clip scene; appends were discarded.
            self._demo_count = 1
            self.gui_demo_count.content = "**videos processed:** 1"
            self.gui_demo_status.content = (
                f"**demo:** re-initialized from {Path(self._demo_init_video).name} "
                f"({self.scene.num_splats:,} splats; appends cleared) — click Train"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_demo_status.content = f"**demo:** re-init error — {type(e).__name__}: {e}"
        finally:
            self.gui_demo_reinit_btn.disabled = False
            self.gui_demo_reset_btn.disabled = False
            self.gui_demo_request_btn.disabled = False
            self._trainer_op_lock.release()

    def _on_init_click(self) -> None:
        """Run the user-supplied initializer with the current Init-panel
        values. Updates status + recomputes the home pose so 'reset camera'
        targets the new bbox."""
        if self._initializer is None:
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self.gui_init_status.content = "**init:** busy — another operation is running"
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
                void_weight=float(self.gui_init_void_weight.value),
                use_densify=bool(self.gui_init_densify.value),
                mode=str(self.gui_init_mode.value),
            )
            # viser runs on_click in a thread pool, so blocking here is fine.
            self._initializer(opts)
            # Auto-snap Mesh-tab sliders to the rasterizer-mode defaults
            # (2DGS wants tighter truncation + finer voxels than 3DGS — see
            # mesher.default_tsdf_params for the values). Skipped silently if
            # the user doesn't have a trainer or the mesh panel didn't build.
            t = self._trainer_ref
            if (t is not None and getattr(self, "gui_mesh_density", None) is not None
                    and getattr(t, "mode", None) is not None):
                try:
                    from mesher import default_tsdf_params
                    p = default_tsdf_params(str(t.mode))
                    self.gui_mesh_density.value = float(p["density"])
                    self.gui_mesh_shell.value = float(p["shell_thickness"])
                except Exception:
                    pass
            # Recompute home pose from the newly-populated scene + push it to
            # every connected client so 'reset camera' / the orbit up-axis
            # track the new bbox.
            self._snap_home_to_scene()
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
            self._trainer_op_lock.release()

    def _on_reset_click(self) -> None:
        """Pause training (caller's responsibility actually pauses; we also
        try via training_control) and ask the user's resetter to tear down."""
        if not self._trainer_op_lock.acquire(blocking=False):
            self.gui_init_status.content = "**init:** busy — another operation is running"
            return
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
            self._trainer_op_lock.release()

    def _on_prune_click(self) -> None:
        """Pause training, run `trainer.prune_splats()`, report counts, and
        leave training paused so the user can inspect before resuming."""
        t = self._trainer_ref
        if t is None or getattr(t, "prune_splats", None) is None:
            self.gui_init_status.content = "**prune:** trainer doesn't support pruning"
            return
        if not getattr(t, "_initialized", False):
            self.gui_init_status.content = "**prune:** initialize training first"
            return
        if not self._trainer_op_lock.acquire(blocking=False):
            self.gui_init_status.content = "**prune:** busy — another operation is running"
            return
        try:
            self.gui_init_prune_btn.disabled = True
            if self.training_control is not None:
                try:
                    self.training_control.pause()
                except Exception:
                    pass
            self.gui_init_status.content = "**prune:** running…"
            counts = t.prune_splats()
            # Recompute home pose so reset-camera targets the post-prune bbox.
            if self.scene.means is not None and int(self.scene.num_splats) > 0:
                means_np = self.scene.means.detach().cpu().numpy()
                self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
            parts = [f"{k}={counts[k]}" for k in ("opacity", "scale", "aniso", "knn")
                     if k in counts and counts[k] >= 0]
            detail = ", ".join(parts) if parts else "no filters fired"
            self.gui_init_status.content = (
                f"**prune:** {counts['started']:,} → {counts['kept']:,} "
                f"(−{counts['removed_total']:,}; {detail}) — click resume"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_init_status.content = f"**prune:** error — {type(e).__name__}: {e}"
        finally:
            self.gui_init_prune_btn.disabled = False
            self._trainer_op_lock.release()

    @staticmethod
    def _make_loss_figure(
        history: list[tuple[int, float]],
        components_history: list[tuple[int, dict[str, float]]] | None = None,
    ):
        """Build a small plotly figure from a (step, loss) list plus an
        optional per-term components history. Both are downsampled
        stride-uniformly to ≤1000 points so websocket pushes stay cheap.
        Y-axis is log so L1 (~0.005) and depth_sup (~5) can be read on the
        same plot. Click traces in the legend to hide/show."""
        import plotly.graph_objects as go
        h = history
        if len(h) > 1000:
            stride = len(h) // 1000
            h = h[::stride]
        xs = [p[0] for p in h]
        ys = [p[1] for p in h]
        traces = [go.Scatter(x=xs, y=ys, mode="lines", name="loss",
                             line=dict(width=2, color="black"))]

        if components_history:
            ch = components_history
            if len(ch) > 1000:
                stride = len(ch) // 1000
                ch = ch[::stride]
            # Collect every key ever seen; build a per-key trace with only
            # the steps where that key was active (warmup-gated terms have
            # gaps before their start step).
            keys: list[str] = []
            for _, comps in ch:
                for k in comps:
                    if k not in keys:
                        keys.append(k)
            for k in keys:
                kx = [step for step, comps in ch if k in comps]
                ky = [comps[k] for _, comps in ch if k in comps]
                if kx:
                    traces.append(go.Scatter(x=kx, y=ky, mode="lines", name=k))

        fig = go.Figure(data=traces)
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            xaxis_title="step",
            yaxis_title="loss (log)",
            yaxis_type="log",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1.0),
        )
        return fig

    def _start_training_pump(self) -> None:
        self._pump_stop_event.clear()
        t = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread = t
        t.start()

    def _pump_loop(self) -> None:
        last_plot_t = 0.0
        last_log_t = 0.0
        last_splat_version = -1
        ctl = self.training_control
        while not self._pump_stop_event.is_set():
            time.sleep(0.1)  # 10 Hz readouts
            # Log drain runs even before training is wired so the user sees
            # boot/init output while just sitting on the Setup panel.
            now = time.perf_counter()
            if now - last_log_t >= 0.5:  # 2 Hz log push
                last_log_t = now
                try:
                    line = self._log_buf.drain_text()
                    if line is not None and hasattr(self, "gui_log"):
                        safe = line.replace("`", "'") if line else "(idle)"
                        self.gui_log.content = f"`{safe}`"
                except Exception:
                    pass
            if ctl is None:
                continue
            with self.scene.read() as s:
                step = s.step
                splats = s.num_splats
                history_snapshot = list(s.loss_history)
                components_snapshot = dict(s.last_loss_components)
                components_history_snapshot = list(s.loss_components_history)
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
                if components_snapshot:
                    self.gui_train_loss_breakdown.content = (
                        f"`{format_loss_components(components_snapshot)}`"
                    )
                else:
                    self.gui_train_loss_breakdown.content = "`breakdown: —`"
                self.gui_train_splat_count.content = f"**splats:** {int(splats):,}"
                self.gui_splat_count_readout.content = (
                    f"**splat_count:** {int(splats):,}"
                )
                # Mirror the live readouts into the Demo tab when it's built.
                if getattr(self, "gui_demo_train_status", None) is not None:
                    self.gui_demo_train_status.content = f"**status:** {ctl.status()}"
                    self.gui_demo_step.content = f"**step:** {int(step):,}"
                    self.gui_demo_splat_count.content = f"**splats:** {int(splats):,}"
            except Exception:
                # Server may have shut down; tolerate races on exit.
                continue
            if now - last_plot_t >= 0.5:  # 2 Hz plot
                try:
                    self.gui_train_loss_plot.figure = self._make_loss_figure(
                        history_snapshot, components_history_snapshot,
                    )
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
            try:
                self._log_buf.uninstall()
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
