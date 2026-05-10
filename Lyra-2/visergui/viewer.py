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
from typing import Iterable

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


def derive_splat_centers_layer(scene: "SceneState") -> PointCloudLayer | None:
    with scene.read() as s:
        if s.means is None or s.num_splats == 0:
            return None
        means_np = s.means.detach().cpu().numpy().astype(np.float32)
        dc_sh = s.sh[:, 0, :].detach().cpu().numpy()
        rgb01 = (SH_C0 * dc_sh + 0.5).clip(0.0, 1.0)
        colors_u8 = (rgb01 * 255.0).astype(np.uint8)
    return PointCloudLayer(
        name="splat_centers",
        points=means_np,
        colors_rgb=colors_u8,
        point_size=0.0005,
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
    # Pull back along world +Z (viser default up) and slightly along +X for a
    # 3/4-ish view. Adjust empirically once you've seen your scene.
    offset = np.array([0.6, -0.6, 0.6], dtype=np.float64)
    offset = offset / np.linalg.norm(offset) * diag * 1.2
    pos = center + offset
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(center[0]), float(center[1]), float(center[2])),
        (0.0, 0.0, 1.0),
    )


class ViewerApp:
    """Owns the viser server, the GUI controls, and per-client render dispatch."""

    def __init__(
        self,
        ply_path: Path,
        host: str,
        port: int,
        flip_x: bool = True,
        extra_point_paths: Iterable[Path] = (),
        max_points: int = 1_000_000,
        derive_splat_points: bool = True,
        scene: SceneState | None = None,
        training_control: TrainingControl | None = None,
    ) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # Allow the caller to construct SceneState in advance — the trainer
        # needs a reference *before* the .ply finishes loading so it can
        # close over scene.write() inside its step function.
        self.scene = scene if scene is not None else SceneState()
        self.training_control = training_control
        self.renderer = Renderer(device=device_str)

        print(f"loading splat .ply {ply_path} on {device_str}...")
        self.scene.load_from_ply(ply_path, device=device, flip_x=flip_x)
        print(
            f"  {self.scene.num_splats} splats, sh_degree={self.scene.sh_degree}, "
            f"flip_x={flip_x}"
        )

        # Home pose computed once from the splat bbox.
        means_np = self.scene.means.detach().cpu().numpy()
        self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
        print(f"  home pose: pos={self.home_position} look_at={self.home_look_at}")

        if derive_splat_points:
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

        self._build_gui()
        if self.training_control is not None:
            self._start_training_pump()
        self.server.on_client_connect(self._on_client_connect)

    # ---- GUI construction -------------------------------------------- #

    def _build_gui(self) -> None:
        # If training is attached, split the right panel into two tabs so
        # the inspection controls and the training controls don't fight for
        # vertical space. Static viewer keeps the original flat layout.
        if self.training_control is not None:
            tabs = self.server.gui.add_tab_group()
            with tabs.add_tab("Inspect"):
                self._build_inspect_panel()
            with tabs.add_tab("Training"):
                self._build_training_gui(with_folder=False)
        else:
            self._build_inspect_panel()

    def _build_inspect_panel(self) -> None:
        # Top-level: Display toggle (splats vs point clouds).
        self.gui_display = self.server.gui.add_dropdown(
            "Display",
            options=("splats", "points"),
            initial_value="splats",
            hint="Switch the whole scene between splat rasterization and point clouds.",
        )
        self.gui_display.on_update(lambda _ev: self._apply_display_mode())

        sh_options = tuple(str(d) for d in range(self.scene.sh_degree + 1))
        with self.server.gui.add_folder("Render"):
            self.gui_sh_degree = self.server.gui.add_dropdown(
                "sh_degree",
                options=sh_options,
                initial_value=str(self.scene.sh_degree),
                hint="SH degree used by the rasterizer. Capped at the loaded .ply's degree.",
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

        with self.server.gui.add_folder("Point Clouds"):
            self.global_size_mult_handle = self.server.gui.add_slider(
                "global_size_mult", min=0.1, max=10.0, step=0.1, initial_value=1.0
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
        with self.server.gui.add_folder(name):
            visible = self.server.gui.add_checkbox("visible", initial_value=layer.visible)
            size = self.server.gui.add_slider(
                "point_size", min=0.0005, max=0.1, step=0.0005,
                initial_value=layer.point_size,
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

    def _push_layer(self, name: str) -> None:
        layer = self.scene.point_clouds.get(name)
        handles = self._handles.get(name)
        if layer is None or handles is None:
            self._remove_pushed(name)
            return
        if self.display_mode != "points" or not bool(handles["visible"].value):
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
        self.server.scene.add_point_cloud(
            name=viser_name, points=resolved.points, colors=colors, point_size=size,
        )
        self._pushed.add(viser_name)

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
        ctl = self.training_control
        while not self._pump_stop_event.is_set():
            time.sleep(0.1)  # 10 Hz readouts
            if ctl is None:
                continue
            with self.scene.read() as s:
                step = s.step
                splats = s.num_splats
                history_snapshot = list(s.loss_history)
            latest_loss = history_snapshot[-1][1] if history_snapshot else 0.0
            try:
                self.gui_train_status.content = f"**status:** {ctl.status()}"
                self.gui_train_step.content = f"**step:** {int(step):,}"
                self.gui_train_loss.content = f"**loss:** {float(latest_loss):.6f}"
                self.gui_train_splat_count.content = f"**splats:** {int(splats):,}"
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
