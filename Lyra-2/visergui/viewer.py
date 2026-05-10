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
    point_size: float = 0.01
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
        point_size=0.005,
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
    """Splat tensors + named point-cloud layers, all guarded by an RLock."""

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
    ) -> tuple[np.ndarray, float]:
        with scene.read() as s:
            if s.num_splats == 0 or s.means is None:
                return np.zeros((height, width, 3), dtype=np.uint8), 0.0

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
            return img_np, (t1 - t0) * 1000.0


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
    ) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        self.scene = SceneState()
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
        self._handles: dict[str, dict[str, object]] = {}
        self._pushed: set[str] = set()

        # ---- Perf throttling state --------------------------------------- #
        self._fps_ema: float = 0.0
        self._render_ms_ema: float = 0.0
        self._last_frame_t: float = 0.0
        self._last_readout_push: float = 0.0
        self._last_cam_push: float = 0.0

        self._build_gui()
        self.server.on_client_connect(self._on_client_connect)

    # ---- GUI construction -------------------------------------------- #

    def _build_gui(self) -> None:
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
            self.gui_fps_readout = self.server.gui.add_number(
                "fps", initial_value=0.0, disabled=True
            )
            self.gui_render_ms_readout = self.server.gui.add_number(
                "render_ms", initial_value=0.0, disabled=True
            )

        with self.server.gui.add_folder("Scene"):
            self.gui_splat_count_readout = self.server.gui.add_number(
                "splat_count", initial_value=int(self.scene.num_splats), disabled=True
            )
            self.gui_camera_pos_readout = self.server.gui.add_vector3(
                "camera_pos", initial_value=(0.0, 0.0, 0.0), disabled=True
            )
            self.gui_camera_look_readout = self.server.gui.add_vector3(
                "look_dir", initial_value=(0.0, 0.0, 1.0), disabled=True
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
            count = self.server.gui.add_text(
                "count", initial_value=str(len(layer.points)), disabled=True
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

    def _render_for(self, client: viser.ClientHandle) -> None:
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

        max_dim = max(W, H)
        if max_dim > max_res:
            scale = max_res / max_dim
            W = max(1, int(round(W * scale)))
            H = max(1, int(round(H * scale)))

        img, render_ms = self.renderer.render(
            self.scene, cam, W, H,
            sh_degree=sh_degree,
            color_mode=color_mode,
            near=near,
            far=far,
        )
        client.scene.set_background_image(img)
        self._update_perf_readouts(render_ms)
        self._update_camera_readout(cam)

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
            self.gui_fps_readout.value = round(float(self._fps_ema), 1)
            self.gui_render_ms_readout.value = round(float(self._render_ms_ema), 2)
            self._last_readout_push = now

    def _update_camera_readout(self, cam: viser.CameraHandle) -> None:
        now = time.perf_counter()
        if now - self._last_cam_push < 0.1:  # 10 Hz
            return
        R = _quat_wxyz_to_rotmat(np.asarray(cam.wxyz))
        look = R[:, 2]  # camera +Z = forward in OpenCV = viser convention
        pos = np.asarray(cam.position)
        self.gui_camera_pos_readout.value = (
            round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3),
        )
        self.gui_camera_look_readout.value = (
            round(float(look[0]), 3), round(float(look[1]), 3), round(float(look[2]), 3),
        )
        self._last_cam_push = now

    # ---- Run ---------------------------------------------------------- #

    def run(self) -> None:
        host = self.server.get_host()
        port = self.server.get_port()
        print(f"viser server listening on http://{host}:{port}")
        self.server.sleep_forever()


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
