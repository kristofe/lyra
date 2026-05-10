"""
Phase 1.5 — splat rendering with a point-cloud toggle.

Loads a 3DGS .ply (Phase 1 path) and optionally additional point clouds via
`--points`. A "Render mode" GUI toggle switches the scene between gsplat-
rasterized splats and viser-native point clouds. The splat .ply is also
auto-derived as a `splat_centers` point-cloud layer so the toggle is useful
out of the box without any extra inputs.

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
    colors_rgb: np.ndarray         # (N, 3) uint8, "natural" colors from source
    metadata: dict = field(default_factory=dict)
    visible: bool = True
    point_size: float = 0.01       # world units
    color_mode: str = "rgb"        # "rgb" | "axis" | "confidence" | "uniform"
    uniform_color: tuple[int, int, int] = (255, 51, 51)


def compute_colors(layer: PointCloudLayer) -> np.ndarray:
    """Resolve the (N, 3) uint8 array to push to viser, given layer.color_mode."""
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
    """Loads point clouds from .ply, .npy, .npz. Subsamples to a max budget."""

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

    def _downsample(
        self, points: np.ndarray, colors: np.ndarray, extras: dict
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(points), size=self.max_points, replace=False)
        idx.sort()
        return (
            points[idx],
            colors[idx],
            {k: (v[idx] if hasattr(v, "__getitem__") else v) for k, v in extras.items()},
        )

    @staticmethod
    def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        points = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        names = set(v.dtype.names)
        if {"red", "green", "blue"}.issubset(names):
            r, g, b = v["red"], v["green"], v["blue"]
            if r.dtype.kind == "f":  # float in [0, 1]
                colors = (
                    np.stack([r, g, b], axis=-1).clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
            else:
                colors = np.stack([r, g, b], axis=-1).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 200, dtype=np.uint8)
        return points, colors, {}

    @staticmethod
    def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
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
    def _load_npy(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        points = np.load(str(path)).astype(np.float32)
        colors_path = path.parent / (path.stem + ".colors.npy")
        if colors_path.exists():
            colors = np.load(str(colors_path)).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 200, dtype=np.uint8)
        return points, colors, {}


def derive_splat_centers_layer(scene: "SceneState") -> PointCloudLayer | None:
    """Build a PointCloudLayer from the loaded splat scene's centers + DC color."""
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
# Splat .ply loader (unchanged from Phase 1)
# --------------------------------------------------------------------------- #


@dataclass
class LoadedSplats:
    means: torch.Tensor       # (N, 3) float32
    quats: torch.Tensor       # (N, 4) wxyz
    scales: torch.Tensor      # (N, 3) float32
    opacities: torch.Tensor   # (N,)  float32
    sh: torch.Tensor          # (N, K, 3) float32
    sh_degree: int


class PlyLoader:
    """Parses a 3DGS .ply (Inria-format export) into CUDA tensors."""

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
            assert len(rest_props) % 3 == 0, f"f_rest_* count {len(rest_props)} not /3"
            K_minus_1 = len(rest_props) // 3
            K = K_minus_1 + 1
            sh_degree = int(round(math.sqrt(K))) - 1
            assert (sh_degree + 1) ** 2 == K, f"K={K} not a perfect square"
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

    def load_from_ply(
        self, path: str | Path, device: torch.device, flip_x: bool = True
    ) -> None:
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
# Renderer (gsplat path, unchanged from Phase 1)
# --------------------------------------------------------------------------- #


class Renderer:
    """Single splat-render entry point. Point clouds bypass this entirely —
    viser draws them client-side."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)

    @torch.inference_mode()
    def render(
        self,
        scene: SceneState,
        camera: viser.CameraHandle,
        width: int,
        height: int,
    ) -> np.ndarray:
        with scene.read() as s:
            if s.num_splats == 0 or s.means is None:
                return np.zeros((height, width, 3), dtype=np.uint8)

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
                sh_degree=s.sh_degree,
                render_mode="RGB",
            )
            img = rgb[0].clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
            return img.cpu().numpy()


# --------------------------------------------------------------------------- #
# ViewerApp
# --------------------------------------------------------------------------- #


_VISER_PC_PREFIX = "pc/"


def _viser_pc_name(layer_name: str) -> str:
    return _VISER_PC_PREFIX + layer_name


class ViewerApp:
    """Owns the viser server, GUI controls, and dispatch between splat /
    point-cloud render modes."""

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
            print(
                f"  layer '{layer.name}': {len(layer.points)} pts, "
                f"metadata={list(layer.metadata.keys())}"
            )

        self.server = viser.ViserServer(host=host, port=port)
        # Per-layer GUI handles, keyed by layer name.
        self._handles: dict[str, dict[str, object]] = {}
        # Pushed viser names so we can clean up on switch.
        self._pushed: set[str] = set()

        # Top-level GUI: render mode + global size.
        self.render_mode_handle = self.server.gui.add_dropdown(
            "Render mode",
            options=("splats", "points"),
            initial_value="splats",
        )
        self.render_mode_handle.on_update(lambda _ev: self._apply_render_mode())

        with self.server.gui.add_folder("Point Clouds"):
            self.global_size_mult_handle = self.server.gui.add_slider(
                "global_size_mult", min=0.1, max=10.0, step=0.1, initial_value=1.0
            )
            self.global_size_mult_handle.on_update(
                lambda _ev: self._push_all_visible_layers()
            )
            for name in list(self.scene.point_clouds.keys()):
                self._build_layer_gui(name)

        self.server.on_client_connect(self._on_client_connect)

    # ---- mode-state predicates ---------------------------------------- #

    @property
    def render_mode(self) -> str:
        return str(self.render_mode_handle.value)

    # ---- per-layer GUI ------------------------------------------------ #

    def _build_layer_gui(self, name: str) -> None:
        layer = self.scene.point_clouds[name]
        with self.server.gui.add_folder(name):
            visible = self.server.gui.add_checkbox("visible", initial_value=layer.visible)
            size = self.server.gui.add_slider(
                "point_size",
                min=0.0005,
                max=0.1,
                step=0.0005,
                initial_value=layer.point_size,
            )
            color_mode = self.server.gui.add_dropdown(
                "color_mode",
                options=("rgb", "axis", "confidence", "uniform"),
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

    # ---- mode-switching ---------------------------------------------- #

    def _apply_render_mode(self) -> None:
        if self.render_mode == "splats":
            for name in list(self.scene.point_clouds.keys()):
                self._remove_pushed(name)
            for client in self.server.get_clients().values():
                self._render_for(client)
        else:  # points
            for client in self.server.get_clients().values():
                client.scene.set_background_image(None)
            self._push_all_visible_layers()

    def _push_all_visible_layers(self) -> None:
        for name in list(self.scene.point_clouds.keys()):
            self._push_layer(name)

    def _push_layer(self, name: str) -> None:
        layer = self.scene.point_clouds.get(name)
        handles = self._handles.get(name)
        if layer is None or handles is None:
            self._remove_pushed(name)
            return
        if self.render_mode != "points" or not bool(handles["visible"].value):
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
        # add_point_cloud is idempotent on name — replaces existing.
        self.server.scene.add_point_cloud(
            name=viser_name,
            points=resolved.points,
            colors=colors,
            point_size=size,
        )
        self._pushed.add(viser_name)

    def _remove_pushed(self, name: str) -> None:
        viser_name = _viser_pc_name(name)
        if viser_name in self._pushed:
            self.server.scene.remove_by_name(viser_name)
            self._pushed.discard(viser_name)

    # ---- client lifecycle / rendering -------------------------------- #

    def _on_client_connect(self, client: viser.ClientHandle) -> None:
        client.camera.on_update(lambda _cam: self._render_for(client))
        if self.render_mode == "splats":
            self._render_for(client)
        else:
            client.scene.set_background_image(None)

    def _render_for(self, client: viser.ClientHandle) -> None:
        if self.render_mode != "splats":
            return
        cam = client.camera
        try:
            W = int(cam.image_width)
            H = int(cam.image_height)
        except (TypeError, ValueError):
            return
        if W <= 0 or H <= 0:
            return
        img = self.renderer.render(self.scene, cam, W, H)
        client.scene.set_background_image(img)

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
        description="Phase 1.5 splat viewer with point-cloud toggle."
    )
    parser.add_argument("ply_path", type=Path, help="3DGS .ply scene to load.")
    parser.add_argument(
        "--points",
        type=Path,
        action="append",
        default=[],
        help="Additional point cloud (.ply / .npy / .npz). Repeatable.",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Skip the 180°-about-X flip applied by default to the splat .ply.",
    )
    parser.add_argument(
        "--no-derive-points",
        action="store_true",
        help="Don't auto-create the 'splat_centers' point cloud layer from the splat .ply.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help="Per-layer point cap (uniform random subsample with seed=42). 0 disables.",
    )
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
