"""
Phase 1 — static .ply rendering with gsplat.

Loads a 3DGS .ply file once, then serves a viser session that rasterizes the
scene from the live browser camera using `gsplat.rasterization`. Replaces the
synthetic Phase 0 renderer body; class boundaries (SceneState / Renderer /
ViewerApp / main) and the SSH-forward comment below are unchanged.

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
from dataclasses import dataclass
from pathlib import Path

import gsplat
import numpy as np
import torch
import viser
from plyfile import PlyData


# --------------------------------------------------------------------------- #
# Camera math (module-level so test_camera.py can import them directly)
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
    """World-to-camera 4x4 matrix in OpenCV convention (x-right, y-down, z-forward).

    Viser's CameraHandle.wxyz is the c2w rotation R in `P_world = [R | t] p_camera`,
    already in OpenCV camera-frame conventions (look=+Z, up=-Y, right=+X) per
    viser/_viser.py. No basis change is needed — invert to get world-to-camera.
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


# --------------------------------------------------------------------------- #
# PlyLoader
# --------------------------------------------------------------------------- #


@dataclass
class LoadedSplats:
    means: torch.Tensor       # (N, 3) float32
    quats: torch.Tensor       # (N, 4) wxyz, unit-normalized, float32
    scales: torch.Tensor      # (N, 3) float32 (already exp'd)
    opacities: torch.Tensor   # (N,)  float32 (already sigmoid'd)
    sh: torch.Tensor          # (N, K, 3) float32, K = (sh_degree + 1)**2
    sh_degree: int


class PlyLoader:
    """Parses a standard 3DGS .ply (Inria export) into CUDA tensors."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def load(self, path: str | Path, flip_x: bool = True) -> LoadedSplats:
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        prop_names = v.dtype.names

        means_np = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

        # Inria convention: opacity stored as logit, scales as log.
        opacities_np = _sigmoid(v["opacity"].astype(np.float32))
        scales_np = np.exp(
            np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
        )

        quats_np = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        ).astype(np.float32)
        quats_np = quats_np / np.linalg.norm(quats_np, axis=-1, keepdims=True).clip(min=1e-12)

        # SH coefficients.
        dc_np = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
        ).astype(np.float32)  # (N, 3)

        rest_props = sorted(
            (n for n in prop_names if n.startswith("f_rest_")),
            key=lambda s: int(s.split("_")[-1]),
        )
        if rest_props:
            assert len(rest_props) % 3 == 0, (
                f"f_rest_* count {len(rest_props)} not divisible by 3"
            )
            K_minus_1 = len(rest_props) // 3
            K = K_minus_1 + 1
            sh_degree = int(round(math.sqrt(K))) - 1
            assert (sh_degree + 1) ** 2 == K, (
                f"f_rest_* count implies K={K} which is not a perfect square"
            )
            rest_np = np.stack([v[n] for n in rest_props], axis=-1).astype(np.float32)
            # (N, 3 * (K-1)) -> (N, 3, K-1) -> (N, K-1, 3) (channels-last per gsplat)
            rest_np = rest_np.reshape(-1, 3, K_minus_1).transpose(0, 2, 1)
            sh_np = np.concatenate([dc_np[:, None, :], rest_np], axis=1)  # (N, K, 3)
        else:
            sh_degree = 0
            sh_np = dc_np[:, None, :]  # (N, 1, 3)

        if flip_x:
            # Inria PLYs are COLMAP-convention; rotate 180° about world X to
            # match viser's display. Means: (x, y, z) -> (x, -y, -z).
            # Quats: q_new = q_flip * q_old, with q_flip = (0, 1, 0, 0).
            means_np[:, 1:] *= -1.0
            q_flip = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            quats_np = _quat_mul_wxyz(
                np.broadcast_to(q_flip, quats_np.shape), quats_np
            )
            # Scales are per-axis magnitudes — never flipped.

        device = self.device
        return LoadedSplats(
            means=torch.from_numpy(means_np).to(device),
            quats=torch.from_numpy(quats_np).to(device),
            scales=torch.from_numpy(scales_np).to(device),
            opacities=torch.from_numpy(opacities_np).to(device),
            sh=torch.from_numpy(sh_np).to(device),
            sh_degree=sh_degree,
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# SceneState
# --------------------------------------------------------------------------- #


class SceneState:
    """Splat tensors guarded by an RLock. Phase 5's training thread will swap
    these the same way `load_from_ply` does."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.means: torch.Tensor | None = None
        self.quats: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.opacities: torch.Tensor | None = None
        self.sh: torch.Tensor | None = None
        self.sh_degree: int = 0
        self.num_splats: int = 0

    @contextmanager
    def read(self):
        with self._lock:
            yield self

    @contextmanager
    def write(self):
        with self._lock:
            yield self

    def load_from_ply(self, path: str | Path, device: torch.device, flip_x: bool = True) -> None:
        loaded = PlyLoader(device).load(path, flip_x=flip_x)
        with self.write():
            self.means = loaded.means
            self.quats = loaded.quats
            self.scales = loaded.scales
            self.opacities = loaded.opacities
            self.sh = loaded.sh
            self.sh_degree = loaded.sh_degree
            self.num_splats = int(loaded.means.shape[0])


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class Renderer:
    """Single render entry point. gsplat.rasterization in inference mode."""

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
                # No scene loaded yet — render solid black.
                return np.zeros((height, width, 3), dtype=np.uint8)

            viewmat_np = viser_camera_to_opencv_viewmat(camera.position, camera.wxyz)
            viewmats = torch.from_numpy(viewmat_np.astype(np.float32))[None].to(self.device)

            fov = float(camera.fov)
            fy = 0.5 * height / math.tan(0.5 * fov)
            fx = fy  # square pixels; aspect determines W/H, not the focal ratio
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
            # rgb: (1, H, W, 3) float in roughly [0, 1] (already 0.5-offset & clamped by gsplat).
            img = rgb[0].clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
            return img.cpu().numpy()


# --------------------------------------------------------------------------- #
# ViewerApp
# --------------------------------------------------------------------------- #


class ViewerApp:
    """Owns the viser server and per-client render dispatch. No GUI controls
    in Phase 1 — Phase 2 reintroduces them."""

    def __init__(self, ply_path: Path, host: str, port: int, flip_x: bool = True) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        self.scene = SceneState()
        self.renderer = Renderer(device=device_str)

        print(f"loading {ply_path} on {device_str}...")
        self.scene.load_from_ply(ply_path, device=device, flip_x=flip_x)
        print(
            f"loaded {self.scene.num_splats} splats, sh_degree={self.scene.sh_degree}, "
            f"flip_x={flip_x}"
        )

        self.server = viser.ViserServer(host=host, port=port)
        self.server.on_client_connect(self._on_client_connect)

    def _on_client_connect(self, client: viser.ClientHandle) -> None:
        client.camera.on_update(lambda _cam: self._render_for(client))
        self._render_for(client)

    def _render_for(self, client: viser.ClientHandle) -> None:
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
    parser = argparse.ArgumentParser(description="Phase 1 splat viewer (static .ply).")
    parser.add_argument("ply_path", type=Path, help="Path to a 3DGS .ply file.")
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Skip the 180°-about-X flip applied by default to Inria-format PLYs.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if not args.ply_path.exists():
        raise SystemExit(f"ply not found: {args.ply_path}")

    ViewerApp(
        ply_path=args.ply_path,
        host=args.host,
        port=args.port,
        flip_x=not args.no_flip,
    ).run()


if __name__ == "__main__":
    main()
