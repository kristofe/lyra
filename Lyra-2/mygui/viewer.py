"""Native 3D Gaussian Splatting viewer for Lyra-2.

    python viewer.py path/to/scene.ply [--no-flip] [--width 1280] [--height 720]

WASD = move planar, Space/Ctrl = up/down, RMB-drag = look, Shift = boost,
scroll = adjust speed. ImGui panel exposes FOV, speed, SH degree, render
mode (RGB / Depth), adaptive resolution, and a reset button.

Render path: gsplat -> torch CUDA tensor -> cudaMemcpy2DToArray to a CUDA-
registered GL texture -> fullscreen-quad blit. No host roundtrip.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import glfw
import moderngl
import numpy as np
import torch
from cuda.bindings import runtime as cudart
from gsplat import rasterization
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from plyfile import PlyData


GL_TEXTURE_2D = 0x0DE1


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------


def _check(ret, where: str):
    """cuda-python returns (err, *outputs). Validate the err."""
    if isinstance(ret, tuple):
        err = ret[0]
    else:
        err = ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA error at {where}: {err}")


# ---------------------------------------------------------------------------
# PLY loader
# ---------------------------------------------------------------------------


@dataclass
class GaussianScene:
    means: torch.Tensor       # (N, 3) float32
    quats: torch.Tensor       # (N, 4) wxyz, normalized
    scales: torch.Tensor      # (N, 3) post-exp
    opacities: torch.Tensor   # (N,) post-sigmoid
    sh_coeffs: torch.Tensor   # (N, K, 3) where K = (sh_degree+1)**2
    sh_degree: int


def _quat_left_multiply_180_x(quats_wxyz: torch.Tensor) -> torch.Tensor:
    # q_flip = (0, 1, 0, 0) in wxyz (180 deg rotation about +x).
    # (q_flip * q) = (-qx, qw, -qz, qy)
    qw, qx, qy, qz = quats_wxyz.unbind(dim=-1)
    return torch.stack([-qx, qw, -qz, qy], dim=-1)


def load_ply(path: str, device: torch.device, flip_to_y_up: bool = True) -> GaussianScene:
    plydata = PlyData.read(path)
    v = plydata.elements[0]

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacity_logit = np.asarray(v["opacity"]).astype(np.float32)

    scale_names = sorted(
        (p.name for p in v.properties if p.name.startswith("scale_")),
        key=lambda n: int(n.split("_")[-1]),
    )
    rot_names = sorted(
        (p.name for p in v.properties if p.name.startswith("rot_")),
        key=lambda n: int(n.split("_")[-1]),
    )
    f_dc_names = sorted(
        (p.name for p in v.properties if p.name.startswith("f_dc_")),
        key=lambda n: int(n.split("_")[-1]),
    )
    f_rest_names = sorted(
        (p.name for p in v.properties if p.name.startswith("f_rest_")),
        key=lambda n: int(n.split("_")[-1]),
    )

    if len(scale_names) != 3 or len(rot_names) != 4 or len(f_dc_names) != 3:
        raise ValueError(
            f"{path}: unexpected 3DGS layout "
            f"(scale={len(scale_names)}, rot={len(rot_names)}, f_dc={len(f_dc_names)})"
        )

    scales_log = np.stack([np.asarray(v[n]) for n in scale_names], axis=1).astype(np.float32)
    rots = np.stack([np.asarray(v[n]) for n in rot_names], axis=1).astype(np.float32)
    f_dc = np.stack([np.asarray(v[n]) for n in f_dc_names], axis=1).astype(np.float32)  # (N, 3)

    # f_rest layout in the Inria 3DGS convention: shape after read is
    # (N, 3 * (K - 1)), stored channel-major: first all R coeffs, then G,
    # then B. Reshape -> (N, 3, K-1) -> transpose to (N, K-1, 3).
    if f_rest_names:
        n_rest = len(f_rest_names) // 3
        if n_rest * 3 != len(f_rest_names):
            raise ValueError(f"f_rest_* count {len(f_rest_names)} not a multiple of 3")
        rest = np.stack([np.asarray(v[n]) for n in f_rest_names], axis=1).astype(np.float32)
        rest = rest.reshape(-1, 3, n_rest).transpose(0, 2, 1)  # (N, K-1, 3)
        sh_full = np.concatenate([f_dc[:, None, :], rest], axis=1)  # (N, K, 3)
    else:
        sh_full = f_dc[:, None, :]  # (N, 1, 3)

    K = sh_full.shape[1]
    sh_degree = int(round(math.sqrt(K))) - 1
    if (sh_degree + 1) ** 2 != K:
        raise ValueError(f"SH coefficient count {K} is not a square (deg+1)^2")

    means_t = torch.from_numpy(xyz.astype(np.float32)).to(device)
    quats_t = torch.from_numpy(rots).to(device)
    scales_t = torch.exp(torch.from_numpy(scales_log).to(device))
    opacity_t = torch.sigmoid(torch.from_numpy(opacity_logit).to(device))
    sh_t = torch.from_numpy(sh_full).to(device)

    quats_t = quats_t / torch.linalg.vector_norm(quats_t, dim=-1, keepdim=True).clamp_min(1e-12)

    if flip_to_y_up:
        # Inria/COLMAP scenes render upside down in y-up worlds: rotate 180 deg
        # about +x. Means: (x, y, z) -> (x, -y, -z); quats: q_flip * q.
        means_t[:, 1] = -means_t[:, 1]
        means_t[:, 2] = -means_t[:, 2]
        quats_t = _quat_left_multiply_180_x(quats_t)

    return GaussianScene(
        means=means_t.contiguous(),
        quats=quats_t.contiguous(),
        scales=scales_t.contiguous(),
        opacities=opacity_t.contiguous(),
        sh_coeffs=sh_t.contiguous(),
        sh_degree=sh_degree,
    )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class FlyCamera:
    """Position + yaw/pitch + FOV in OpenCV w2c convention.

    Yaw rotates around world +y (CCW from above). Pitch is a rotation that
    tilts the look direction up (+) / down (-); clamped to (-89 deg, 89 deg).
    """

    def __init__(self, position=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, fov_deg=60.0):
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov_deg = float(fov_deg)
        self.move_speed = 1.0
        self.is_moving = False
        self._last_motion_t = -1e9

        self._init_position = self.position.copy()
        self._init_yaw = self.yaw
        self._init_pitch = self.pitch
        self._init_fov = self.fov_deg

    def reset(self) -> None:
        self.position[:] = self._init_position
        self.yaw = self._init_yaw
        self.pitch = self._init_pitch
        self.fov_deg = self._init_fov

    @property
    def forward(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return np.array([sy * cp, sp, cy * cp], dtype=np.float64)

    @property
    def right(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([cy, 0.0, -sy], dtype=np.float64)

    @property
    def planar_forward(self) -> np.ndarray:
        # Forward projected to y=0 (so W/S walks rather than diving when pitched).
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([sy, 0.0, cy], dtype=np.float64)

    def viewmat_w2c(self) -> np.ndarray:
        # OpenCV: rows are right, down, forward (x_cam, y_cam, z_cam in world).
        f = self.forward
        r = self.right
        # down = right x forward (gives world -y at upright).
        d = np.cross(r, f)
        R = np.stack([r, d, f], axis=0)  # (3, 3)
        t = -R @ self.position
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def K_matrix(self, width: int, height: int) -> np.ndarray:
        fov_y = math.radians(self.fov_deg)
        fy = (height * 0.5) / math.tan(fov_y * 0.5)
        fx = fy
        K = np.array(
            [[fx, 0.0, width * 0.5],
             [0.0, fy, height * 0.5],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return K


class CameraController:
    """GLFW input -> FlyCamera, gated on ImGui capture flags."""

    BOOST = 5.0
    PITCH_LIMIT = math.radians(89.0)
    LOOK_SENSITIVITY = 0.0025  # rad / pixel
    SCROLL_SPEED_FACTOR = 1.15  # multiplicative per scroll tick

    def __init__(self, camera: FlyCamera, window):
        self.camera = camera
        self.window = window
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        self._scroll_acc = 0.0
        self._rmb_active = False
        self._last_mouse = None  # (x, y) at last rmb-active frame

        glfw.set_scroll_callback(window, self._on_scroll)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)

    def _on_scroll(self, window, xoff, yoff):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return
        self._scroll_acc += float(yoff)

    def _on_mouse_button(self, window, button, action, mods):
        io = imgui.get_io()
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                if io.want_capture_mouse:
                    return
                self._rmb_active = True
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                self._last_mouse = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE and self._rmb_active:
                self._rmb_active = False
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                self._last_mouse = None

    def update(self, dt: float) -> None:
        cam = self.camera
        io = imgui.get_io()

        moved = False

        # Mouse look (only while RMB is held, never while ImGui has the mouse).
        if self._rmb_active and not io.want_capture_mouse:
            x, y = glfw.get_cursor_pos(self.window)
            if self._last_mouse is not None:
                lx, ly = self._last_mouse
                dx = x - lx
                dy = y - ly
                if dx != 0.0 or dy != 0.0:
                    moved = True
                cam.yaw += dx * self.LOOK_SENSITIVITY
                cam.pitch -= dy * self.LOOK_SENSITIVITY
                cam.pitch = max(-self.PITCH_LIMIT, min(self.PITCH_LIMIT, cam.pitch))
            self._last_mouse = (x, y)

        # Scroll -> speed (multiplicative).
        if self._scroll_acc != 0.0:
            cam.move_speed *= self.SCROLL_SPEED_FACTOR ** self._scroll_acc
            cam.move_speed = float(np.clip(cam.move_speed, 0.05, 200.0))
            self._scroll_acc = 0.0

        # Keyboard.
        kb_blocked = io.want_capture_keyboard
        if not kb_blocked:
            forward = cam.planar_forward
            right = cam.right
            up = np.array([0.0, 1.0, 0.0])
            move = np.zeros(3, dtype=np.float64)
            if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
                move += forward
            if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                move -= forward
            if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
                move += right
            if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
                move -= right
            if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
                move += up
            if (glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                    or glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS):
                move -= up

            n = np.linalg.norm(move)
            if n > 0.0:
                move /= n
                speed = cam.move_speed
                if (glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                        or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS):
                    speed *= self.BOOST
                cam.position += move * speed * dt
                moved = True

        # is_moving sticks for 100 ms after last motion -- avoids flickering
        # the adaptive resolution between key presses.
        now = time.perf_counter()
        if moved:
            self._last_motion_t = now
        cam.is_moving = (now - self._last_motion_t) < 0.1


# ---------------------------------------------------------------------------
# CUDA-GL blitter
# ---------------------------------------------------------------------------


class CudaGLBlitter:
    """A GL texture registered with CUDA. ``upload((H, W, 4) uint8 cuda)`` blits."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self._resource = None
        self.texture = None
        self.width = 0
        self.height = 0
        self.resize(width, height)

    def _create(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.texture = self.ctx.texture((self.width, self.height), 4, dtype="f1")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        err, resource = cudart.cudaGraphicsGLRegisterImage(
            int(self.texture.glo), GL_TEXTURE_2D, int(flags)
        )
        _check(err, "cudaGraphicsGLRegisterImage")
        self._resource = resource

    def _destroy(self) -> None:
        if self._resource is not None:
            err, = cudart.cudaGraphicsUnregisterResource(self._resource)
            _check(err, "cudaGraphicsUnregisterResource")
            self._resource = None
        if self.texture is not None:
            self.texture.release()
            self.texture = None

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self._destroy()
        self._create(width, height)

    def upload(self, rgba_uint8: torch.Tensor) -> None:
        assert rgba_uint8.is_cuda and rgba_uint8.dtype == torch.uint8
        assert rgba_uint8.is_contiguous()
        H, W, C = rgba_uint8.shape
        if not (C == 4 and H == self.height and W == self.width):
            raise ValueError(
                f"blit shape mismatch: tensor {(H, W, C)} vs texture {(self.height, self.width, 4)}"
            )
        err, = cudart.cudaGraphicsMapResources(1, self._resource, 0)
        _check(err, "cudaGraphicsMapResources")
        try:
            err, array = cudart.cudaGraphicsSubResourceGetMappedArray(self._resource, 0, 0)
            _check(err, "cudaGraphicsSubResourceGetMappedArray")
            row_bytes = W * 4
            err, = cudart.cudaMemcpy2DToArray(
                array, 0, 0,
                rgba_uint8.data_ptr(), row_bytes,
                row_bytes, H,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            )
            _check(err, "cudaMemcpy2DToArray")
        finally:
            err, = cudart.cudaGraphicsUnmapResources(1, self._resource, 0)
            _check(err, "cudaGraphicsUnmapResources")

    def release(self) -> None:
        self._destroy()


# ---------------------------------------------------------------------------
# Blit shader
# ---------------------------------------------------------------------------


VERT_BLIT = """
#version 330
out vec2 v_uv;
const vec2 corners[4] = vec2[4](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);
void main() {
    vec2 p = corners[gl_VertexID];
    v_uv = 0.5 * (p + 1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

FRAG_BLIT = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_tex;
void main() {
    f_color = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
}
"""


# ---------------------------------------------------------------------------
# Turbo colormap LUT (256, 3) uint8.
# Polynomial fit from Google's turbo (mikhailov.cc). One-shot CPU build at
# startup is fine; lookup happens entirely on GPU.
# ---------------------------------------------------------------------------


def _build_turbo_lut(device: torch.device) -> torch.Tensor:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    # polynomial coefficients (from Google's turbo approximation, public domain)
    r = (0.13572138 + 4.61539260 * x - 42.66032258 * x ** 2
         + 132.13108234 * x ** 3 - 152.94239396 * x ** 4 + 59.28637943 * x ** 5)
    g = (0.09140261 + 2.19418839 * x + 4.84296658 * x ** 2
         - 14.18503333 * x ** 3 + 4.27729857 * x ** 4 + 2.82956604 * x ** 5)
    b = (0.10667330 + 12.64194608 * x - 60.58204836 * x ** 2
         + 110.36276771 * x ** 3 - 89.90310912 * x ** 4 + 27.34824973 * x ** 5)
    rgb = np.stack([r, g, b], axis=1).clip(0.0, 1.0)
    lut = (rgb * 255.0).astype(np.uint8)
    return torch.from_numpy(lut).to(device)  # (256, 3) uint8 on CUDA


# ---------------------------------------------------------------------------
# Render-time tonemap + depth visualization (all on CUDA).
# ---------------------------------------------------------------------------


def rgb_to_rgba_uint8(rgb_float: torch.Tensor) -> torch.Tensor:
    # rgb_float: (H, W, 3) float in ~[0,1]. Returns (H, W, 4) uint8 on CUDA.
    rgb = rgb_float.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    H, W, _ = rgb.shape
    a = torch.full((H, W, 1), 255, dtype=torch.uint8, device=rgb.device)
    return torch.cat([rgb, a], dim=-1).contiguous()


def depth_to_rgba_uint8(depth: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    # depth: (H, W) float, in scene units. lut: (256, 3) uint8.
    valid = depth > 0
    if valid.any():
        d_valid = depth[valid]
        d_min = d_valid.min()
        d_max = d_valid.max().clamp_min(d_min + 1e-6)
        norm = ((depth - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
    else:
        norm = torch.zeros_like(depth)
    idx = (norm * 255.0).to(torch.long).clamp(0, 255)
    rgb = lut[idx]  # (H, W, 3) uint8
    rgb = torch.where(valid.unsqueeze(-1), rgb, torch.zeros_like(rgb))
    H, W, _ = rgb.shape
    a = torch.full((H, W, 1), 255, dtype=torch.uint8, device=rgb.device)
    return torch.cat([rgb, a], dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------


RENDER_MODES = ["RGB", "Depth"]


class ViewerApp:
    def __init__(self, scene_path: str, flip: bool, width: int, height: int):
        self.scene_path = scene_path
        self.flip = flip
        self.window_w = int(width)
        self.window_h = int(height)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.device = torch.device("cuda")

        # GLFW window + GL context.
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.window = glfw.create_window(
            self.window_w, self.window_h, "Lyra-2 viewer", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()
        self.blit_prog = self.ctx.program(vertex_shader=VERT_BLIT, fragment_shader=FRAG_BLIT)
        self.blit_vao = self.ctx.vertex_array(self.blit_prog, [])

        # Scene + camera + controller.
        print(f"Loading {scene_path} ...", flush=True)
        t0 = time.perf_counter()
        self.scene = load_ply(scene_path, self.device, flip_to_y_up=flip)
        print(
            f"  {self.scene.means.shape[0]:,} splats, sh_degree={self.scene.sh_degree}, "
            f"loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

        self.camera = self._auto_fit_camera(self.scene)
        self.controller = CameraController(self.camera, self.window)

        # Initial render texture sized to current framebuffer.
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.fb_w, self.fb_h = fb_w, fb_h
        self.adaptive = True
        self.render_mode = 0  # index into RENDER_MODES
        self.sh_degree_ui = self.scene.sh_degree
        self.blitter = CudaGLBlitter(self.ctx, fb_w, fb_h)
        self._render_w, self._render_h = fb_w, fb_h

        self.turbo_lut = _build_turbo_lut(self.device)

        # ImGui.
        imgui.create_context()
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable.value
        self.imgui_impl = GlfwRenderer(self.window)

        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_size)

        self._last_t = time.perf_counter()
        self._smoothed_dt = 1.0 / 60.0
        self._last_resize_t = self._last_t

    @staticmethod
    def _auto_fit_camera(scene: GaussianScene) -> "FlyCamera":
        # Place the camera back along world -z from the scene centroid so the
        # default pose (yaw=0, looking +z) actually sees the scene. Use the
        # 5/95-percentile band for robustness against stray faraway splats
        # (3DGS scenes routinely have a few outliers tens of meters away).
        means = scene.means
        if means.shape[0] > 200_000:
            idx = torch.randint(0, means.shape[0], (200_000,), device=means.device)
            sample = means.index_select(0, idx)
        else:
            sample = means
        qs = torch.tensor([0.05, 0.5, 0.95], device=sample.device)
        q = torch.quantile(sample, qs, dim=0)  # (3, 3): rows = quantiles, cols = xyz
        median = q[1].tolist()
        z_lo = q[0, 2].item()
        z_hi = q[2, 2].item()
        z_span = max(z_hi - z_lo, 1.0)
        pos = (median[0], median[1], z_lo - 0.5 * z_span)
        return FlyCamera(position=pos, yaw=0.0, pitch=0.0, fov_deg=60.0)

    def _on_framebuffer_size(self, _window, w, h):
        if w <= 0 or h <= 0:
            return
        self.fb_w, self.fb_h = w, h
        self._last_resize_t = time.perf_counter()
        # The render-resolution recompute below in run() will resize the blitter.

    def _desired_render_size(self) -> tuple[int, int]:
        scale = 0.5 if (self.adaptive and self.camera.is_moving) else 1.0
        rw = max(2, int(round(self.fb_w * scale)))
        rh = max(2, int(round(self.fb_h * scale)))
        return rw, rh

    @torch.inference_mode()
    def _render_frame(self, render_w: int, render_h: int) -> torch.Tensor:
        # Build viewmat (1, 4, 4) and K (1, 3, 3) on CUDA.
        viewmat_np = self.camera.viewmat_w2c()
        K_np = self.camera.K_matrix(render_w, render_h)
        viewmats = torch.as_tensor(viewmat_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        Ks = torch.as_tensor(K_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        sh_degree = int(min(self.sh_degree_ui, self.scene.sh_degree))

        render_colors, _alphas, _info = rasterization(
            means=self.scene.means,
            quats=self.scene.quats,
            scales=self.scene.scales,
            opacities=self.scene.opacities,
            colors=self.scene.sh_coeffs,
            viewmats=viewmats,
            Ks=Ks,
            width=render_w,
            height=render_h,
            near_plane=0.01,
            far_plane=1000.0,
            packed=False,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
        )
        # render_colors: (1, H, W, 4) — RGB + expected depth.
        out = render_colors[0]
        if self.render_mode == 0:
            return rgb_to_rgba_uint8(out[..., :3])
        else:
            return depth_to_rgba_uint8(out[..., 3], self.turbo_lut)

    def _draw_panel(self) -> None:
        cam = self.camera
        scene = self.scene
        imgui.set_next_window_size((360, 380), imgui.Cond_.first_use_ever.value)
        imgui.begin("Viewer")

        fps = 1.0 / max(self._smoothed_dt, 1e-6)
        imgui.text(f"FPS: {fps:.1f}")
        imgui.text(f"Splats: {scene.means.shape[0]:,}")
        imgui.text(f"Render res: {self._render_w} x {self._render_h}")
        imgui.text(f"Window/FB: {self.fb_w} x {self.fb_h}")
        imgui.separator()

        changed, fov = imgui.slider_float("FOV", cam.fov_deg, 30.0, 110.0)
        if changed:
            cam.fov_deg = float(fov)

        log_speed = math.log10(max(cam.move_speed, 1e-3))
        changed, log_speed = imgui.slider_float("Move speed (log10)", log_speed, -1.0, math.log10(50.0))
        if changed:
            cam.move_speed = float(10.0 ** log_speed)
        imgui.text(f"  = {cam.move_speed:.3f} m/s")

        max_sh = scene.sh_degree
        items = [str(d) for d in range(0, max_sh + 1)]
        changed, idx = imgui.combo("SH degree", min(self.sh_degree_ui, max_sh), items)
        if changed:
            self.sh_degree_ui = int(idx)

        changed, idx = imgui.combo("Render mode", self.render_mode, RENDER_MODES)
        if changed:
            self.render_mode = int(idx)

        changed, val = imgui.checkbox("Adaptive resolution (0.5x while moving)", self.adaptive)
        if changed:
            self.adaptive = bool(val)

        if imgui.button("Reset camera"):
            cam.reset()

        imgui.separator()
        imgui.text(f"pos: ({cam.position[0]:+.3f}, {cam.position[1]:+.3f}, {cam.position[2]:+.3f})")
        imgui.text(f"yaw: {math.degrees(cam.yaw):+.2f}°  pitch: {math.degrees(cam.pitch):+.2f}°")
        imgui.text("RMB-drag = look | WASD = move | Space/Ctrl = up/down | Shift = boost")

        imgui.end()

    def run(self) -> int:
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.imgui_impl.process_inputs()

                now = time.perf_counter()
                dt = now - self._last_t
                self._last_t = now
                self._smoothed_dt = 0.9 * self._smoothed_dt + 0.1 * dt

                self.controller.update(dt)

                # Adapt render resolution.
                rw, rh = self._desired_render_size()
                if rw != self._render_w or rh != self._render_h:
                    self.blitter.resize(rw, rh)
                    self._render_w, self._render_h = rw, rh

                rgba = self._render_frame(self._render_w, self._render_h)
                self.blitter.upload(rgba)

                # Draw fullscreen quad sampled from the texture.
                self.ctx.viewport = (0, 0, self.fb_w, self.fb_h)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                self.blitter.texture.use(location=0)
                self.blit_prog["u_tex"].value = 0
                self.blit_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

                # ImGui.
                imgui.new_frame()
                self._draw_panel()
                imgui.render()
                self.imgui_impl.render(imgui.get_draw_data())

                glfw.swap_buffers(self.window)
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        try:
            self.blitter.release()
        except Exception:
            pass
        try:
            self.imgui_impl.shutdown()
        except Exception:
            pass
        glfw.destroy_window(self.window)
        glfw.terminate()


def main() -> int:
    ap = argparse.ArgumentParser(description="Lyra-2 native 3DGS viewer")
    ap.add_argument("path", help="Path to a 3DGS .ply file")
    ap.add_argument("--no-flip", action="store_true",
                    help="Skip the 180-degree X rotation applied on load (use if the .ply is already y-up)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    app = ViewerApp(
        scene_path=args.path,
        flip=not args.no_flip,
        width=args.width,
        height=args.height,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
