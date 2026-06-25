"""
Native OpenGL splat viewer — same gsplat rasterizer as viewer.py, but displayed
in a local GL window instead of a viser browser tab.

Why: viewer.py renders with gsplat on CUDA, copies the frame to a uint8 array,
and ships it to the browser as a viser *background image* over a websocket. That
browser/websocket hop (JPEG encode + network + three.js) is the latency
bottleneck, not the rasterizer. This script keeps the exact same
`Renderer.render()` path (so output is pixel-identical to training) and just
blits the frame straight to an OpenGL texture in a glfw window.

Architecture:
  - `Renderer.render()` only reads `.position`, `.wxyz`, `.fov` off its camera
    arg, so we feed it a duck-typed `NativeCamera` (no change to viewer.py).
  - `OrbitCamera` turns mouse/scroll input into (position, wxyz) in viser's
    OpenCV camera convention (look=+Z, up=-Y, right=+X), matching what
    `viser_camera_to_opencv_viewmat` expects.
  - Single-threaded loop: gsplat render -> upload uint8 to a GL texture ->
    draw a full-screen quad -> swap. Keeps torch(CUDA) and GL on one thread.

Run (from the repo root, in the `splat` env):
  conda run -n splat python visergui/native_viewer.py --ply splats.ply

Controls:
  left-drag   orbit            right-drag  pan
  scroll      zoom (dolly)     W / S       dolly in / out
  R           reset view       C           cycle RGB / Depth / Normals
  [ / ]       narrow/widen FOV ESC / Q     quit

Display note: needs a real OpenGL context (a desktop session). On a headless
box, run under xvfb-run or use an EGL-capable setup; this viewer targets the
local workstation.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make `import viewer` work no matter the CWD (the script lives in visergui/).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _ensure_cuda_headers() -> None:
    """Make sure gsplat's first-run CUDA JIT compile can find the toolkit headers.

    gsplat compiles its kernels lazily on the first `rasterization()` call. The
    host (c++) compile needs a *complete* CUDA include dir — `cuda_runtime.h`
    AND `crt/host_config.h`. On boxes where the active env borrows `nvcc` from a
    sibling conda env, torch's inferred include path can be incomplete, giving
    `fatal error: cuda_runtime.h: No such file or directory`. If CPATH doesn't
    already resolve those headers, prepend the first complete `targets/*/include`
    toolkit found among the conda envs. No-op once the kernels are cached.
    """
    cpath = os.environ.get("CPATH", "")
    has = lambda d: (os.path.exists(os.path.join(d, "crt", "host_config.h"))
                     and os.path.exists(os.path.join(d, "cuda_runtime.h")))
    if any(has(p) for p in cpath.split(os.pathsep) if p):
        return
    roots = []
    if os.environ.get("CONDA_PREFIX"):
        roots.append(os.path.dirname(os.environ["CONDA_PREFIX"]))
    roots += [os.path.expanduser("~/miniconda3/envs"),
              os.path.expanduser("~/anaconda3/envs")]
    for root in roots:
        for inc in sorted(glob.glob(os.path.join(root, "*", "targets", "*", "include"))):
            if has(inc):
                os.environ["CPATH"] = inc + (os.pathsep + cpath if cpath else "")
                return


_ensure_cuda_headers()

import glfw  # noqa: E402
import moderngl  # noqa: E402
import torch  # noqa: E402

from viewer import (  # noqa: E402
    Renderer,
    SceneState,
    _compute_home_pose,
    _rotmat_to_wxyz,
)


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


@dataclass
class NativeCamera:
    """Duck-types the bits of viser.CameraHandle that Renderer.render() reads."""

    position: np.ndarray  # (3,) world xyz
    wxyz: np.ndarray      # (4,) c2w rotation, viser/OpenCV frame
    fov: float            # vertical FOV in radians


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector `v` about unit-ish `axis` by `angle` radians."""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


class OrbitCamera:
    """Orbit/pan/zoom controller producing (position, wxyz, fov) for the renderer.

    `offset` is the vector from `target` to the camera; `position = target +
    offset`. World-up is the scene's up vector (-Y after the flip_x .ply
    transform). The c2w basis is built in OpenCV camera convention so the wxyz
    matches `viser_camera_to_opencv_viewmat`'s expectation.
    """

    def __init__(self, position, target, up, fov_rad: float) -> None:
        self.target = np.asarray(target, dtype=np.float64).reshape(3)
        self.up = np.asarray(up, dtype=np.float64).reshape(3)
        self.up /= max(float(np.linalg.norm(self.up)), 1e-12)
        self.offset = np.asarray(position, dtype=np.float64).reshape(3) - self.target
        self.fov = float(fov_rad)
        # Remember the home pose for reset.
        self._home = (self.target.copy(), self.offset.copy(), self.fov)

    # --- derived geometry --------------------------------------------------- #

    @property
    def position(self) -> np.ndarray:
        return self.target + self.offset

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.offset))

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (right, down, forward) world-space camera axes (OpenCV frame)."""
        fwd = -self.offset
        fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
        down0 = -self.up
        right = np.cross(down0, fwd)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        down = np.cross(fwd, right)  # orthonormal, right-handed [right|down|fwd]
        return right, down, fwd

    def camera(self) -> NativeCamera:
        right, down, fwd = self._basis()
        r_c2w = np.column_stack([right, down, fwd])  # columns = cam axes in world
        wxyz = _rotmat_to_wxyz(r_c2w)
        return NativeCamera(position=self.position, wxyz=wxyz, fov=self.fov)

    # --- interaction -------------------------------------------------------- #

    def orbit(self, dyaw: float, dpitch: float) -> None:
        # Yaw about world-up, then pitch about the current horizontal right axis.
        self.offset = _rodrigues(self.offset, self.up, dyaw)
        right = np.cross(self.up, self.offset)
        if float(np.linalg.norm(right)) > 1e-9:
            new_offset = _rodrigues(self.offset, right, dpitch)
            d = new_offset / max(float(np.linalg.norm(new_offset)), 1e-12)
            # Don't let the view cross the poles (offset parallel to up).
            if abs(float(np.dot(d, self.up))) < 0.999:
                self.offset = new_offset

    def pan(self, dx: float, dy: float) -> None:
        right, down, _fwd = self._basis()
        up_screen = -down
        k = self.distance * 0.0015
        # Drag grabs the scene: target slides opposite the cursor motion.
        self.target = self.target + (-dx * right + dy * up_screen) * k

    def dolly(self, steps: float) -> None:
        self.offset = self.offset * float(1.1 ** (-steps))
        if self.distance < 1e-3:
            self.offset = self.offset / max(self.distance, 1e-9) * 1e-3

    def adjust_fov(self, d_deg: float) -> None:
        deg = math.degrees(self.fov) + d_deg
        self.fov = math.radians(min(120.0, max(5.0, deg)))

    def reset(self) -> None:
        target, offset, fov = self._home
        self.target = target.copy()
        self.offset = offset.copy()
        self.fov = fov


# --------------------------------------------------------------------------- #
# GL display
# --------------------------------------------------------------------------- #


_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(tex, v_uv);
}
"""


class QuadBlitter:
    """Uploads a uint8 HxWx3 frame to a texture and draws it full-screen.

    The texture is sized to whatever frame arrives (adaptive_res shrinks it
    while the camera moves), and the quad upscales it with linear filtering.
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        # Full-screen triangle strip: top-of-screen samples v=0 (frame row 0).
        verts = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 0.0,
                 1.0,  1.0, 1.0, 0.0,
            ],
            dtype="f4",
        )
        self.vbo = ctx.buffer(verts.tobytes())
        self.vao = ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self.tex: moderngl.Texture | None = None
        self._shape: tuple[int, int] | None = None

    def draw(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        if self._shape != (h, w):
            if self.tex is not None:
                self.tex.release()
            self.tex = self.ctx.texture((w, h), 3, dtype="f1")
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._shape = (h, w)
        self.tex.write(np.ascontiguousarray(frame).tobytes())
        self.tex.use(0)
        self.prog["tex"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


_COLOR_MODES = ("RGB", "Depth", "Normals")


class NativeViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)

        print(f"[native_viewer] loading {args.ply} ...")
        self.scene = SceneState()
        self.scene.load_from_ply(args.ply, self.device, flip_x=not args.no_flip_x)
        print(f"[native_viewer] {self.scene.num_splats:,} splats, "
              f"sh_degree={self.scene.sh_degree}")

        self.renderer = Renderer(device=args.device)

        with self.scene.read() as s:
            means_np = s.means.detach().cpu().numpy()
        pos, look_at, up = _compute_home_pose(means_np)
        self.cam = OrbitCamera(pos, look_at, up, math.radians(args.fov))

        self.color_idx = 0
        self.near = args.near
        self.far = args.far

        # Mouse state.
        self._last_x = 0.0
        self._last_y = 0.0
        self._left = False
        self._right = False

        self._init_window()

    # --- window / GL -------------------------------------------------------- #

    def _init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed — is a display available?")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.win = glfw.create_window(
            self.args.width, self.args.height, "Lyra native splat viewer", None, None
        )
        if not self.win:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed — no GL context?")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1 if self.args.vsync else 0)

        self.ctx = moderngl.create_context()
        self.blitter = QuadBlitter(self.ctx)

        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_mouse_button_callback(self.win, self._on_button)
        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_key_callback(self.win, self._on_key)

    # --- input callbacks ---------------------------------------------------- #

    def _on_cursor(self, _win, x, y) -> None:
        dx, dy = x - self._last_x, y - self._last_y
        self._last_x, self._last_y = x, y
        if self._left:
            self.cam.orbit(-dx * 0.005, -dy * 0.005)
        elif self._right:
            self.cam.pan(dx, dy)

    def _on_button(self, _win, button, action, _mods) -> None:
        down = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left = down
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right = down
        if down:
            self._last_x, self._last_y = glfw.get_cursor_pos(_win)

    def _on_scroll(self, _win, _xoff, yoff) -> None:
        self.cam.dolly(float(yoff))

    def _on_key(self, win, key, _sc, action, _mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_R:
            self.cam.reset()
        elif key == glfw.KEY_C and action == glfw.PRESS:
            self.color_idx = (self.color_idx + 1) % len(_COLOR_MODES)
            print(f"[native_viewer] color mode: {_COLOR_MODES[self.color_idx]}")
        elif key == glfw.KEY_W:
            self.cam.dolly(1.0)
        elif key == glfw.KEY_S:
            self.cam.dolly(-1.0)
        elif key == glfw.KEY_LEFT_BRACKET:
            self.cam.adjust_fov(-2.0)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.cam.adjust_fov(+2.0)

    # --- main loop ---------------------------------------------------------- #

    def run(self) -> None:
        last_title = time.perf_counter()
        frames = 0
        render_ms = 0.0
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            fb_w, fb_h = glfw.get_framebuffer_size(self.win)
            if fb_w == 0 or fb_h == 0:  # minimized
                glfw.wait_events_timeout(0.1)
                continue

            cam = self.cam.camera()
            img, render_ms, _moving = self.renderer.render(
                self.scene,
                cam,
                fb_w,
                fb_h,
                sh_degree=self.scene.sh_degree,
                color_mode=_COLOR_MODES[self.color_idx],
                near=self.near,
                far=self.far,
                adaptive_res=self.args.adaptive_res,
                moving_scale=self.args.moving_scale,
            )

            self.ctx.viewport = (0, 0, fb_w, fb_h)
            self.ctx.clear(0.0, 0.0, 0.0)
            self.blitter.draw(img)
            glfw.swap_buffers(self.win)

            frames += 1
            now = time.perf_counter()
            if now - last_title >= 0.5:
                fps = frames / (now - last_title)
                glfw.set_window_title(
                    self.win,
                    f"Lyra native splat viewer — {fps:5.1f} fps | "
                    f"render {render_ms:4.1f} ms | {_COLOR_MODES[self.color_idx]} | "
                    f"{self.scene.num_splats:,} splats",
                )
                last_title = now
                frames = 0

        glfw.terminate()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = _HERE.parent
    p = argparse.ArgumentParser(description="Native OpenGL Gaussian-splat viewer.")
    p.add_argument("--ply", type=Path, default=repo_root / "splats.ply",
                   help="Path to an Inria-style 3DGS .ply (default: ./splats.ply).")
    p.add_argument("--width", type=int, default=1280, help="Initial window width.")
    p.add_argument("--height", type=int, default=720, help="Initial window height.")
    p.add_argument("--device", type=str, default="cuda", help="torch device.")
    p.add_argument("--fov", type=float, default=60.0,
                   help="Initial vertical FOV in degrees.")
    p.add_argument("--near", type=float, default=0.1, help="Depth-mode near plane.")
    p.add_argument("--far", type=float, default=100.0, help="Depth-mode far plane.")
    p.add_argument("--no-flip-x", action="store_true",
                   help="Disable the Inria x-flip applied on load.")
    p.add_argument("--no-adaptive-res", dest="adaptive_res", action="store_false",
                   help="Always render at full resolution (no down-res while moving).")
    p.add_argument("--moving-scale", type=float, default=0.5,
                   help="Resolution scale applied while the camera moves.")
    p.add_argument("--vsync", action="store_true", help="Cap to display refresh.")
    p.set_defaults(adaptive_res=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not Path(args.ply).exists():
        raise SystemExit(f"ply not found: {args.ply}")
    NativeViewer(args).run()


if __name__ == "__main__":
    main()
