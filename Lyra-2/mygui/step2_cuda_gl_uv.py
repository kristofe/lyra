"""Step 2: CUDA-GL interop with a synthetic torch UV-gradient tensor.

This is the regression harness for the zero-CPU-roundtrip blit. It builds a
``(H, W, 4) uint8`` torch tensor on CUDA each frame (a UV gradient with a
pulsing time component) and copies it directly into a GL texture via
``cudaGraphicsGLRegisterImage`` + ``cudaMemcpy2DToArray``. A fullscreen
quad samples the texture and draws it to the window.

If the final viewer ever stops working, run this first to isolate whether
the breakage is in the gsplat/camera path or in interop.

    python step2_cuda_gl_uv.py
"""

from __future__ import annotations

import ctypes
import time

import glfw
import moderngl
import torch
from cuda.bindings import runtime as cudart
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer


WINDOW_W, WINDOW_H = 1280, 720
TITLE = "Lyra-2 mygui — step2: cuda-gl uv gradient"

GL_TEXTURE_2D = 0x0DE1


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


def _check(err, where: str):
    if isinstance(err, tuple):
        err, *rest = err
    if int(err) != 0:
        raise RuntimeError(f"CUDA error at {where}: {err}")


class CudaGLBlitter:
    """Owns a GL texture and a CUDA registration of it.

    ``upload(tensor)`` copies an ``(H, W, 4) uint8`` CUDA tensor straight into
    the texture via cudaMemcpy2DToArray, no host roundtrip.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self._resource = None
        self.texture = None
        self.resize(width, height)

    def _create(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.texture = self.ctx.texture((self.width, self.height), 4, dtype="f1")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        # Register the GL texture with CUDA. WriteDiscard tells the driver
        # we don't read prior contents — it can drop stale data.
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
        self._destroy()
        self._create(width, height)

    def upload(self, rgba_uint8: torch.Tensor) -> None:
        # rgba_uint8: (H, W, 4) uint8, contiguous, on CUDA.
        assert rgba_uint8.is_cuda and rgba_uint8.dtype == torch.uint8
        assert rgba_uint8.is_contiguous()
        H, W, C = rgba_uint8.shape
        assert C == 4 and H == self.height and W == self.width

        err, = cudart.cudaGraphicsMapResources(1, self._resource, 0)
        _check(err, "cudaGraphicsMapResources")
        try:
            err, array = cudart.cudaGraphicsSubResourceGetMappedArray(
                self._resource, 0, 0
            )
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


def make_uv_tensor(W: int, H: int, t: float, device: torch.device) -> torch.Tensor:
    # Build (H, W, 4) uint8 RGBA UV gradient, pulsing in B with sin(t).
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gx * 255.0).to(torch.uint8)
    g = (gy * 255.0).to(torch.uint8)
    b = ((0.5 + 0.5 * torch.sin(torch.tensor(t * 2.0, device=device))) * 255.0).to(torch.uint8)
    b = b.expand(H, W)
    a = torch.full((H, W), 255, dtype=torch.uint8, device=device)
    return torch.stack([r, g, b, a], dim=-1).contiguous()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — cuda-gl interop needs a CUDA device")
    device = torch.device("cuda")

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WINDOW_W, WINDOW_H, TITLE, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    blit_prog = ctx.program(vertex_shader=VERT_BLIT, fragment_shader=FRAG_BLIT)
    blit_vao = ctx.vertex_array(blit_prog, [])

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    blitter = CudaGLBlitter(ctx, fb_w, fb_h)

    def on_resize(_w, _h):
        new_w, new_h = glfw.get_framebuffer_size(window)
        if new_w > 0 and new_h > 0:
            ctx.viewport = (0, 0, new_w, new_h)
            blitter.resize(new_w, new_h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    imgui.create_context()
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.docking_enable.value
    impl = GlfwRenderer(window)

    last_t = time.perf_counter()
    smoothed_dt = 1.0 / 60.0

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            smoothed_dt = 0.9 * smoothed_dt + 0.1 * dt

            tex = make_uv_tensor(blitter.width, blitter.height, now, device)
            blitter.upload(tex)

            ctx.disable(moderngl.DEPTH_TEST)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            blitter.texture.use(location=0)
            blit_prog["u_tex"].value = 0
            blit_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

            imgui.new_frame()
            imgui.set_next_window_size((320, 140), imgui.Cond_.first_use_ever.value)
            imgui.begin("Step 2")
            imgui.text(f"FPS: {1.0 / max(smoothed_dt, 1e-6):.1f}")
            imgui.text(f"Texture: {blitter.width} x {blitter.height}")
            imgui.text("UV gradient is generated on CUDA, blitted via interop.")
            imgui.end()
            imgui.render()
            impl.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
    finally:
        blitter.release()
        impl.shutdown()
        glfw.destroy_window(window)
        glfw.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
