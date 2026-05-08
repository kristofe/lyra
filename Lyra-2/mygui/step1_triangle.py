"""Step 1: GLFW + moderngl + imgui-bundle scaffold.

Opens a window, draws a colored triangle, shows a dockable ImGui panel with
an FPS readout. Proves the windowing + GL + ImGui stack is wired correctly
before we layer CUDA-GL interop on top.

    python step1_triangle.py
"""

from __future__ import annotations

import time

import glfw
import moderngl
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer


WINDOW_W, WINDOW_H = 1280, 720
TITLE = "Lyra-2 mygui — step1: triangle"


VERT = """
#version 330
in vec2 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


def main() -> int:
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
    program = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    import numpy as np

    verts = np.array(
        [
            #  x,    y,   r,   g,   b
            -0.6, -0.5, 1.0, 0.2, 0.2,
             0.6, -0.5, 0.2, 1.0, 0.2,
             0.0,  0.6, 0.2, 0.4, 1.0,
        ],
        dtype="f4",
    )
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f 3f", "in_pos", "in_color")])

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

            fb_w, fb_h = glfw.get_framebuffer_size(window)
            ctx.viewport = (0, 0, fb_w, fb_h)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.clear(0.07, 0.08, 0.10, 1.0)
            vao.render(moderngl.TRIANGLES)

            imgui.new_frame()
            imgui.set_next_window_size((300, 140), imgui.Cond_.first_use_ever.value)
            imgui.begin("Step 1")
            imgui.text(f"FPS: {1.0 / max(smoothed_dt, 1e-6):.1f}")
            imgui.text(f"Framebuffer: {fb_w} x {fb_h}")
            imgui.text("If you see a triangle and this panel, the stack is good.")
            imgui.end()
            imgui.render()
            impl.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
    finally:
        impl.shutdown()
        glfw.destroy_window(window)
        glfw.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
