"""Tiny imgui_bundle GUI: text box + file picker + Run button.

Hack as needed: replace the body of run_command() with whatever you actually
want to launch.

    python runner.py
"""

import subprocess

from imgui_bundle import imgui, immapp
from imgui_bundle import portable_file_dialogs as pfd


# Module-level state. Lists used as mutable cells so the gui() closure can
# reassign without `global`.
text = [""]
file_path = [""]
output = [""]
pending_dialog = [None]  # holds an open pfd.open_file() while user picks


def run_command() -> None:
    cmd = [
        "python",
        "-c",
        f"print('text:', {text[0]!r}); print('file:', {file_path[0]!r})",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output[0] = result.stdout + result.stderr


def gui() -> None:
    # Text box.
    _, text[0] = imgui.input_text("text", text[0])

    # File picker row: Browse button + chosen path.
    if imgui.button("Browse..."):
        if pending_dialog[0] is None:
            pending_dialog[0] = pfd.open_file("Pick a file")
    imgui.same_line()
    imgui.text(file_path[0] or "(no file)")

    # Poll the dialog so the UI doesn't block.
    if pending_dialog[0] is not None and pending_dialog[0].ready():
        chosen = pending_dialog[0].result()
        if chosen:
            file_path[0] = chosen[0]
        pending_dialog[0] = None

    # Run button.
    if imgui.button("Run"):
        run_command()

    # Output.
    imgui.text("output:")
    imgui.input_text_multiline(
        "##output",
        output[0],
        (-1, 200),
        flags=imgui.InputTextFlags_.read_only.value,
    )


if __name__ == "__main__":
    immapp.run(gui_function=gui, window_title="Runner", window_size=(640, 400))
