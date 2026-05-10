"""
Training-side seam for the splat viewer. Stdlib-only.

This module is deliberately import-light: a headless training script can
import `BackgroundTrainingThread` (or the `TrainingControl` Protocol)
without pulling in viser, gsplat, torch, plotly, or anything else.

The viewer (`visergui/viewer.py`) imports only `TrainingControl` from this
module — and only as a type hint, since it's a Protocol satisfied
structurally. Anything with start/pause/resume/stop/status methods works.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class TrainingControl(Protocol):
    """Five-method contract for anything driving a training loop the viewer
    should be able to pause / resume / stop. Structural — no inheritance
    required."""

    def start(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> str: ...   # "running" | "paused" | "stopped"


class BackgroundTrainingThread:
    """Convenience driver: runs `step_fn` in a daemon thread until stopped.

    `step_fn` is whatever the caller wants — a method, a closure, anything
    callable. It receives no arguments, and its return value is ignored
    here (loss reporting goes through SceneState.record_step in the
    GUI-attached case).

    Pause/resume use `threading.Event`; pause is cheap (sleeps 5 ms in a
    loop). Auto-pauses on exception so a bad batch doesn't crash the
    process — the traceback is printed and the loop sleeps until the user
    resumes manually.

    Required only when training shares the main thread with something else
    (e.g. the viser server). Headless training scripts can call `step_fn`
    in a plain loop and skip this entirely.
    """

    def __init__(self, step_fn: Callable[[], object]) -> None:
        self._step_fn = step_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # start paused so caller controls the first step

    # ---- TrainingControl methods ------------------------------------- #

    def start(self) -> None:
        """Spin up the worker thread (idempotent). Starts paused."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        """Signal stop; join the thread with a 5 s timeout."""
        self._stop.set()
        self._paused.clear()  # let the loop wake to see the stop flag
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)

    def status(self) -> str:
        if self._thread is None or not self._thread.is_alive():
            return "stopped"
        if self._paused.is_set():
            return "paused"
        return "running"

    # ---- Loop body --------------------------------------------------- #

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.005)
                continue
            try:
                self._step_fn()
            except Exception:
                traceback.print_exc()
                self._paused.set()  # auto-pause so the user can investigate
