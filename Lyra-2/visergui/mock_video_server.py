"""Mock video-generation server for debugging the Demo tab.

Stands in for the real (synchronous, ~30 s) generation server: it accepts the
same multipart POST the Demo tab sends (`image` file + `prompt` text) and
replies with a real `.mp4` from a directory — handing them out in order and
cycling back to the start. That lets you exercise the whole
first-video → prepare_and_init, second-video → append_video flow with known
clips and no GPU model.

Matches the contract in visergui/video_api.py: response Content-Type is
`video/mp4` and the body IS the video.

Usage:
    python visergui/mock_video_server.py            # serves assets/ours/*.mp4 on :8000
    python visergui/mock_video_server.py --dir my_clips --port 9000 --delay 3

Then point the Demo tab's "server URL" at  http://localhost:8000/generate
(the default), or launch the app with --demo-server-url to match. Any path is
accepted, so the exact URL suffix doesn't matter.

Stdlib only — no Flask/requests needed.
"""

from __future__ import annotations

import argparse
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import cycle
from pathlib import Path

# Best-effort extraction of the `prompt` text field for nicer logging. Never
# fatal — the server returns a video regardless of what (or whether) it parses.
_PROMPT_RE = re.compile(
    rb'name="prompt"\r?\n\r?\n(.*?)\r?\n--', re.DOTALL,
)


def _make_handler(videos: list[Path], delay: float):
    clips = cycle(videos)  # endless, wraps back to the first after the last
    state = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter default access log
            pass

        def _serve_next(self):
            # Drain the request body (so the client isn't left mid-write) and
            # try to surface the prompt for the console.
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            m = _PROMPT_RE.search(body)
            prompt = m.group(1).decode("utf-8", "replace").strip() if m else "(no prompt parsed)"

            clip = next(clips)
            state["n"] += 1
            data = clip.read_bytes()
            print(
                f"[mock] request #{state['n']}: prompt={prompt!r}  "
                f"({length:,} bytes in) → {clip.name} ({len(data):,} bytes out)",
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)  # simulate generation latency

            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                self._serve_next()
            except Exception as e:  # report instead of dropping the connection
                msg = f"mock server error: {type(e).__name__}: {e}".encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def do_GET(self):
            # Health check / browser sanity ping.
            names = ", ".join(v.name for v in videos)
            msg = (
                f"mock video server up. {len(videos)} clip(s) in rotation: "
                f"{names}. POST an image+prompt to get the next one."
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    return Handler


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Mock video-generation server for the Demo tab.")
    p.add_argument(
        "--dir", type=Path, default=repo_root / "assets" / "ours",
        help="Directory of .mp4 clips to hand out in sorted order (default: assets/ours).",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to sleep before responding, to mimic generation latency "
             "(default 0.5; set higher to test the 'requesting…' UI state).",
    )
    args = p.parse_args()

    video_dir = args.dir.expanduser()
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 files found in {video_dir}")

    print(f"[mock] serving {len(videos)} clip(s) from {video_dir} in this order:")
    for i, v in enumerate(videos):
        print(f"         {i}: {v.name}")
    print(f"[mock] listening on http://{args.host}:{args.port}  "
          f"(point Demo→'server URL' here, e.g. http://localhost:{args.port}/generate)")

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(videos, args.delay))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
