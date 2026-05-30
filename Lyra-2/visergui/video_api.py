"""Swappable client for the video-generation REST server.

The Demo tab (viewer.py) sends an image + prompt to a generation server and
gets a video back. The server call is *synchronous*: one HTTP request blocks
for the duration of generation and the response carries the video.

This targets the Lyra2 ``demo_server`` (lyra_2/_src/inference/demo_server.py):
    POST <server_url>/generate   multipart/form-data
        image                 = (filename, raw image bytes)
        prompt                = <text>            # optional, generic fallback
        resolution            = "480p" | "H,W"    # preset label or raw H,W
        num_frames_zoom_in    = 1 + 80k           # 81, 161, 241, ...
        num_frames_zoom_out   = 1 + 80k
        zoom_in_strength      = float
        zoom_out_strength     = float
    → response Content-Type video/mp4 : body IS the mp4.
    GET  <server_url>/resolutions → {"default": "480p", "presets": [...]}

The exact contract lives in this one module; the GUI / orchestration in
viewer.py / train_and_view.py call through ``request_video`` /
``fetch_resolutions`` and don't depend on the wire format.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import requests

# Response JSON keys checked, in order, for a video download URL.
_URL_KEYS = ("video_url", "url", "video", "output_url", "result_url")

# Static fallback presets (mirror demo_server.RESOLUTION_PRESETS) used when the
# server can't be reached at GUI-build time. Labels map to "H,W" the server
# also accepts directly.
DEFAULT_RESOLUTION_PRESETS = ("480p", "360p", "320p", "240p")
DEFAULT_RESOLUTION = "480p"


def _sibling_url(server_url: str, path: str) -> str:
    """Return ``server_url`` with its final path segment replaced by ``path``.

    e.g. ('http://h:8080/generate', 'resolutions') -> 'http://h:8080/resolutions'.
    """
    parts = urlsplit(server_url)
    base = parts.path.rsplit("/", 1)[0]
    new_path = f"{base}/{path}".replace("//", "/")
    return urlunsplit((parts.scheme, parts.netloc, new_path, "", ""))


def fetch_resolutions(server_url: str, timeout: float = 5.0):
    """GET <server>/resolutions → (preset_labels tuple, default_label).

    Falls back to the static presets if the server is unreachable or returns an
    unexpected payload, so the GUI can always build a dropdown.
    """
    try:
        resp = requests.get(_sibling_url(server_url, "resolutions"), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        presets = payload.get("presets") or []
        labels = tuple(p["label"] for p in presets if "label" in p)
        default = payload.get("default") or (labels[0] if labels else DEFAULT_RESOLUTION)
        if labels:
            return labels, default
    except Exception:
        pass
    return DEFAULT_RESOLUTION_PRESETS, DEFAULT_RESOLUTION


def request_video(
    server_url: str,
    image_bytes: bytes,
    image_name: str,
    prompt: str,
    gen_opts: dict | None = None,
    timeout: float = 600.0,
) -> bytes:
    """POST image+prompt(+zoom/resolution opts) and return raw video bytes.

    Blocks until the server responds (generation can take from ~30 s with DMD
    up to several minutes at full resolution, hence the generous default
    timeout). ``gen_opts`` carries the optional Lyra2 zoom controls
    (``resolution``, ``num_frames_zoom_in``/``out``, ``zoom_in``/``out_strength``);
    only keys with a non-None value are sent. Raises on a non-2xx status, an
    unexpected content type, or an empty body so the caller can surface it.
    """
    if not server_url:
        raise ValueError("server URL is empty")
    if not image_bytes:
        raise ValueError("no image bytes to send")

    # prompt is optional on the server (generic fallback); only send if given.
    data: dict[str, object] = {}
    if prompt and prompt.strip():
        data["prompt"] = prompt
    for key in (
        "resolution",
        "num_frames_zoom_in",
        "num_frames_zoom_out",
        "zoom_in_strength",
        "zoom_out_strength",
    ):
        val = (gen_opts or {}).get(key)
        if val is not None and val != "":
            data[key] = val

    resp = requests.post(
        server_url,
        files={"image": (image_name or "image.png", image_bytes)},
        data=data,
        timeout=timeout,
    )
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()

    # JSON envelope → follow the embedded download URL.
    if "application/json" in content_type:
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"server returned JSON but not an object "
                f"(got {type(payload).__name__}); expected "
                f"{{'video_url': ...}}"
            )
        url = next((payload[k] for k in _URL_KEYS if payload.get(k)), None)
        if not url:
            raise ValueError(
                f"server returned JSON without a video URL "
                f"(keys tried: {_URL_KEYS}); got {list(payload)!r}"
            )
        dl = requests.get(url, timeout=timeout)
        dl.raise_for_status()
        video_bytes = dl.content
    elif (content_type.startswith("video/")
          or "application/octet-stream" in content_type
          or not content_type):
        # video/* or application/octet-stream (or a server that omits the
        # header) → the body is the video itself.
        video_bytes = resp.content
    else:
        # text/html error pages, login redirects, queue/status pages, etc.
        # all pass raise_for_status with a 2xx; refuse them here rather than
        # hand the caller a "video" that's really an HTML blob.
        raise ValueError(
            f"server returned non-video content (Content-Type={content_type!r}); "
            f"expected video/*, application/octet-stream, or application/json"
        )

    if not video_bytes:
        raise ValueError("server returned an empty video body")
    return video_bytes
