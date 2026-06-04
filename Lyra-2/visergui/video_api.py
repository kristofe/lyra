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
RESOLUTION_PRESETS = {
    "480p": (480, 832),
    "360p": (368, 640),
    "320p": (320, 576),
    "240p": (256, 448),
}
DEFAULT_RESOLUTION_PRESETS = tuple(RESOLUTION_PRESETS)
DEFAULT_RESOLUTION = "240p"


def resolve_resolution(value: str) -> str:
    """Map a preset label ('240p') or raw 'H,W'/'HxW' to a validated 'H,W' string.

    Used by backends (e.g. the sequence server) that need explicit H,W rather than a
    label. Unknown labels fall back to the default preset.
    """
    v = str(value).strip()
    if v in RESOLUTION_PRESETS:
        h, w = RESOLUTION_PRESETS[v]
        return f"{h},{w}"
    parts = v.replace("x", ",").split(",")
    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
        return f"{int(parts[0])},{int(parts[1])}"
    h, w = RESOLUTION_PRESETS[DEFAULT_RESOLUTION]
    return f"{h},{w}"


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


# Static fallback camera options (mirror demo_server) for when /trajectories
# isn't reachable when the GUI builds the tab.
DEFAULT_TRAJECTORIES = (
    "horizontal_zoom", "horizontal", "horizontal_lift", "orbit_horizontal",
    "orbit_vertical", "spiral", "dolly_zoom", "rotate_spot", "back", "original",
)
DEFAULT_DIRECTIONS = ("left", "right", "up", "down")
DEFAULT_TRAJECTORY = "horizontal_zoom"
DEFAULT_DIRECTION = "right"


def fetch_trajectories(server_url: str, timeout: float = 5.0):
    """GET <server>/trajectories → (trajectories, directions, default_traj, default_dir).

    Falls back to the static lists if the server is unreachable, so the GUI can
    always build the dropdowns.
    """
    try:
        resp = requests.get(_sibling_url(server_url, "trajectories"), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        trajs = tuple(payload.get("trajectories") or ())
        dirs = tuple(payload.get("directions") or ())
        dtraj = payload.get("default") or (trajs[0] if trajs else DEFAULT_TRAJECTORY)
        ddir = payload.get("default_direction") or DEFAULT_DIRECTION
        if trajs and dirs:
            return trajs, dirs, dtraj, ddir
    except Exception:
        pass
    return DEFAULT_TRAJECTORIES, DEFAULT_DIRECTIONS, DEFAULT_TRAJECTORY, DEFAULT_DIRECTION


def request_video(
    server_url: str,
    image_bytes: bytes,
    image_name: str,
    prompt: str,
    gen_opts: dict | None = None,
    timeout: float = 600.0,
) -> bytes:
    """POST image+prompt(+trajectory/resolution opts) and return raw video bytes.

    Blocks until the server responds (generation can take from ~tens of seconds
    with DMD up to minutes at full resolution, hence the generous default
    timeout). ``gen_opts`` carries the single-trajectory Lyra2 controls
    (``resolution``, ``trajectory``, ``direction``, ``num_frames``, ``strength``);
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
        "trajectory",
        "direction",
        "num_frames",
        "strength",
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


def _extract_video_bytes(resp, timeout: float) -> bytes:
    """Turn a /generate(_custom) response into raw mp4 bytes.

    Accepts a video/* (or octet-stream) body directly, or a JSON envelope with a
    download URL. Raises on non-2xx, unexpected content type, or empty body.
    """
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    if "application/json" in content_type:
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"server returned JSON but not an object "
                f"(got {type(payload).__name__}); expected {{'video_url': ...}}"
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
        video_bytes = resp.content
    else:
        raise ValueError(
            f"server returned non-video content (Content-Type={content_type!r}); "
            f"expected video/*, application/octet-stream, or application/json"
        )
    if not video_bytes:
        raise ValueError("server returned an empty video body")
    return video_bytes


def request_custom_video(
    server_url: str,
    image_bytes: bytes,
    image_name: str,
    trajectory_npz_bytes: bytes,
    multiview_npz_bytes: bytes | None = None,
    prompt: str = "",
    resolution: str | None = None,
    pose_scale: float = 1.0,
    timeout: float = 1200.0,
) -> bytes:
    """POST a seed image + a caller-authored trajectory npz (and optional multiview
    anchors npz) to ``/generate_custom`` and return raw mp4 bytes.

    ``server_url`` is the base /generate endpoint; this swaps the final path segment
    to ``/generate_custom``. Used by the gap-fill workflow (see nbv_trajectory.py).
    """
    if not server_url:
        raise ValueError("server URL is empty")
    if not image_bytes:
        raise ValueError("no image bytes to send")
    if not trajectory_npz_bytes:
        raise ValueError("no trajectory npz to send")

    url = _sibling_url(server_url, "generate_custom")
    files = {
        "image": (image_name or "image.png", image_bytes),
        "trajectory_npz": ("trajectory.npz", trajectory_npz_bytes,
                           "application/octet-stream"),
    }
    if multiview_npz_bytes:
        files["multiview_npz"] = ("multiview.npz", multiview_npz_bytes,
                                  "application/octet-stream")
    data: dict[str, object] = {"pose_scale": pose_scale}
    if prompt and prompt.strip():
        data["prompt"] = prompt
    if resolution:
        data["resolution"] = resolution

    resp = requests.post(url, files=files, data=data, timeout=timeout)
    return _extract_video_bytes(resp, timeout)


def _sequence_endpoint(server_url: str) -> str:
    """Return the collaborator's `.../sequence/generate` endpoint.

    Accepts either the full endpoint or a base URL (with/without trailing slash).
    """
    u = server_url.rstrip("/")
    if u.endswith("/sequence/generate"):
        return u
    return u + "/sequence/generate"


def request_sequence_video(
    server_url: str,
    *,
    token: str,
    trajectory_npz_bytes: bytes,
    resolution: str,
    num_frames: int,
    prompt: str = "",
    image_bytes: bytes | None = None,
    image_name: str = "image.png",
    session_id: str | None = None,
    timeout: float = 600.0,
) -> tuple[bytes, str | None]:
    """POST to a collaborator's session-based `/sequence/generate` server.

    First call (``session_id is None``) sends ``image`` + ``trajectory`` and the
    server returns an ``X-Session-Id`` header. Subsequent calls send ``session_id`` +
    ``trajectory`` (no image) to continue that server-side session. Auth is a Bearer
    token. The trajectory is an npz in the lyra custom-traj schema (w2c/intrinsics/
    image_height/image_width).

    Returns ``(video_bytes, session_id)`` — the session id is read from the response's
    ``X-Session-Id`` header (falls back to the one passed in).
    """
    if not server_url:
        raise ValueError("sequence server URL is empty")
    if not token:
        raise ValueError("sequence server requires a Bearer token")
    if not trajectory_npz_bytes:
        raise ValueError("no trajectory npz to send")
    if session_id is None and not image_bytes:
        raise ValueError("first sequence call requires an image")

    url = _sequence_endpoint(server_url)
    headers = {"Authorization": f"Bearer {token}"}
    files = {
        "trajectory": ("trajectory.npz", trajectory_npz_bytes,
                       "application/octet-stream"),
    }
    data: dict[str, object] = {"resolution": resolution, "num_frames": int(num_frames)}
    if prompt and prompt.strip():
        data["prompt"] = prompt
    if session_id is None:
        files["image"] = (image_name or "image.png", image_bytes)
    else:
        data["session_id"] = session_id

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
    video_bytes = _extract_video_bytes(resp, timeout)
    new_sid = resp.headers.get("X-Session-Id") or session_id
    if new_sid:
        new_sid = new_sid.strip()
    return video_bytes, new_sid
