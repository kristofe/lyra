"""Swappable client for the video-generation REST server.

The Demo tab (viewer.py) sends an image + prompt to a generation server and
gets a video back. The server call is *synchronous*: one HTTP request blocks
for ~30 s and the response carries the video.

The exact request/response contract for the real server is not finalized, so
this module isolates it in one small function. When the real spec lands, edit
`request_video` only — the GUI and orchestration in viewer.py / train_and_view.py
don't change.

ASSUMED CONTRACT (edit here if the real server differs):
  POST <server_url>
    multipart/form-data:
      image  = (filename, raw image bytes)   # the conditioning frame
      prompt = <text>                         # the generation prompt
  Response (either form is accepted):
    - Content-Type video/* or application/octet-stream → body IS the video.
    - Content-Type application/json → {"video_url": "..."} (or "url"/"video"):
      a follow-up GET fetches the actual video bytes.
"""

from __future__ import annotations

import requests

# Response JSON keys checked, in order, for a video download URL.
_URL_KEYS = ("video_url", "url", "video", "output_url", "result_url")


def request_video(
    server_url: str,
    image_bytes: bytes,
    image_name: str,
    prompt: str,
    timeout: float = 180.0,
) -> bytes:
    """POST image+prompt to the generation server and return raw video bytes.

    Blocks until the server responds (the generation itself takes ~30 s, hence
    the generous default timeout). Raises on a non-2xx status, an unexpected
    content type, or an empty body so the caller can surface the failure.
    """
    if not server_url:
        raise ValueError("server URL is empty")
    if not image_bytes:
        raise ValueError("no image bytes to send")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is empty")

    resp = requests.post(
        server_url,
        files={"image": (image_name or "image.png", image_bytes)},
        data={"prompt": prompt},
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
