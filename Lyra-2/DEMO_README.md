# Lyra 2.0 Demo: image → zoom video, server + GUI

Turn a single image into a Lyra2 **zoom-in / zoom-out exploration video** through a
small REST server, then (optionally) drive it from the viser GUI's **Demo tab** —
upload an image, pick zoom/resolution options, click **Request video**, and the clip
is fed straight into the live 3D-reconstruction pipeline.

Two pieces:

| Piece | File | Role |
|---|---|---|
| **Server** | [`lyra_2/_src/inference/demo_server.py`](lyra_2/_src/inference/demo_server.py) | Loads the model **once**, serves `POST /generate` (image → mp4). Wraps the same pipeline as `lyra2_zoomgs_inference.py`. |
| **GUI client** | [`visergui/`](visergui/) (`train_and_view.py`, `viewer.py`, `video_api.py`) | The **Demo tab** POSTs the image + options to the server and runs the returned video through init/append. |

You can use the server on its own (curl / scripts) and add the GUI later.

---

## 0. Prerequisites

- Checkpoints downloaded under `checkpoints/` (see the main [README](README.md) →
  *Download Checkpoints*).
- The **`lyra2` conda env** installed (see [INSTALL.md](INSTALL.md) /
  [INSTALL_BLACKWELL.md](INSTALL_BLACKWELL.md)).
- A CUDA GPU with enough free VRAM (the model needs the better part of an 80–96 GB
  card; make sure another training session isn't already holding it).

> ⚠️ **Always launch through the conda env** (`conda run -n lyra2 …` or
> `conda activate lyra2` first). The env's activation hook sets an `LD_PRELOAD`
> shim that hides a stray system `libcudart.so.13`; if you call the env's
> `python` binary directly the shim never loads and the depth step (DA3) aborts
> with `RuntimeError: Multiple libcudart libraries found`.

---

## 1. Run the server

From the repo root:

```bash
PYTHONPATH=. conda run --no-capture-output -n lyra2 \
  python -m lyra_2._src.inference.demo_server --use_dmd --port 8080
```

The model loads once at startup (~40–120 s); when you see the server is ready, every
request reuses it. Generation is serialized on a single GPU lock (one request at a
time).

### Server options

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8080` | Listen port. |
| `--host` | `0.0.0.0` | Bind address. |
| `--use_dmd` / `--no-use_dmd` | **on** | DMD 4-step distillation LoRA. **On = ~tens of seconds/clip.** Off = full 50-step sampling, minutes/clip, best quality. |
| `--checkpoint_dir` | `checkpoints/model` | Lyra2 model checkpoint dir. |
| `--experiment` | `lyra2` | Experiment config name. |
| `--default-resolution` | `480,832` | Server-side default `H,W` if a request omits one. |
| `--default-trajectory` | `horizontal_zoom` | Default camera path. |
| `--output-dir` | `outputs/demo_server` | Where per-job temp files / mp4s are written. |

### Endpoints

| Method / path | Purpose |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true}` once warm. |
| `GET /resolutions` | `{"default":"480p","presets":[{label,height,width}, …]}` — drives the GUI dropdown. |
| `POST /generate` | multipart form → returns an `mp4` (`Content-Type: video/mp4`). |

`POST /generate` form fields (all optional except `image`):

| Field | Default | Notes |
|---|---|---|
| `image` | — (required) | The conditioning frame (png/jpg). |
| `prompt` | generic caption | Text prompt. If omitted the server uses *"a high quality scenic photo"*. |
| `resolution` | `480p` | A preset label (`480p`/`360p`/`320p`/`240p`) **or** a raw `"H,W"`. H and W must be multiples of 16. Smaller = faster + less VRAM. |
| `trajectory` | `horizontal_zoom` | Camera path applied to both segments. Any name in `CAMERA_TRAJECTORY_CHOICES` (e.g. `horizontal_zoom`, `dolly_zoom`, `orbit_horizontal`, `spiral`). Not yet exposed in the GUI. |
| `num_frames_zoom_in` | `81` | Frames for the zoom-**in** segment. Must be `1 + 80k` (81, 161, 241, …). |
| `num_frames_zoom_out` | `241` | Frames for the zoom-**out** segment. Must be `1 + 80k`. |
| `zoom_in_strength` | `0.5` | How far the camera dollies forward. A collision check can cap actual motion. |
| `zoom_out_strength` | `1.5` | How far the camera pulls back. |
| `fps` | `16` | Output frame rate. |
| `seed` | `1` | RNG seed. |

The final clip is **zoom-out (reversed) + zoom-in** concatenated, so a 81+81 request
yields ~162 frames.

### Smoke-test with curl

```bash
# health
curl -s localhost:8080/health

# resolution presets
curl -s localhost:8080/resolutions

# generate a fast 240p clip from a sample image
curl -s -X POST localhost:8080/generate \
  -F image=@assets/samples/00.png \
  -F prompt="a scenic landscape" \
  -F resolution=240p \
  -F num_frames_zoom_in=81 \
  -F num_frames_zoom_out=81 \
  -o /tmp/out.mp4
```

Invalid inputs (resolution not ÷16, frame count not `1 + 80k`) return a clean
`400` with a JSON `{"detail": "..."}`; the server stays up.

---

## 2. Use the GUI (Demo tab)

With the server running, launch the viewer **in the same env**:

```bash
cd visergui
conda run --no-capture-output -n lyra2 python train_and_view.py \
  --demo-server-url http://localhost:8080/generate
  # …plus your usual train_and_view.py args
```

`--demo-server-url` defaults to `http://localhost:8080/generate`, so if you run the
server on the default port you can omit it.

Open the viser URL it prints, go to the **Demo** tab:

1. **Generate** folder — set the **server URL** (pre-filled) and an optional
   **prompt**, then **upload an image**.
2. **Lyra2 camera (zoom)** folder — the new controls:
   - **resolution** — dropdown populated live from the server's `/resolutions`
     (falls back to the static presets if the server isn't up when the tab builds).
   - **zoom-in frames** / **zoom-out frames** — `1 + 80k` step.
   - **zoom-in strength** / **zoom-out strength** — push-in / pull-back distance.
3. Click **Request video**. The server generates the clip; the GUI then runs it
   through the **same init/append pipeline** as the Train tab — the *first* video
   does pose + init (DA3), each *subsequent* video is an incremental append.
4. **Settings (synced with Train)**, **Incremental**, and **Training** folders behave
   as before (frozen splats, auto-train, etc.).

### GUI defaults (CLI)

Boot-time defaults for the Demo tab's zoom controls:

| Flag | Default |
|---|---|
| `--demo-server-url` | `http://localhost:8080/generate` |
| `--demo-prompt` | `""` (empty → server fallback caption) |
| `--demo-resolution` | `480p` |
| `--demo-zoom-in-frames` | `81` |
| `--demo-zoom-out-frames` | `241` |
| `--demo-zoom-in-strength` | `0.5` |
| `--demo-zoom-out-strength` | `1.5` |

---

## 3. Test the GUI without the GPU (mock server)

To exercise the Demo tab UI without loading the model, use the stdlib mock that hands
back canned clips (it ignores the zoom fields and just returns an mp4):

```bash
python visergui/mock_video_server.py            # serves assets/ours/*.mp4 on :8000
# then point the GUI at it:
#   --demo-server-url http://localhost:8000/generate
```

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: Multiple libcudart libraries found` | Launched the env python directly. Use `conda run -n lyra2` (or `conda activate lyra2`) so the `LD_PRELOAD` shim loads. See INSTALL_BLACKWELL.md. |
| `503 Model not loaded yet` | The server is still loading; wait for `/health` to report `model_loaded: true`. |
| `CUDA out of memory` at startup | Another process holds the GPU (e.g. a running training session). Free it, or use a smaller `resolution`. |
| `400` "must be 1 + 80k" / "multiples of 16" | Fix the frame count / resolution in the request or GUI. |
| GUI resolution dropdown only shows the static presets | The server wasn't reachable when the Demo tab was built. Start the server first, or it still works — the labels are valid. |

---

## Notes

- The server **does not** run GS reconstruction; that's a separate step
  (`vipe_da3_gs_recon.py`). The mp4 returned here is the flythrough video, which the
  GUI then reconstructs live.
- The underlying CLI is unchanged — `demo_server.py` imports the leaf helpers from
  [`lyra2_zoomgs_inference.py`](lyra_2/_src/inference/lyra2_zoomgs_inference.py), so
  the server and the CLI produce the same result for the same inputs.
