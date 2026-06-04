# Lyra 2.0 Demo: image → camera-move video, server + GUI

Turn a single image into a Lyra2 **camera-move exploration video** (one trajectory
per call — dolly, strafe, orbit, spiral, …) through a small REST server, then
(optionally) drive it from the viser GUI's **Demo tab** — upload an image, pick a
camera motion + resolution, click **Request video**, and the clip is fed straight
into the live 3D-reconstruction pipeline.

Two pieces:

| Piece | File | Role |
|---|---|---|
| **Server** | [`lyra_2/_src/inference/demo_server.py`](lyra_2/_src/inference/demo_server.py) | Loads the model **once**, serves `POST /generate` (image → mp4). Reuses the leaf helpers of `lyra2_zoomgs_inference.py`. |
| **GUI client** | [`visergui/`](visergui/) (`train_and_view.py`, `viewer.py`, `video_api.py`) | The full viewer's **Demo tab** POSTs the image + options to the server and runs the returned video through init/append. |
| **Slim Demo viewer** | [`visergui/demo.py`](visergui/demo.py) | A stripped-down standalone app: **just** splat rendering + a Demo tab, talking **only** to the *remote* session server (`/sequence/generate`). See [§3](#3-slim-standalone-demo-viewer-viserguidemopy). |

You can use the server on its own (curl / scripts) and add the GUI later. If you only
want the remote-server demo (no local `demo_server`), skip straight to
[§3 — the slim `demo.py` viewer](#3-slim-standalone-demo-viewer-viserguidemopy).

---

## 0. Prerequisites

- Checkpoints downloaded under `checkpoints/` (see the main [README](README.md) →
  *Download Checkpoints*).
- The **`lyra2` conda env** installed (see [INSTALL.md](INSTALL.md) /
  [INSTALL_BLACKWELL.md](INSTALL_BLACKWELL.md)).
- A CUDA GPU with enough free VRAM (the model needs the better part of an 80–96 GB
  card; make sure another training session isn't already holding it).

> ⚠️ **The lyra2 activation hook must run for the launching process.** It sets an
> `LD_PRELOAD` shim that hides a stray system `libcudart.so.13`; without it the
> depth step (DA3) aborts with `RuntimeError: Multiple libcudart libraries found`.
> Do this with **exactly one** of these — never both, and never on top of another
> active env:
>
> - **Activate once, then plain python** (from a *base* terminal):
>   `conda activate lyra2` → `PYTHONPATH=. python -m lyra_2._src.inference.demo_server …`
> - **`conda run` from base** (no prior activate):
>   `PYTHONPATH=. conda run --no-capture-output -n lyra2 python -m …`
>
> Stacking them (`conda activate lyra2` **and then** `conda run -n lyra2`), or
> activating lyra2 on top of another env, makes the hook resolve the shim against
> the wrong prefix (you'll see `libnocudart13.so … cannot be preloaded`) and can
> pull in the wrong Python (e.g. `ModuleNotFoundError: No module named 'termcolor'`).
> Start from a clean `(base)` shell.

---

## 1. Run the server

From the repo root, in a clean `(base)` terminal, pick **one** launch style
(see the activation warning above):

```bash
# Style A — activate once, then plain python:
conda activate lyra2
PYTHONPATH=. python -m lyra_2._src.inference.demo_server --use_dmd --port 8000

# Style B — conda run from base (no prior activate):
PYTHONPATH=. conda run --no-capture-output -n lyra2 \
  python -m lyra_2._src.inference.demo_server --use_dmd --port 8000
```

The model loads once at startup (~40–120 s); when you see the server is ready, every
request reuses it. Generation is serialized on a single GPU lock (one request at a
time).

### Server options

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8000` | Listen port. |
| `--host` | `0.0.0.0` | Bind address. |
| `--use_dmd` / `--no-use_dmd` | **on** | DMD 4-step distillation LoRA. **On = ~tens of seconds/clip.** Off = full 50-step sampling, minutes/clip, best quality. |
| `--checkpoint_dir` | `checkpoints/model` | Lyra2 model checkpoint dir. |
| `--experiment` | `lyra2` | Experiment config name. |
| `--default-resolution` | `256,448` | Server-side default `H,W` if a request omits one (240p). |
| `--default-trajectory` | `horizontal_zoom` | Default camera motion. |
| `--output-dir` | `outputs/demo_server` | Where per-job temp files / mp4s are written. |

### Endpoints

| Method / path | Purpose |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true}` once warm. |
| `GET /resolutions` | `{"default":"240p","presets":[{label,height,width}, …]}` — drives the GUI resolution dropdown. |
| `GET /trajectories` | `{"default":"horizontal_zoom","default_direction":"right","directions":[…],"trajectories":[…]}` — drives the GUI camera dropdowns. |
| `POST /generate` | multipart form → returns an `mp4` (`Content-Type: video/mp4`) plus an `X-Job-Id` response header. |
| `GET /last_frame/{job_id}` | returns the **final frame** of a previous job as a PNG — POST it back as the next `image` to continue the scene (see *Chaining*). |

**One camera move per call** — `/generate` runs a single trajectory (no zoom-in/out
pairing). So `num_frames=81` yields an 81-frame clip.

`POST /generate` form fields (all optional except `image`):

| Field | Default | Notes |
|---|---|---|
| `image` | — (required) | The conditioning frame (png/jpg). |
| `prompt` | generic caption | Text prompt. If omitted the server uses *"a high quality scenic photo"*. |
| `resolution` | `240p` | A preset label (`480p`/`360p`/`320p`/`240p`) **or** a raw `"H,W"`. H and W must be multiples of 16. Smaller = faster + less VRAM. |
| `trajectory` | `horizontal_zoom` | The camera motion. Any of the 27 names from `GET /trajectories` — e.g. `horizontal_zoom` (dolly in/out along Z), `horizontal` (strafe X), `horizontal_lift` (Y), `orbit_horizontal`/`orbit_vertical`, `spiral`/`spiral_outwards`, `dolly_zoom` (Vertigo), `rotate_spot` (pan in place), `back`, `original` (locked-off). |
| `direction` | `right` | `left`/`right`/`up`/`down`. Meaning depends on the trajectory — for `horizontal_zoom`, `right` = forward (in), `left` = backward (out). |
| `num_frames` | `81` | Frames in the clip. Must be `1 + 80k` (81, 161, 241, …, up to 801). More frames = a longer, **genuinely continuous** move — this is the right way to "keep going". See *Continuation*. |
| `strength` | `0.5` | Move magnitude (distance for dolly/strafe, angle for orbits). A collision check can cap forward motion. |
| `fps` | `16` | Output frame rate. |
| `seed` | `1` | RNG seed. |

### Smoke-test with curl

```bash
# health
curl -s localhost:8000/health

# resolution + camera-motion options
curl -s localhost:8000/resolutions
curl -s localhost:8000/trajectories

# generate a fast 240p orbit clip from a sample image
curl -s -X POST localhost:8000/generate \
  -F image=@assets/samples/00.png \
  -F prompt="a scenic landscape" \
  -F resolution=240p \
  -F trajectory=orbit_horizontal \
  -F direction=right \
  -F num_frames=81 \
  -F strength=0.5 \
  -o /tmp/out.mp4
```

Invalid inputs (resolution not ÷16, frame count not `1 + 80k`, unknown
trajectory/direction) return a clean `400` with a JSON `{"detail": "..."}`; the
server stays up.

### Chaining — continue the same scene across calls

Each call is independent: it re-estimates depth from whatever seed image you give and
starts the camera at identity. To continue a scene, grab the **last frame** of one
clip and POST it as the seed of the next:

```bash
# call 1: capture the job id from the response header
JOB=$(curl -s -D - -o /tmp/clip1.mp4 -X POST localhost:8000/generate \
  -F image=@assets/samples/00.png -F trajectory=horizontal_zoom -F num_frames=81 \
  | grep -i '^x-job-id:' | tr -d '\r' | awk '{print $2}')

# fetch the final frame of clip 1
curl -s "localhost:8000/last_frame/$JOB" -o /tmp/clip1_last.png

# call 2: continue from there with a different motion
curl -s -o /tmp/clip2.mp4 -X POST localhost:8000/generate \
  -F image=@/tmp/clip1_last.png -F trajectory=orbit_horizontal -F num_frames=81
```

> You can seed from **any** frame (the original or a generated one) — you're not
> locked to the first frame. **Caveat:** because each call re-grounds (fresh depth +
> identity pose), continuity is *visual*, not a single consistent 3D coordinate frame,
> so geometry/appearance drift accumulates over many chained calls. For truly fused
> continuity, use the GUI's incremental-append pipeline (it stitches clips into one 3D
> scene) or author a single long path via `lyra2_custom_traj_inference` (its AR cache
> carries spatial memory across 80-frame chunks within that one call).

### Continuation — why "keep going" means one longer call

Each `POST /generate` (and the GUI's "continue from last clip") rebuilds **all**
spatial state from scratch: the camera resets to identity (`initial_w2c = eye(4)`),
depth/scale are re-estimated from the seed frame, and the model's spatial memory
(its `Sparse3DCache` + history latents) is recreated. So chaining separate calls only
hands off the *last frame* — it does **not** carry position, world scale, or
geometry, which is why multiple short calls don't really "continue."

The mechanism that *does* continue is the **autoregressive chunk loop inside a single
call**: the pipeline generates the path in 80-frame chunks and carries its cache
across them. So to keep the camera moving coherently, **raise `num_frames` in one
call** (161, 241, 321, … up to 801) rather than making several calls:

```bash
# one continuous 241-frame strafe — shared cache + scale across the whole move
curl -s -X POST localhost:8000/generate \
  -F image=@assets/samples/00.png \
  -F trajectory=horizontal -F direction=left \
  -F num_frames=241 -F strength=2.0 -F resolution=240p \
  -o /tmp/long.mp4
```

For continuity across *separate* clips, prefer the GUI's incremental-append (fuses
clips into one 3D scene) or `lyra2_custom_traj_inference` with a long authored path.

---

## 2. Use the GUI (Demo tab)

With the server running, launch the viewer **in the same env**:

```bash
cd visergui
conda run --no-capture-output -n lyra2 python train_and_view.py \
  --demo-server-url http://localhost:8000/generate
  # …plus your usual train_and_view.py args
```

`--demo-server-url` defaults to `http://localhost:8000/generate`, so if you run the
server on the default port you can omit it.

Open the viser URL it prints, go to the **Demo** tab:

1. **Generate** folder — set the **server URL** (pre-filled) and an optional
   **prompt**, then **upload an image**.
2. **Lyra2 camera** folder — one camera move per request:
   - **resolution** — dropdown populated live from the server's `/resolutions`
     (falls back to the static presets if the server isn't up when the tab builds).
   - **trajectory** — dropdown of all 27 camera motions, populated from
     `/trajectories`.
   - **direction** — `left`/`right`/`up`/`down`.
   - **num_frames** — `1 + 80k` step.
   - **strength** — move magnitude (distance for dolly/strafe, angle for orbits).
3. Click **Request video**. The server generates the clip; the GUI then runs it
   through the **same init/append pipeline** as the Train tab — the *first* video
   does pose + init (DA3), each *subsequent* video is an incremental append. (This
   append pipeline is the GUI's built-in scene-continuity path — no manual frame
   chaining needed.)
4. **Settings (synced with Train)**, **Incremental**, and **Training** folders behave
   as before (frozen splats, auto-train, etc.).

### GUI defaults (CLI)

Boot-time defaults for the Demo tab's camera controls:

| Flag | Default |
|---|---|
| `--demo-server-url` | `http://localhost:8000/generate` |
| `--demo-prompt` | `""` (empty → server fallback caption) |
| `--demo-resolution` | `240p` |
| `--demo-trajectory` | `horizontal_zoom` |
| `--demo-direction` | `right` |
| `--demo-num-frames` | `81` |
| `--demo-strength` | `0.5` |
| `--demo-backend` | `local` (`local` or `sequence`) |
| `--demo-token` | `$LYRA_DEMO_TOKEN`, else `lai_server/lyra_token.txt` |
| `--demo-sequence-url` | the Lightning cloudspace URL (pre-fills the URL field when backend=sequence) |

### Backend switch: local vs. a collaborator's "sequence" server

The Demo→**Generate** folder has a **backend** dropdown:

- **local** — our `demo_server` `/generate` (the default; everything above).
- **sequence** — a collaborator's session-based server: `POST <url>/sequence/generate`
  with `Authorization: Bearer <token>` and **server-side scene continuity**. The GUI
  synthesizes a trajectory `.npz` from the same camera controls (trajectory /
  direction / num_frames / strength) and sends it as a file part. The **first**
  Request also sends the uploaded image and captures the server's `X-Session-Id`
  (shown in the **session** line); **subsequent** Requests send only the `session_id`
  + a new trajectory, so the camera continues the same scene server-side. **Reset**
  ends the session (next Request starts a fresh one).

  The server URL and bearer token are pre-filled by default (URL = the Lightning
  cloudspace; token = `lai_server/lyra_token.txt`), so usually you just switch
  **backend → sequence** and Request. Override with `--demo-sequence-url` /
  `--demo-token` if they change. The trajectory npz uses the lyra custom-traj schema
  (`w2c (N,4,4)`, `intrinsics (N,3,3)`, `image_height`, `image_width`).

  > **Note — Lightning URL needs the port prefix.** The reachable host is
  > `https://8000-<id>.cloudspaces.litng.ai` (the `8000-` prefix routes to the app's
  > port). The bare `https://<id>.cloudspaces.litng.ai` returns a plain `404 page not
  > found` from the gateway. Verified live: `GET /health` → `{"status":"ready"}`, and
  > a gen0 (new session) → gen1 (continue, no image) pair returns mp4s with a growing
  > `X-Cache-Entries` (spatial memory accumulating).

> The returned clips are appended into the local splat scene exactly like the local
> backend (first = init, later = incremental append).

---

## 3. Slim standalone Demo viewer (`visergui/demo.py`)

[`visergui/demo.py`](visergui/demo.py) is a **simpler, self-contained** alternative to
the full `train_and_view.py` + `viewer.py` for the demo workflow. It keeps only:

1. **Splat rendering** — a live viser viewport of the reconstructed gaussians (it
   reuses the proven `SceneState` + `Renderer` from `viewer.py`).
2. **A single Demo tab** — generate clips from the **remote session server**, turn
   them into splats, and (optionally) train.

It talks to **one** backend only: the collaborator's `/sequence/generate` server
(Bearer token, server-side scene continuity). It does **not** use the local
`demo_server`, and it does **not** build the Train / Mesh / Inpaint / Incremental
tabs. Use it when you just want *image → remote video → splats* with minimal UI.

### Run it (in the `splat` env)

`demo.py` is a viewer/trainer client — it runs in the **`splat`** conda env (the same
env as `viewer.py`/`train_and_view.py`), **not** `lyra2`. The model lives on the
remote server; nothing heavy loads locally except DA3 for reconstruction.

```bash
cd /path/to/Lyra-2
conda run -n splat python visergui/demo.py \
  --demo-sequence-url https://8000-<id>.cloudspaces.litng.ai \
  --port 8080
# then open http://localhost:8080  (ssh -L 8080:localhost:8080 ... if remote)
```

The server URL and Bearer token are pre-filled (URL = the Lightning cloudspace;
token = `$LYRA_DEMO_TOKEN`, else `lai_server/lyra_token.txt`). The same **port-prefix
gotcha** applies — the reachable host is `https://8000-<id>.cloudspaces.litng.ai`
(see the note in [§2](#backend-switch-local-vs-a-collaborators-sequence-server)).

### Workflow

1. **Generate** folder — server URL + token are pre-filled; set an optional prompt,
   then either **Upload image** (from your browser) or **Choose image…** (a popup
   gallery of every `.jpg`/`.png` in `assets/ours/`, configurable via `--assets-dir`).
2. **Camera trajectory** folder — pick **resolution** (480p/360p/320p/240p — this
   sets the generated video's H×W), **trajectory** (any of the 27 camera moves),
   **direction**, **num_frames** (`1 + 80k`), **strength**.
3. Click **Request video**. A clip comes back from the server and is turned into
   splats **automatically**: the **first** clip runs DA3 pose + init; each **later**
   clip is **appended** (it continues the same server session). Nothing trains
   automatically. Repeat Request to chain more clips.
4. Click **Train** when you want to optimize (or tick *auto-train after each clip*).

### Reconstruction controls — keeping the splat count sane

The sequence server's clips all branch from the same start point, so they overlap
heavily; without dedup the appended splats stack into millions. Two knobs (in the
**Reconstruction** folder, live-adjustable, applied to both init and append):

| Control | Default | Effect |
|---|---|---|
| **max_points (per-clip cap)** | `1,000,000` | Hard cap on splats added per clip — the init subsamples to this, and each appended clip seeds at most this many candidates. Caps the *trained* voxel init too (not just the unused v0 ply). Lower to bound the total. |
| **seed dedup radius (× init voxel)** | `3.0` | A new splat is dropped if an existing one is within this many init-voxels. **Raise to 4–5** to merge the overlapping sequence clips instead of stacking duplicates. |

Plus the usual init settings: `max_frames`, `confidence_quantile`, `remove_sky`,
`sh_max_deg`, `lpips_weight`, `void_weight`, `densify`, `mode` (3dgs/2dgs).

### Other controls

- **Initialize from downloaded clips** (Reconstruction) — a clean **rebuild** from
  every clip downloaded so far (init the first, append the rest). Handy after
  changing reconstruction settings or to recover from a bad append.
- **Training** folder — **Train**, **Pause**, **Prune splats** (drop floater / spiky
  / oversized splats and report before→after counts), **Reset** (clear splats +
  cameras and end the server session).
- **View** folder — show cameras, camera size, max render res, Reset camera.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--demo-sequence-url` | the Lightning cloudspace URL | Remote server base (or full `/sequence/generate`). |
| `--demo-token` | `$LYRA_DEMO_TOKEN`, else `lai_server/lyra_token.txt` | Bearer token. |
| `--assets-dir` | `<repo>/assets/ours` | Folder the **Choose image…** picker scans. |
| `--demo-resolution` / `--demo-trajectory` / `--demo-direction` / `--demo-num-frames` / `--demo-strength` | `240p` / `horizontal_zoom` / `right` / `81` / `0.5` | Camera-move defaults. |
| `--max-points` | `1,000,000` | Per-clip splat cap (see above). |
| `--seed-dedup` | `3.0` | Seed dedup radius in init-voxel units. |
| `--max-frames` / `--confidence-quantile` / `--remove-sky` / `--sh-max-deg` / `--lpips-weight` / `--void-weight` / `--densify` / `--mode` | as Train tab | Reconstruction defaults. |
| `--port` / `--host` / `--out-dir` | `8080` / `0.0.0.0` / `vipe_outputs` | Server bind + output root. |

---

## 4. Test the GUI without the GPU (mock server)

To exercise the Demo tab UI without loading the model, use the stdlib mock that hands
back canned clips (it ignores the camera fields and just returns an mp4):

```bash
python visergui/mock_video_server.py            # serves assets/ours/*.mp4 on :8000
# then point the GUI at it:
#   --demo-server-url http://localhost:8000/generate
```

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: Multiple libcudart libraries found` | The activation hook didn't run for this process. Use one of the two launch styles above (activate-then-python, or conda-run-from-base). See INSTALL_BLACKWELL.md. |
| `libnocudart13.so … cannot be preloaded` **and/or** `ModuleNotFoundError: No module named 'termcolor'` | Double/nested activation (e.g. another env active, then `conda activate lyra2`, then `conda run -n lyra2`). Start from a clean `(base)` shell and use exactly one launch style. |
| `503 Model not loaded yet` | The server is still loading; wait for `/health` to report `model_loaded: true`. |
| `CUDA out of memory` at startup | Another process holds the GPU (e.g. a running training session). Free it, or use a smaller `resolution`. |
| `400` "must be 1 + 80k" / "multiples of 16" / "Invalid trajectory/direction" | Fix the frame count / resolution / trajectory / direction in the request or GUI. |
| GUI dropdowns only show the static fallback options | The server wasn't reachable when the Demo tab was built. Start the server first; the listed options are still valid. |
| `500 Server Error … /sequence/generate` | The remote server hit an error generating — usually a seed image that gives degenerate depth (e.g. a flat diagram). Use a real photo. The server's detail message (`Sequential generation error: …`) is in its own logs. |
| Splat count balloons into the millions (`demo.py`) | The overlapping sequence clips weren't deduped enough. Raise **seed dedup radius** to 4–5 and/or lower **max_points**, then **Initialize from downloaded clips** to rebuild. |
| `demo.py` "Choose image…" shows nothing | No `.jpg`/`.png` in the scanned folder. Point `--assets-dir` at a folder that has some. |

---

## Notes

- The server **does not** run GS reconstruction; that's a separate step
  (`vipe_da3_gs_recon.py`). The mp4 returned here is the flythrough video, which the
  GUI then reconstructs live.
- `demo_server.py` reuses the leaf helpers from
  [`lyra2_zoomgs_inference.py`](lyra_2/_src/inference/lyra2_zoomgs_inference.py) (depth,
  trajectory, and the per-direction generate call), so a single-trajectory server
  request produces the same frames as one direction of the underlying CLI.
- For arbitrary, fully-authored camera paths (your own per-frame `w2c`/intrinsics),
  see `lyra2_custom_traj_inference` — it takes a trajectory `.npz` and is the route
  for paths the 27 presets don't cover.
