"""
demo.py — a slim, self-contained Demo viewer.

This is a stripped-down sibling of ``train_and_view.py`` + ``viewer.py``. It keeps
ONLY two things:

  1. **Splat rendering** — a live viser viewport that renders the reconstructed
     gaussians (reusing the proven ``SceneState`` + ``Renderer`` from viewer.py).
  2. **The Demo tab** — upload or choose an image, pick a camera trajectory, hit
     "Request video": a clip comes back from a *remote* generation server and is
     turned into splats automatically (DA3 pose + init for the first clip, append
     for each later one — the bits we need from the Train tab) and shows up in the
     viewport. Repeat to chain clips in one server session. "Initialize" does a
     clean rebuild from every downloaded clip. Nothing trains automatically.

It talks to ONE backend: the collaborator's session server (``/sequence/generate``,
Bearer token, server-side scene continuity). It does NOT use our local
``demo_server`` and does NOT build the Train / Inpaint / Mesh / Incremental tabs.

The video → splats machinery is delegated to ``SplatTrainer`` exactly as
``train_and_view.py`` wires it; this file is just the thin viser glue + the Demo
GUI around it.

Usage (run in the `splat` conda env):
    conda run -n splat python visergui/demo.py \
        --demo-sequence-url https://8000-<id>.cloudspaces.litng.ai \
        --port 8080
    # then open http://localhost:8080 (ssh -L 8080:localhost:8080 ... if remote)
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import viser

# Make sibling modules (video_api, viewer, splat_trainer, training) and the repo
# root (lyra_2 package) importable regardless of the launching CWD.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import video_api
from training import BackgroundTrainingThread
from splat_trainer import SplatTrainer
# Reuse the rendering core verbatim — no point reimplementing what already works.
from viewer import SceneState, Renderer, _compute_home_pose, _rotmat_to_wxyz
from lyra_2._src.inference.camera_traj_utils import (
    CAMERA_TRAJECTORY_CHOICES,
    build_camera_trajectory,
)

_DIRECTIONS = ("left", "right", "up", "down")
_RESOLUTIONS = tuple(video_api.RESOLUTION_PRESETS)  # ("480p","360p","320p","240p")


def _default_demo_token() -> str:
    """Bearer token for the sequence backend: $LYRA_DEMO_TOKEN, else the contents
    of lai_server/lyra_token.txt (repo-root sibling), else empty."""
    tok = os.environ.get("LYRA_DEMO_TOKEN", "").strip()
    if tok:
        return tok
    try:
        return (_REPO / "lai_server" / "lyra_token.txt").read_text().strip()
    except Exception:
        return ""


class DemoApp:
    """A minimal viser app: splat viewport + a single Demo tab driving a remote
    session-based generation server, with DA3 init/append turning each returned
    clip into splats."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        scene: SceneState,
        trainer: SplatTrainer,
        control: BackgroundTrainingThread,
        defaults: dict,
    ) -> None:
        self.scene = scene
        self.trainer = trainer
        self.control = control
        self.defaults = defaults
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Folder scanned by the "Choose image" picker (server-side images).
        self._assets_dir = Path(
            defaults.get("assets_dir", _REPO / "assets" / "ours")
        ).expanduser()

        self.renderer = Renderer(device=self.device)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(dark_mode=True)

        # ---- demo state ------------------------------------------------- #
        self._op_lock = threading.Lock()       # serialize trainer-mutating ops
        self._demo_image: tuple[str, bytes] | None = None
        self._session_id: str | None = None    # server-side session
        self._count = 0
        self._clips: list[str] = []
        self._train_cam_names: list[str] = []

        # ---- camera home pose ------------------------------------------- #
        self.home_position = (0.0, 0.0, 5.0)
        self.home_look_at = (0.0, 0.0, 0.0)
        self.home_up = (0.0, -1.0, 0.0)

        # ---- render settle (crisp still frame after motion) ------------- #
        self._settle_timers: dict[int, threading.Timer] = {}
        self._settle_delay_s = 0.15

        # ---- pump (re-render on splat changes, mirror train readouts) ---- #
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None

        self._build_gui()
        self.server.on_client_connect(self._on_client_connect)
        self._start_pump()

    # --------------------------------------------------------------------- #
    # GUI
    # --------------------------------------------------------------------- #

    def _build_gui(self) -> None:
        d = self.defaults
        tabs = self.server.gui.add_tab_group()
        with tabs.add_tab("Demo"):
            with self.server.gui.add_folder("Generate (remote server)"):
                # Connection settings rarely change — keep them tucked away collapsed.
                with self.server.gui.add_folder("Connection", expand_by_default=False):
                    self.gui_server = self.server.gui.add_text(
                        "server_url", initial_value=str(d.get("server_url", "")),
                    )
                    self.gui_token = self.server.gui.add_text(
                        "token", initial_value=str(d.get("token", "")),
                    )
                self.gui_session = self.server.gui.add_markdown("**session:** none")
                self.gui_prompt = self.server.gui.add_text(
                    "prompt", initial_value=str(d.get("prompt", "")),
                )
                self.gui_image_btn = self.server.gui.add_upload_button(
                    "Upload image", icon=viser.Icon.PHOTO, mime_type="image/*",
                )
                self.gui_choose_btn = self.server.gui.add_button(
                    "Choose image…", icon=viser.Icon.FOLDER_OPEN,
                )
                self.gui_image_status = self.server.gui.add_markdown(
                    "**image:** none — upload or choose one"
                )
                self.gui_request_btn = self.server.gui.add_button(
                    "Request video", icon=viser.Icon.MOVIE,
                )
                self.gui_status = self.server.gui.add_markdown(
                    "**demo:** upload an image, then Request video"
                )
                self.gui_count = self.server.gui.add_markdown(
                    "**clips downloaded:** 0"
                )

            with self.server.gui.add_folder("Camera trajectory"):
                self.gui_resolution = self.server.gui.add_dropdown(
                    "resolution", options=_RESOLUTIONS,
                    initial_value=str(d.get("resolution", "240p")),
                )
                self.gui_trajectory = self.server.gui.add_dropdown(
                    "trajectory", options=CAMERA_TRAJECTORY_CHOICES,
                    initial_value=str(d.get("trajectory", "horizontal_zoom")),
                )
                self.gui_direction = self.server.gui.add_dropdown(
                    "direction", options=_DIRECTIONS,
                    initial_value=str(d.get("direction", "right")),
                )
                self.gui_num_frames = self.server.gui.add_number(
                    "num_frames", initial_value=int(d.get("num_frames", 81)),
                    min=81, max=801, step=80,
                )
                self.gui_strength = self.server.gui.add_slider(
                    "strength", min=0.0, max=2.0, step=0.05,
                    initial_value=float(d.get("strength", 0.5)),
                )

            with self.server.gui.add_folder("Reconstruction"):
                self.gui_init_btn = self.server.gui.add_button(
                    "Initialize from downloaded clips", icon=viser.Icon.PLAYER_PLAY,
                )
                self.gui_max_frames = self.server.gui.add_number(
                    "max_frames", initial_value=int(d.get("max_frames", 32)),
                    min=4, max=240, step=1,
                )
                self.gui_max_points = self.server.gui.add_number(
                    "max_points (per-clip cap)",
                    initial_value=int(d.get("max_points", 1_000_000)),
                    min=50_000, max=3_000_000, step=50_000,
                    hint="Caps the splats added per clip: the init subsamples to "
                         "this, and each appended clip seeds at most this many "
                         "candidates before dedup. Lower it if the total balloons.",
                )
                self.gui_seed_dedup = self.server.gui.add_slider(
                    "seed dedup radius (× init voxel)",
                    min=1.0, max=8.0, step=0.5,
                    initial_value=float(d.get("seed_dedup", 3.0)),
                    hint="When appending a clip, a new splat is dropped if an "
                         "existing one is within this many init-voxels. The "
                         "sequence server's clips all branch from the same start, "
                         "so they overlap heavily — raise this (3–5) to merge the "
                         "duplicates instead of stacking millions of splats.",
                )
                self.gui_scale_clamp = self.server.gui.add_number(
                    "max splat scale (× init voxel)",
                    initial_value=float(d.get("scale_clamp", 2.0)),
                    min=0.5, max=100_000.0, step=0.5,
                    hint="Caps each gaussian's per-axis size at this many init-"
                         "voxels, every training step (2DGS mode ignores it). The "
                         "default 2.0 keeps splats tight; raise it (10s–1000s) to "
                         "let splats grow big and cover more per gaussian. A huge "
                         "value is effectively unclamped. Live — takes effect on "
                         "the next step.",
                )
                self.gui_conf_q = self.server.gui.add_slider(
                    "confidence_quantile", min=0.0, max=0.95, step=0.05,
                    initial_value=float(d.get("confidence_quantile", 0.6)),
                )
                self.gui_remove_sky = self.server.gui.add_checkbox(
                    "remove_sky", initial_value=bool(d.get("remove_sky", True)),
                )
                self.gui_sh_max_deg = self.server.gui.add_number(
                    "sh_max_deg", initial_value=int(d.get("sh_max_deg", 2)),
                    min=0, max=3, step=1,
                )
                self.gui_lpips_weight = self.server.gui.add_slider(
                    "lpips_weight", min=0.0, max=1.0, step=0.01,
                    initial_value=float(d.get("lpips_weight", 0.05)),
                )
                self.gui_void_weight = self.server.gui.add_slider(
                    "void_weight", min=0.0, max=2.0, step=0.05,
                    initial_value=float(d.get("void_weight", 0.5)),
                )
                self.gui_densify = self.server.gui.add_checkbox(
                    "densify", initial_value=bool(d.get("densify", True)),
                )
                self.gui_mode = self.server.gui.add_dropdown(
                    "mode", options=("3dgs", "2dgs"),
                    initial_value=str(d.get("mode", "3dgs")),
                )
                self.gui_autotrain = self.server.gui.add_checkbox(
                    "auto-train after each clip", initial_value=False,
                )

            with self.server.gui.add_folder("Training"):
                self.gui_train_btn = self.server.gui.add_button("Train")
                self.gui_pause_btn = self.server.gui.add_button("Pause")
                self.gui_prune_btn = self.server.gui.add_button(
                    "Prune splats", icon=viser.Icon.FILTER,
                )
                with self.server.gui.add_folder("Prune settings",
                                                expand_by_default=False):
                    self.gui_prune_opa = self.server.gui.add_slider(
                        "min opacity", min=0.0, max=1.0, step=0.01,
                        initial_value=float(d.get("prune_opa_min", 0.05)),
                        hint="Remove splats with opacity below this.",
                    )
                    self.gui_prune_scale = self.server.gui.add_slider(
                        "max scale (× scene)", min=0.01, max=2.0, step=0.01,
                        initial_value=float(d.get("prune_scale_max_frac", 0.10)),
                        hint="Remove splats whose largest axis exceeds this "
                             "fraction of the scene scale (kills giant blobs).",
                    )
                    self.gui_prune_aniso = self.server.gui.add_number(
                        "max anisotropy", min=1.0, max=200.0, step=1.0,
                        initial_value=float(d.get("prune_aniso_max", 10.0)),
                        hint="Remove needle/disk splats whose long:short axis "
                             "ratio exceeds this.",
                    )
                    self.gui_prune_knn = self.server.gui.add_checkbox(
                        "KNN floater removal",
                        initial_value=bool(d.get("prune_use_knn", True)),
                        hint="Drop isolated floaters (mean-neighbor-distance "
                             "outliers). Needs scipy.",
                    )
                    self.gui_prune_knn_k = self.server.gui.add_number(
                        "KNN neighbors (k)", min=4, max=100, step=1,
                        initial_value=int(d.get("prune_knn_k", 20)),
                    )
                    self.gui_prune_knn_std = self.server.gui.add_slider(
                        "KNN std threshold", min=0.5, max=6.0, step=0.1,
                        initial_value=float(d.get("prune_knn_std", 2.0)),
                        hint="A splat is a floater if its mean neighbor distance "
                             "is more than this many std-devs above the mean. "
                             "Lower = more aggressive.",
                    )
                self.gui_reset_btn = self.server.gui.add_button("Reset")
                self.gui_train_status = self.server.gui.add_markdown(
                    "**status:** stopped"
                )
                self.gui_step = self.server.gui.add_markdown("**step:** 0")
                self.gui_splat_count = self.server.gui.add_markdown("**splats:** 0")

            with self.server.gui.add_folder("View"):
                self.gui_show_cams = self.server.gui.add_checkbox(
                    "show cameras", initial_value=True,
                )
                self.gui_cam_scale = self.server.gui.add_slider(
                    "camera size", min=0.01, max=1.0, step=0.01,
                    initial_value=0.15,
                )
                self.gui_max_res = self.server.gui.add_slider(
                    "max render res", min=256, max=2048, step=64,
                    initial_value=1280,
                )
                self.gui_reset_cam_btn = self.server.gui.add_button("Reset camera")

        # ---- wiring ----------------------------------------------------- #
        self.gui_image_btn.on_upload(lambda _e: self._on_image_upload())
        self.gui_choose_btn.on_click(lambda _e: self._on_choose_image_click())
        self.gui_init_btn.on_click(lambda _e: self._on_initialize_click())
        self.gui_request_btn.on_click(lambda _e: self._on_request_click())
        self.gui_train_btn.on_click(lambda _e: self._on_resume_training())
        self.gui_pause_btn.on_click(lambda _e: self._on_pause_training())
        self.gui_prune_btn.on_click(lambda _e: self._on_prune_click())
        self.gui_reset_btn.on_click(lambda _e: self._on_reset_click())
        self.gui_reset_cam_btn.on_click(lambda _e: self._on_reset_camera())
        self.gui_show_cams.on_update(lambda _e: self._publish_cams())
        self.gui_cam_scale.on_update(lambda _e: self._publish_cams())
        self.gui_seed_dedup.on_update(lambda _e: self._apply_splat_budget())
        self.gui_max_points.on_update(lambda _e: self._apply_splat_budget())
        self.gui_scale_clamp.on_update(lambda _e: self._apply_splat_budget())
        # lpips/void are read by the trainer every step → push them live.
        self.gui_lpips_weight.on_update(lambda _e: self._apply_live_loss_weights())
        self.gui_void_weight.on_update(lambda _e: self._apply_live_loss_weights())
        # Push the GUI defaults into the trainer up front so the first clip already
        # uses the chosen dedup radius / point cap (not the trainer's bare defaults).
        self._apply_splat_budget()

    # --------------------------------------------------------------------- #
    # Rendering glue
    # --------------------------------------------------------------------- #

    def _on_client_connect(self, client: viser.ClientHandle) -> None:
        client.camera.on_update(lambda _cam: self._render_for(client))
        try:
            client.camera.up_direction = self.home_up
            client.camera.position = self.home_position
            client.camera.look_at = self.home_look_at
        except Exception:
            pass
        self._render_for(client)

    def _render_for(self, client: viser.ClientHandle, *,
                    force_full_res: bool = False) -> None:
        cam = client.camera
        try:
            W = int(cam.image_width)
            H = int(cam.image_height)
        except (TypeError, ValueError):
            return
        if W <= 0 or H <= 0:
            return
        max_res = int(self.gui_max_res.value)
        max_dim = max(W, H)
        if max_dim > max_res:
            s = max_res / max_dim
            W = max(1, int(round(W * s)))
            H = max(1, int(round(H * s)))
        img, _ms, moving = self.renderer.render(
            self.scene, cam, W, H,
            sh_degree=3, color_mode="RGB",
            near=0.01, far=1000.0,
            adaptive_res=True, moving_scale=0.4,
            force_full_res=force_full_res,
        )
        client.scene.set_background_image(img)
        if moving and not force_full_res:
            self._schedule_settle(client)
        else:
            self._cancel_settle(client)

    def _render_all_clients(self) -> None:
        for client in self.server.get_clients().values():
            self._render_for(client)

    def _schedule_settle(self, client: viser.ClientHandle) -> None:
        cid = int(client.client_id)
        old = self._settle_timers.pop(cid, None)
        if old is not None:
            old.cancel()
        t = threading.Timer(self._settle_delay_s, lambda: self._settle_render(cid))
        t.daemon = True
        self._settle_timers[cid] = t
        t.start()

    def _cancel_settle(self, client: viser.ClientHandle) -> None:
        old = self._settle_timers.pop(int(client.client_id), None)
        if old is not None:
            old.cancel()

    def _settle_render(self, client_id: int) -> None:
        client = self.server.get_clients().get(client_id)
        if client is None:
            return
        self._settle_timers.pop(client_id, None)
        self._render_for(client, force_full_res=True)

    def _on_reset_camera(self) -> None:
        for client in self.server.get_clients().values():
            try:
                client.camera.up_direction = self.home_up
                client.camera.position = self.home_position
                client.camera.look_at = self.home_look_at
            except Exception:
                pass

    def _snap_home_to_scene(self) -> None:
        if self.scene.means is None:
            return
        means_np = self.scene.means.detach().cpu().numpy()
        self.home_position, self.home_look_at, self.home_up = _compute_home_pose(means_np)
        self._on_reset_camera()

    def _publish_cams(self) -> None:
        """Draw the trainer's per-frame camera frustums so the trajectory is
        visible. Clears + redraws each call; honors the 'show cameras' toggle."""
        for n in self._train_cam_names:
            try:
                self.server.scene.remove_by_name(n)
            except Exception:
                pass
        self._train_cam_names.clear()
        d = self.trainer.data
        if d is None or not bool(self.gui_show_cams.value):
            return
        c2w = d.c2w.detach().cpu().numpy()
        K = d.K.detach().cpu().numpy()
        H, W = int(d.H), int(d.W)
        aspect = W / max(H, 1)
        scale = float(self.gui_cam_scale.value)
        for i in range(c2w.shape[0]):
            R = c2w[i, :3, :3]
            t = c2w[i, :3, 3]
            wxyz = _rotmat_to_wxyz(R)
            fy = float(K[i, 1, 1])
            fov_y = 2.0 * math.atan(0.5 * H / max(fy, 1e-9))
            name = f"demo_cams/{i:04d}"
            self.server.scene.add_camera_frustum(
                name=name, fov=fov_y, aspect=aspect, scale=scale,
                color=(255, 153, 51),
                wxyz=tuple(float(x) for x in wxyz),
                position=tuple(float(x) for x in t),
            )
            self._train_cam_names.append(name)

    # --------------------------------------------------------------------- #
    # Training control
    # --------------------------------------------------------------------- #

    def _on_resume_training(self) -> None:
        if self.control is None:
            return
        self.control.start()   # idempotent
        self.control.resume()

    def _on_pause_training(self) -> None:
        if self.control is not None:
            try:
                self.control.pause()
            except Exception:
                pass

    def _on_prune_click(self) -> None:
        """Pause training and drop floater / spiky / oversized splats
        (trainer.prune_splats), then report the before→after counts. Leaves
        training paused so you can inspect before resuming."""
        t = self.trainer
        if not bool(getattr(t, "_initialized", False)):
            self.gui_status.content = "**demo:** nothing to prune — initialize a scene first"
            return
        if getattr(t, "prune_splats", None) is None:
            self.gui_status.content = "**demo:** trainer doesn't support pruning"
            return
        if not self._op_lock.acquire(blocking=False):
            self.gui_status.content = "**demo:** busy — another operation is running"
            return
        try:
            self.gui_prune_btn.disabled = True
            self._on_pause_training()
            self.gui_status.content = "**demo:** pruning…"
            counts = t.prune_splats(    # publishes to the scene → pump re-renders
                opa_min=float(self.gui_prune_opa.value),
                scale_max_frac=float(self.gui_prune_scale.value),
                aniso_max=float(self.gui_prune_aniso.value),
                use_knn=bool(self.gui_prune_knn.value),
                knn_k=int(self.gui_prune_knn_k.value),
                knn_std=float(self.gui_prune_knn_std.value),
            )
            self._snap_home_to_scene()
            parts = [f"{k}={counts[k]}" for k in ("opacity", "scale", "aniso", "knn")
                     if k in counts and counts[k] >= 0]
            detail = ", ".join(parts) if parts else "no filters fired"
            self.gui_status.content = (
                f"**demo:** pruned {counts['started']:,} → {counts['kept']:,} "
                f"(−{counts['removed_total']:,}; {detail}) — click Train to resume"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_status.content = f"**demo:** prune error — {type(e).__name__}: {e}"
        finally:
            self.gui_prune_btn.disabled = False
            self._op_lock.release()

    # --------------------------------------------------------------------- #
    # Demo flow
    # --------------------------------------------------------------------- #

    def _on_image_upload(self) -> None:
        f = self.gui_image_btn.value
        content = getattr(f, "content", None)
        if not content:
            return
        self._demo_image = (getattr(f, "name", "image.png"), content)
        self.gui_image_status.content = (
            f"**image:** {self._demo_image[0]} ({len(content) / 1024.0:,.0f} KB)"
        )

    def _list_asset_images(self) -> list[Path]:
        """All .jpg/.jpeg/.png files in the assets folder (case-insensitive)."""
        folder = self._assets_dir
        if not folder.is_dir():
            return []
        files = [p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        return sorted(files, key=lambda p: p.name.lower())

    @staticmethod
    def _thumbnail(path: Path, max_w: int = 320) -> np.ndarray:
        """Downscaled RGB uint8 array for an image-file preview in the modal."""
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_w:
            img = img.resize((max_w, max(1, round(h * max_w / w))))
        return np.asarray(img, dtype=np.uint8)

    def _on_choose_image_click(self) -> None:
        """Open a popup gallery of every image in the assets folder; clicking one
        sets it as the seed image (same as a browser upload)."""
        files = self._list_asset_images()
        if not files:
            self.gui_image_status.content = (
                f"**image:** no .jpg/.png in {self._assets_dir}"
            )
            return
        modal = self.server.gui.add_modal("Choose an image")
        with modal:
            self.server.gui.add_markdown(
                f"**{len(files)} image(s)** in `{self._assets_dir}`"
            )
            for fp in files:
                try:
                    self.server.gui.add_image(self._thumbnail(fp), label=fp.name)
                except Exception:
                    self.server.gui.add_markdown(f"_(could not preview {fp.name})_")
                btn = self.server.gui.add_button(f"Use {fp.name}")
                btn.on_click(lambda _e, p=fp: self._pick_asset_image(p, modal))
            self.server.gui.add_button("Cancel").on_click(lambda _e: modal.close())

    def _pick_asset_image(self, path: Path, modal) -> None:
        """Load a chosen asset image as the seed image, then dismiss the popup."""
        try:
            data = path.read_bytes()
            self._demo_image = (path.name, data)
            self.gui_image_status.content = (
                f"**image:** {path.name} ({len(data) / 1024.0:,.0f} KB)"
            )
        except Exception as e:
            self.gui_image_status.content = (
                f"**image:** failed to load {path.name} — {type(e).__name__}: {e}"
            )
        finally:
            try:
                modal.close()
            except Exception:
                pass

    def _synth_trajectory_npz_bytes(self) -> bytes:
        """Synthesize a trajectory .npz from the current camera controls.

        Uses lyra's ``build_camera_trajectory`` from an identity start pose; the
        server re-grounds depth from the seed image, so a nominal center_depth is
        fine. Schema matches lyra custom-traj: w2c (N,4,4), intrinsics (N,3,3),
        image_height, image_width.
        """
        H, W = (int(x) for x in
                video_api.resolve_resolution(self.gui_resolution.value).split(","))
        fy = 0.5 * H / math.tan(0.5 * math.radians(60.0))  # ~60° vertical FOV
        K = torch.tensor(
            [[fy, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32, device=self.device,
        )
        initial_w2c = torch.eye(4, dtype=torch.float32, device=self.device)
        n = int(self.gui_num_frames.value)
        w2cs, Ks = build_camera_trajectory(
            initial_w2c, K, 1.0, n,
            str(self.gui_trajectory.value),
            str(self.gui_direction.value),
            float(self.gui_strength.value),
        )
        buf = io.BytesIO()
        np.savez(
            buf,
            w2c=w2cs.detach().cpu().numpy().astype(np.float32),
            intrinsics=Ks.detach().cpu().numpy().astype(np.float32),
            image_height=int(H),
            image_width=int(W),
        )
        return buf.getvalue()

    def _request_to_path(self, image_name: str, image_bytes: bytes) -> str:
        """Call the remote /sequence/generate backend and return a path to the
        saved mp4. First call (no session) sends the image; later calls send the
        stored session_id and continue server-side. Updates the session markdown."""
        npz_bytes = self._synth_trajectory_npz_bytes()
        resolution = video_api.resolve_resolution(self.gui_resolution.value)
        video_bytes, sid = video_api.request_sequence_video(
            str(self.gui_server.value),
            token=str(self.gui_token.value).strip(),
            trajectory_npz_bytes=npz_bytes,
            resolution=resolution,
            num_frames=int(self.gui_num_frames.value),
            prompt=str(self.gui_prompt.value).strip(),
            image_bytes=(None if self._session_id else image_bytes),
            image_name=image_name,
            session_id=self._session_id,
        )
        self._session_id = sid
        self.gui_session.content = f"**session:** {sid or 'none'}"
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_bytes)
        tmp.close()
        return tmp.name

    def _apply_splat_budget(self) -> None:
        """Push the dedup-radius, max-points, and scale-clamp controls into the
        trainer. dedup/max-points are read at init/append time; the scale clamp is
        read every training step. Setting them here takes effect on the next clip /
        step. Keeps the splat count + per-splat size in check."""
        try:
            self.trainer.set_seed_dedup_multiplier(float(self.gui_seed_dedup.value))
            self.trainer.max_points = int(self.gui_max_points.value)
            self.trainer.set_scale_clamp(float(self.gui_scale_clamp.value))
        except Exception:
            pass

    def _apply_live_loss_weights(self) -> None:
        """Push lpips_weight + void_weight straight into the trainer. step() reads
        both every iteration, so a change here takes effect mid-training on the next
        step (lpips lazily loads its net the first time the weight goes > 0). The
        other Reconstruction settings (max_frames, confidence, remove_sky, sh_max_deg,
        densify, mode) shape preprocessing/init structure and only apply on the next
        Initialize / Request — not live."""
        try:
            self.trainer.lpips_weight = float(self.gui_lpips_weight.value)
            self.trainer.void_weight = float(self.gui_void_weight.value)
        except Exception:
            pass

    def _init_from_video(self, video_path: str) -> None:
        self._apply_splat_budget()
        self.trainer.prepare_and_init(
            video=Path(video_path),
            max_frames=int(self.gui_max_frames.value),
            confidence=float(self.gui_conf_q.value),
            remove_sky=bool(self.gui_remove_sky.value),
            sh_max_deg=int(self.gui_sh_max_deg.value),
            lpips_weight=float(self.gui_lpips_weight.value),
            void_weight=float(self.gui_void_weight.value),
            use_densify=bool(self.gui_densify.value),
            mode=str(self.gui_mode.value),
        )
        self._publish_cams()

    def _append_from_video(self, video_path: str) -> None:
        self._apply_splat_budget()
        self.trainer.append_video(
            video_path, max_frames=int(self.gui_max_frames.value),
            seed_new_splats=True,
        )
        self._publish_cams()

    def _on_request_click(self) -> None:
        """Fetch one clip from the remote server and automatically build splats from
        it: the first clip inits the scene (DA3 pose + init), each later clip is
        appended. Does NOT train (use Train / the auto-train checkbox). The Initialize
        button does a clean full rebuild from every downloaded clip."""
        if self._demo_image is None:
            self.gui_status.content = "**demo:** upload or choose an image first"
            return
        if not str(self.gui_server.value).strip():
            self.gui_status.content = "**demo:** set the server URL first"
            return
        if not str(self.gui_token.value).strip():
            self.gui_status.content = "**demo:** set the Bearer token first"
            return
        if not self._op_lock.acquire(blocking=False):
            self.gui_status.content = "**demo:** busy — another operation is running"
            return
        try:
            self.gui_request_btn.disabled = True
            self.gui_reset_btn.disabled = True
            self.gui_init_btn.disabled = True
            self._on_pause_training()
            name, image_bytes = self._demo_image
            self.gui_status.content = "**demo:** requesting video from server…"
            video_path = self._request_to_path(name, image_bytes)
            self._clips.append(video_path)
            self._count = len(self._clips)
            self.gui_count.content = f"**clips downloaded:** {self._count}"
            # Auto-initialize: first clip inits the scene, later clips append.
            if not bool(getattr(self.trainer, "_initialized", False)):
                self.gui_status.content = "**demo:** first clip — pose + init (DA3)…"
                self._init_from_video(video_path)
                self._snap_home_to_scene()
            else:
                self.gui_status.content = "**demo:** new clip — append (DA3)…"
                self._append_from_video(video_path)
            auto = bool(self.gui_autotrain.value)
            if auto:
                self._on_resume_training()
            self.gui_status.content = (
                f"**demo:** ready ({self.scene.num_splats:,} splats) — "
                + ("training" if auto else "click Train")
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_status.content = f"**demo:** error — {type(e).__name__}: {e}"
        finally:
            self.gui_request_btn.disabled = False
            self.gui_reset_btn.disabled = False
            self.gui_init_btn.disabled = False
            self._op_lock.release()

    def _on_initialize_click(self) -> None:
        """(Re)build the scene from the clips downloaded from the server so far:
        init on the first clip (DA3 pose + init), append the rest. A clean rebuild —
        replaces any current splats with the full downloaded set. Does not train
        unless the auto-train checkbox is on."""
        clips = [c for c in self._clips if c]
        if not clips:
            self.gui_status.content = (
                "**demo:** no downloaded clips yet — Request video first"
            )
            return
        if not self._op_lock.acquire(blocking=False):
            self.gui_status.content = "**demo:** busy — another operation is running"
            return
        try:
            self.gui_init_btn.disabled = True
            self.gui_request_btn.disabled = True
            self.gui_reset_btn.disabled = True
            self._on_pause_training()
            self.gui_status.content = "**demo:** clearing splats…"
            self.trainer.reset()
            self._publish_cams()   # data is now None → clears frustums
            n = len(clips)
            self.gui_status.content = (
                f"**demo:** initializing from clip 1/{n} — pose + init (DA3)…"
            )
            self._init_from_video(clips[0])
            for i, extra in enumerate(clips[1:], start=2):
                self.gui_status.content = (
                    f"**demo:** appending clip {i}/{n} (DA3)…"
                )
                self._append_from_video(extra)
            self._snap_home_to_scene()
            auto = bool(self.gui_autotrain.value)
            if auto:
                self._on_resume_training()
            self.gui_status.content = (
                f"**demo:** initialized from {n} clip(s) "
                f"({self.scene.num_splats:,} splats) — "
                + ("training" if auto else "click Train")
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui_status.content = f"**demo:** init error — {type(e).__name__}: {e}"
        finally:
            self.gui_init_btn.disabled = False
            self.gui_request_btn.disabled = False
            self.gui_reset_btn.disabled = False
            self._op_lock.release()

    def _on_reset_click(self) -> None:
        """Tear down all splats + cameras and end the server session, so the next
        Request starts a brand-new scene. Keeps the uploaded image + prompt."""
        if not self._op_lock.acquire(blocking=False):
            self.gui_status.content = "**demo:** busy — wait for the current operation"
            return
        try:
            self.gui_reset_btn.disabled = True
            self._on_pause_training()
            self.trainer.reset()
            self._publish_cams()   # data is now None → clears frustums
            self._count = 0
            self._clips = []
            self._session_id = None
            self.gui_session.content = "**session:** none"
            self.gui_count.content = "**clips downloaded:** 0"
            self.gui_splat_count.content = "**splats:** 0"
            self.gui_status.content = (
                "**demo:** reset — upload + Request video to start over"
            )
            self._render_all_clients()
        except Exception as e:
            self.gui_status.content = f"**demo:** reset error — {type(e).__name__}: {e}"
        finally:
            self.gui_reset_btn.disabled = False
            self._op_lock.release()

    # --------------------------------------------------------------------- #
    # Pump + run
    # --------------------------------------------------------------------- #

    def _start_pump(self) -> None:
        self._pump_stop.clear()
        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

    def _pump_loop(self) -> None:
        last_version = -1
        while not self._pump_stop.is_set():
            time.sleep(0.1)
            with self.scene.read() as s:
                version = s.splat_version
                step = s.step
                splats = s.num_splats
            if version != last_version:
                last_version = version
                try:
                    self._render_all_clients()
                except Exception:
                    pass
            ctl = self.control
            if ctl is not None:
                try:
                    self.gui_train_status.content = f"**status:** {ctl.status()}"
                    self.gui_step.content = f"**step:** {int(step):,}"
                    self.gui_splat_count.content = f"**splats:** {int(splats):,}"
                except Exception:
                    continue

    def run(self) -> None:
        host = self.server.get_host()
        port = self.server.get_port()
        print(f"demo viser server listening on http://{host}:{port}")
        try:
            self.server.sleep_forever()
        finally:
            self._pump_stop.set()
            if self._pump_thread is not None:
                self._pump_thread.join(timeout=2.0)
            if self.control is not None:
                try:
                    self.control.stop()
                except Exception:
                    pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Slim Demo viewer: remote video generation → splats, rendered live."
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--out-dir", type=Path, default=Path("vipe_outputs"))
    p.add_argument("--publish-every", type=int, default=25)
    p.add_argument("--max-scale-voxels", type=float, default=2.0)
    p.add_argument("--densify-total-steps", type=int, default=7000)

    # Remote server (the collaborator's session backend).
    p.add_argument(
        "--demo-sequence-url", type=str,
        default="https://8000-01kt5jzg8yj6v9xs6zajyxgm91.cloudspaces.litng.ai",
        help="Base URL (or full .../sequence/generate) of the remote server. The "
             "client appends /sequence/generate automatically.",
    )
    p.add_argument(
        "--demo-token", type=str, default=_default_demo_token(),
        help="Bearer token. Defaults to $LYRA_DEMO_TOKEN, else "
             "lai_server/lyra_token.txt.",
    )
    p.add_argument("--demo-prompt", type=str, default="")
    p.add_argument(
        "--assets-dir", type=str, default=str(_REPO / "assets" / "ours"),
        help="Folder the Demo→'Choose image…' picker scans for .jpg/.png seed images.",
    )

    # Camera trajectory defaults.
    p.add_argument("--demo-resolution", type=str, default="240p",
                   choices=list(_RESOLUTIONS))
    p.add_argument("--demo-trajectory", type=str, default="horizontal_zoom",
                   choices=list(CAMERA_TRAJECTORY_CHOICES))
    p.add_argument("--demo-direction", type=str, default="right",
                   choices=list(_DIRECTIONS))
    p.add_argument("--demo-num-frames", type=int, default=81)
    p.add_argument("--demo-strength", type=float, default=0.5)

    # Reconstruction defaults.
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument(
        "--max-points", type=int, default=1_000_000,
        help="Per-clip splat cap: init subsamples to this and each appended clip "
             "seeds at most this many candidates before dedup. Lower to bound the "
             "total across many clips.",
    )
    p.add_argument(
        "--seed-dedup", type=float, default=3.0,
        help="Seed dedup radius in init-voxel units (1–8). The sequence server's "
             "clips overlap heavily, so 3–5 merges duplicate splats instead of "
             "stacking millions. Live-adjustable in the GUI.",
    )
    p.add_argument("--confidence-quantile", type=float, default=0.6)
    p.add_argument("--remove-sky", type=int, default=1)
    p.add_argument("--sh-max-deg", type=int, default=2)
    p.add_argument("--lpips-weight", type=float, default=0.05)
    p.add_argument("--void-weight", type=float, default=0.5)
    p.add_argument("--densify", type=int, default=1)
    p.add_argument("--mode", choices=("3dgs", "2dgs"), default="3dgs")
    args = p.parse_args()

    scene = SceneState()
    trainer = SplatTrainer(
        output_root=args.out_dir, scene=scene, publish_every=args.publish_every,
        max_points=int(args.max_points),
    )
    trainer.set_scale_clamp(args.max_scale_voxels)
    trainer.set_seed_dedup_multiplier(float(args.seed_dedup))
    trainer.densify_total_steps = int(args.densify_total_steps)
    control = BackgroundTrainingThread(trainer.step)
    control.start()  # paused

    app = DemoApp(
        host=args.host,
        port=args.port,
        scene=scene,
        trainer=trainer,
        control=control,
        defaults=dict(
            server_url=args.demo_sequence_url,
            token=args.demo_token,
            prompt=args.demo_prompt,
            assets_dir=args.assets_dir,
            resolution=args.demo_resolution,
            trajectory=args.demo_trajectory,
            direction=args.demo_direction,
            num_frames=args.demo_num_frames,
            strength=args.demo_strength,
            max_frames=args.max_frames,
            max_points=args.max_points,
            seed_dedup=args.seed_dedup,
            scale_clamp=args.max_scale_voxels,
            confidence_quantile=args.confidence_quantile,
            remove_sky=bool(args.remove_sky),
            sh_max_deg=args.sh_max_deg,
            lpips_weight=args.lpips_weight,
            void_weight=args.void_weight,
            densify=bool(args.densify),
            mode=args.mode,
        ),
    )
    try:
        app.run()
    finally:
        if getattr(trainer, "_initialized", False):
            try:
                trainer.save_current("splats.ply")
                print("saved splats.ply")
            except Exception as e:
                print(f"final-save skipped: {e}")


if __name__ == "__main__":
    main()
