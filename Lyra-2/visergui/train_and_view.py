"""
Live-training composition root: boot the viser viewer empty, then init
training from the browser.

Usage:
    python visergui/train_and_view.py [--video PATH] [--port 8080]

The viewer boots with an empty scene. The Training tab has a Setup folder
with: video path, max_frames, confidence_quantile, remove_sky, Initialize,
and Reset. Clicking Initialize runs `process_video` (DA3 inference) and
`initialize_gaussians`; the trainer is then ready and the user can click
`resume` to start the optimization loop. Reset tears the trainer down so
you can re-init with different parameters.

Remote use:
  On GPU box:  python visergui/train_and_view.py --port 8080
  On laptop:   ssh -L 8080:localhost:8080 user@gpubox
  Browser:     http://localhost:8080
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splat_trainer import SplatTrainer
from training import BackgroundTrainingThread
from viewer import SceneState, ViewerApp


def _default_demo_token() -> str:
    """Bearer token for the sequence backend: $LYRA_DEMO_TOKEN, else the contents
    of lai_server/lyra_token.txt (repo-root sibling), else empty."""
    tok = os.environ.get("LYRA_DEMO_TOKEN", "").strip()
    if tok:
        return tok
    token_file = Path(__file__).resolve().parent.parent / "lai_server" / "lyra_token.txt"
    try:
        return token_file.read_text().strip()
    except Exception:
        return ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Empty-boot live splat trainer + viser viewer."
    )
    p.add_argument(
        "--video", type=Path,
        default=Path("assets/ours/museum_fwd.mp4"),
        help="Default value populated into the Setup→video field at boot.",
    )
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("vipe_outputs"))
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--confidence_quantile", type=float, default=0.6)
    p.add_argument("--remove-sky", type=int, default=1)
    p.add_argument("--publish-every", type=int, default=25,
                   help="Push trainer state into the scene every N steps.")
    p.add_argument("--sh-max-deg", type=int, default=2,
                   help="Default for the Setup→sh_max_deg field. 0 = L1-only (v1).")
    p.add_argument("--lpips-weight", type=float, default=0.05,
                   help="Default for the Setup→lpips_weight field. 0 disables (v1).")
    p.add_argument("--void-weight", type=float, default=0.5,
                   help="Default for the Setup→void_weight slider. Penalises "
                        "rendered alpha inside ~train_mask. 0 disables.")
    p.add_argument("--max-scale-voxels", type=float, default=2.0,
                   help="Default for the Setup→max_scale_voxels slider (live-adjustable).")
    p.add_argument("--densify", type=int, default=1,
                   help="Default for the Setup→densify checkbox (1 = on, 0 = off).")
    p.add_argument("--densify-total-steps", type=int, default=7000,
                   help="Total steps the densify schedule plans for.")
    p.add_argument("--mode", choices=("3dgs", "2dgs"), default="3dgs",
                   help="Default for the Setup→mode dropdown.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--demo-server-url", type=str, default="http://localhost:8000/generate",
        help="Default endpoint for the Demo tab's video-generation server "
             "(Lyra2 demo_server, default port 8000). Editable live in the "
             "GUI. POST image+prompt → mp4.",
    )
    p.add_argument(
        "--demo-backend", type=str, default="local", choices=["local", "sequence"],
        help="Default Demo→backend. 'local' = our demo_server (/generate). "
             "'sequence' = a collaborator's session server (/sequence/generate, "
             "Bearer token, server-side continuity).",
    )
    p.add_argument(
        "--demo-token", type=str, default=_default_demo_token(),
        help="Default Bearer token for the sequence backend. Defaults to "
             "$LYRA_DEMO_TOKEN, else the contents of lai_server/lyra_token.txt. "
             "Ignored by the local backend.",
    )
    p.add_argument(
        "--demo-sequence-url", type=str,
        default="https://8000-01kt5jzg8yj6v9xs6zajyxgm91.cloudspaces.litng.ai",
        help="Default server URL pre-filled when Demo→backend=sequence "
             "(the collaborator's base URL or .../sequence/generate). The client "
             "appends /sequence/generate automatically.",
    )
    p.add_argument(
        "--demo-prompt", type=str, default="",
        help="Default text populated into the Demo→prompt field at boot. "
             "Optional — the server uses a generic caption when empty.",
    )
    p.add_argument(
        "--demo-max-frames", type=int, default=None,
        help="Default for the Demo→max_frames field. Falls back to "
             "--max-frames when unset.",
    )
    # Lyra2 single-trajectory defaults (Demo→'Lyra2 camera' folder).
    p.add_argument(
        "--demo-resolution", type=str, default="240p",
        help="Default Demo→resolution: a preset label (480p/360p/320p/240p) "
             "or a raw 'H,W'. 480p is the model's native size; 240p is faster.",
    )
    p.add_argument(
        "--demo-trajectory", type=str, default="horizontal_zoom",
        help="Default Demo→trajectory (one camera move per request), e.g. "
             "horizontal_zoom, horizontal, orbit_horizontal, spiral, "
             "dolly_zoom, rotate_spot, back, original.",
    )
    p.add_argument(
        "--demo-direction", type=str, default="right",
        choices=["left", "right", "up", "down"],
        help="Default Demo→direction of the camera move.",
    )
    p.add_argument(
        "--demo-num-frames", type=int, default=81,
        help="Default Demo→num_frames. Must be 1 + 80k (81, 161, 241, …).",
    )
    p.add_argument(
        "--demo-strength", type=float, default=0.5,
        help="Default Demo→strength (move magnitude; distance for dolly/strafe, "
             "angle for orbits).",
    )
    p.add_argument(
        "--inpaint-preload", type=int, default=0,
        help="0 (default) = do NOT auto-load the inpainter's diffusers "
             "pipeline; use the Inpaint tab's 'Load model' button (or the "
             "first Inpaint click) to load it on demand. The default model "
             "(FLUX-2-Klein 9B) is ~25 GB on GPU and OOMs alongside "
             "training, which is why autoload is off. Pass 1 to restore the "
             "old behavior: warm it in a daemon thread after the first "
             "Initialize (DA3 must load first).",
    )
    args = p.parse_args()

    scene = SceneState()
    trainer = SplatTrainer(
        output_root=args.out_dir,
        scene=scene,
        publish_every=args.publish_every,
    )
    # Sync the trainer's scale-clamp default with the CLI flag so the GUI
    # slider's initial value matches the trainer's stored value.
    trainer.set_scale_clamp(args.max_scale_voxels)
    # Start the daemon thread early so resume is responsive once init lands.
    # `step()` is guarded: it short-circuits until `setup_refine` runs.
    control = BackgroundTrainingThread(trainer.step)
    control.start()  # paused

    def initializer(opts: dict) -> None:
        video = Path(opts["video"]).expanduser()
        if not video.exists():
            raise SystemExit(f"video not found: {video}")
        trainer.prepare_and_init(
            video=video,
            max_frames=int(opts["max_frames"]),
            confidence=float(opts["confidence_quantile"]),
            remove_sky=bool(opts["remove_sky"]),
            name=args.name,
            sh_max_deg=int(opts.get("sh_max_deg", 0)),
            lpips_weight=float(opts.get("lpips_weight", 0.0)),
            void_weight=float(opts.get("void_weight", 0.5)),
            use_densify=bool(opts.get("use_densify", False)),
            densify_total_steps=int(args.densify_total_steps),
            mode=str(opts.get("mode", "3dgs")),
        )
        # Reuses _republish_cams below (defined after, but Python closures
        # resolve at call time so this works as long as initializer is
        # called after main() has finished setting up). On the empty-boot
        # path that's true: initializer fires from the GUI button.
        _republish_cams()
        # Now that DA3 has loaded cleanly inside prepare_and_init, it's
        # safe to start the inpainter preload. Doing this earlier put
        # accelerate / from_pretrained into a state where DA3's later
        # construction left params on meta and `.to(device)` blew up.
        # start_preload() is idempotent so re-init won't double-spawn.
        inp = getattr(app, "inpainter", None)
        if inp is not None:
            try:
                inp.start_preload()
            except Exception as e:
                print(f"initializer: inpaint preload kick failed: {e}")

    def resetter() -> None:
        import numpy as _np
        control.pause()
        trainer.reset()
        app.publish_training_cameras(
            c2w=_np.zeros((0, 4, 4)), K=_np.zeros((0, 3, 3)),
            images=None, H=1, W=1,
        )
        # Clear the Inpaint tab's captured screenshot / mask / neighbors /
        # result images so they don't lie about which scene they came
        # from. Preserves the cached diffusers pipeline (preload survives).
        inpainter = getattr(app, "inpainter", None)
        if inpainter is not None:
            try:
                inpainter.reset()
            except Exception as e:
                print(f"resetter: inpainter.reset() failed: {e}")

    # Phase 5.1: palette for coloring training-camera frustums by epoch.
    # Wraps modulo so arbitrary epoch counts are supported. Epoch 0 keeps
    # the historical orange the viewer used before per-camera colors.
    _EPOCH_PALETTE = (
        (255, 153,  51),   # 0 — orange (legacy default)
        ( 51, 204, 204),   # 1 — teal
        (204,  51, 204),   # 2 — magenta
        (102, 204,  51),   # 3 — green
        (255, 204,  51),   # 4 — yellow
        ( 51, 153, 255),   # 5 — blue
        (255,  85,  85),   # 6 — coral
        (170, 102, 255),   # 7 — purple
    )

    def _epoch_colors(epochs):
        import numpy as _np
        arr = _np.zeros((len(epochs), 3), dtype=_np.uint8)
        for i, e in enumerate(epochs):
            arr[i] = _EPOCH_PALETTE[int(e) % len(_EPOCH_PALETTE)]
        return arr

    def _republish_cams() -> None:
        d = trainer.data
        if d is None:
            return
        epochs = d.frame_epoch or [0] * int(d.N)
        cols = _epoch_colors(epochs)
        app.publish_training_cameras(
            c2w=d.c2w.detach().cpu().numpy(),
            K=d.K.detach().cpu().numpy(),
            images=d.rgb.detach().cpu().numpy(),
            H=int(d.H), W=int(d.W),
            colors=cols,
        )

    def save_checkpoint(path: str) -> dict:
        return trainer.save_checkpoint(path)

    def load_checkpoint(path: str) -> dict:
        info = trainer.load_checkpoint(path)
        _republish_cams()
        return info

    def append_video(path: str, max_frames: int, seed_new_splats: bool) -> dict:
        info = trainer.append_video(
            path, max_frames=max_frames, seed_new_splats=seed_new_splats,
        )
        _republish_cams()
        return info

    def set_sampling(mode: str, new_frame_weight: float, horizon: int) -> dict:
        info = trainer.set_sampling(
            mode=mode,
            new_frame_weight=new_frame_weight,
            horizon=horizon,
        )
        info["epoch_counts"] = trainer.epoch_frame_counts()
        return info

    def set_freeze_mode(mode: str) -> dict:
        return trainer.set_freeze_mode(mode)

    def recompute_freeze_mask() -> dict:
        return trainer.recompute_freeze_mask()

    def compute_voxel_overlap(voxel_mult: float) -> dict:
        return trainer.compute_voxel_overlap_layer(voxel_mult=voxel_mult)

    def compute_coverage(scope: str) -> dict:
        return trainer.compute_coverage_layer(camera_scope=scope)

    def append_frames(directory: str, seed_new_splats: bool) -> dict:
        from pathlib import Path as _P
        import json as _json
        import numpy as _np
        from PIL import Image as _Image
        d = _P(directory).expanduser()
        if not d.is_dir():
            raise FileNotFoundError(f"frames directory not found: {d}")
        cams = sorted(d.glob("cam_*.json"))
        frames = []
        for cam_path in cams:
            stem = cam_path.stem  # cam_NNNN
            idx = stem.split("_", 1)[1] if "_" in stem else stem
            frame_path = d / f"frame_{idx}.png"
            if not frame_path.exists():
                print(f"append_frames: no image for {cam_path.name}; skipping")
                continue
            meta = _json.loads(cam_path.read_text())
            K = _np.asarray(meta["K"], dtype=_np.float32)
            c2w = _np.asarray(meta["c2w"], dtype=_np.float32)
            rgb = _np.asarray(_Image.open(frame_path).convert("RGB"), dtype=_np.uint8)
            depth = None
            if "depth" in meta and meta["depth"]:
                dp = (d / meta["depth"]).resolve()
                depth = _np.load(str(dp)).astype(_np.float32)
            if depth is None:
                # No supplied depth → ground it against the CURRENT splats so the
                # inpainted/disoccluded region lands in the scene frame instead of
                # the camera origin (raw splat depth is 0 where alpha=0). Mirrors
                # the inpainter "Add frame" path; leaves depth None on failure so
                # append_supplied_frames skips the frame rather than misplacing it.
                try:
                    import splat_trainer as _st
                    Hs, Ws = rgb.shape[:2]
                    _, alpha_np, splat_depth = trainer.render_rgbd_at(c2w, K, Hs, Ws)
                    depth, ginfo = _st.ground_inpaint_depth(
                        rgb, splat_depth, alpha_np,
                        device=trainer.train.params["means"].device)
                    print(f"append_frames: grounded {frame_path.name} depth — "
                          f"s_overlap={ginfo['s_overlap']:.3f} hole={ginfo['hole_frac']:.0%}")
                except Exception as e:
                    print(f"append_frames: depth grounding failed for {frame_path.name} "
                          f"({e}); frame may be skipped (no depth)")
            frames.append({"rgb": rgb, "K": K, "c2w": c2w, "depth": depth})
        info = trainer.append_supplied_frames(frames, seed_new_splats=seed_new_splats)
        _republish_cams()
        return info

    # Demo tab: synchronous call to the generation server, then write the
    # returned video to disk and hand the path back to the GUI, which runs
    # the same init/append pipeline as the Train tab.
    demo_dir = args.out_dir / "demo"
    _demo_counter = {"n": 0}

    def request_demo_video(image_bytes: bytes, image_name: str, prompt: str,
                           server_url: str, gen_opts: dict | None = None) -> Path:
        import video_api
        video_bytes = video_api.request_video(
            server_url or args.demo_server_url, image_bytes, image_name, prompt,
            gen_opts=gen_opts,
        )
        demo_dir.mkdir(parents=True, exist_ok=True)
        _demo_counter["n"] += 1
        out_path = demo_dir / f"clip_{_demo_counter['n']:04d}.mp4"
        out_path.write_bytes(video_bytes)
        print(f"demo: server returned {len(video_bytes):,} bytes → {out_path}")
        return out_path

    app = ViewerApp(
        ply_path=None,
        host=args.host,
        port=args.port,
        scene=scene,
        training_control=control,
        initializer=initializer,
        resetter=resetter,
        on_scale_mult_change=trainer.set_scale_clamp,
        trainer=trainer,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
        append_video=append_video,
        append_frames=append_frames,
        set_sampling=set_sampling,
        set_freeze_mode=set_freeze_mode,
        recompute_freeze_mask=recompute_freeze_mask,
        on_seed_dedup_mult_change=trainer.set_seed_dedup_multiplier,
        compute_voxel_overlap=compute_voxel_overlap,
        compute_coverage=compute_coverage,
        inpaint_preload=bool(args.inpaint_preload),
        request_video=request_demo_video,
        demo_defaults=dict(
            backend=args.demo_backend,
            token=args.demo_token,
            server_url=(args.demo_sequence_url
                        if (args.demo_backend == "sequence" and args.demo_sequence_url)
                        else args.demo_server_url),
            prompt=args.demo_prompt,
            max_frames=(args.demo_max_frames
                        if args.demo_max_frames is not None
                        else args.max_frames),
            resolution=args.demo_resolution,
            trajectory=args.demo_trajectory,
            direction=args.demo_direction,
            num_frames=args.demo_num_frames,
            strength=args.demo_strength,
        ),
        default_init_args=dict(
            video=str(args.video),
            max_frames=args.max_frames,
            confidence_quantile=args.confidence_quantile,
            remove_sky=bool(args.remove_sky),
            sh_max_deg=args.sh_max_deg,
            lpips_weight=args.lpips_weight,
            void_weight=args.void_weight,
            scale_clamp_voxel_mult=args.max_scale_voxels,
            use_densify=bool(args.densify),
            mode=args.mode,
        ),
        derive_splat_points=False,  # live splats invalidate the derived layer
    )
    try:
        app.run()
    finally:
        # app.run()'s finally already stops control; save final PLY if we
        # actually ran any training.
        if getattr(trainer, "_initialized", False):
            try:
                trainer.save_current("splats.ply")
                print("saved splats.ply")
            except Exception as e:
                print(f"final-save skipped: {e}")


if __name__ == "__main__":
    main()
