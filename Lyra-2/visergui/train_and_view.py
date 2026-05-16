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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splat_trainer import SplatTrainer
from training import BackgroundTrainingThread
from viewer import SceneState, ViewerApp


def main() -> None:
    p = argparse.ArgumentParser(
        description="Empty-boot live splat trainer + viser viewer."
    )
    p.add_argument(
        "--video", type=Path,
        default=Path("outputs/zoomgs/videos/14.mp4"),
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
    p.add_argument("--max-scale-voxels", type=float, default=2.0,
                   help="Default for the Setup→max_scale_voxels slider (live-adjustable).")
    p.add_argument("--densify", type=int, default=1,
                   help="Default for the Setup→densify checkbox (1 = on, 0 = off).")
    p.add_argument("--densify-total-steps", type=int, default=7000,
                   help="Total steps the densify schedule plans for.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
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
            use_densify=bool(opts.get("use_densify", False)),
            densify_total_steps=int(args.densify_total_steps),
        )
        d = trainer.data
        if d is None:
            return
        app.publish_training_cameras(
            c2w=d.c2w.detach().cpu().numpy(),
            K=d.K.detach().cpu().numpy(),
            images=d.rgb.detach().cpu().numpy(),
            H=int(d.H), W=int(d.W),
        )

    def resetter() -> None:
        import numpy as _np
        control.pause()
        trainer.reset()
        app.publish_training_cameras(
            c2w=_np.zeros((0, 4, 4)), K=_np.zeros((0, 3, 3)),
            images=None, H=1, W=1,
        )

    app = ViewerApp(
        ply_path=None,
        host=args.host,
        port=args.port,
        scene=scene,
        training_control=control,
        initializer=initializer,
        resetter=resetter,
        on_scale_mult_change=trainer.set_scale_clamp,
        default_init_args=dict(
            video=str(args.video),
            max_frames=args.max_frames,
            confidence_quantile=args.confidence_quantile,
            remove_sky=bool(args.remove_sky),
            sh_max_deg=args.sh_max_deg,
            lpips_weight=args.lpips_weight,
            scale_clamp_voxel_mult=args.max_scale_voxels,
            use_densify=bool(args.densify),
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
