"""
Segmenter panel — segment a single object across every training frame.

Foundation feature (produce + save masks only; no loss/removal wiring):
- Pick an object by CLICKING it in the live 3D view. The clicked viewport
  pixel is turned into a 3D world point by rendering the splat depth at the
  current camera and back-projecting it.
- Project that point into the best-seeing training frame, seed SAM 2 there,
  and propagate the mask forward + backward across the ordered training
  frames (SAM 2 *video* predictor) → one binary mask per frame.
- Store `trainer.object_masks` (N, H, W) bool, save PNGs, show a montage.

Mirrors `InpainterPanel`'s structure (server / trainer_ref / viewer refs,
`_build_gui()`, `on_click` callbacks) and reuses its camera + projection +
preview helpers. The viewer instantiates it once during boot:

    from segmenter import SegmenterPanel
    self.segmenter = SegmenterPanel(server=self.server,
                                    trainer_ref=trainer, viewer=self)
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import viser

# Reuse existing helpers rather than reimplementing them.
from viewer import viser_camera_to_opencv_viewmat, PointCloudLayer
from inpainter import _thumbnail, _tile_images, _overlay_mask
from multiview_mask import (
    _get_predictor,
    _get_video_predictor,
    _backproject_to_world,
    _project_world_to_pixel,
)

if TYPE_CHECKING:
    from splat_trainer import SplatTrainer
    from viewer import ViewerApp


class SegmenterPanel:
    """All state + GUI for single-object multi-frame segmentation. One per viewer."""

    def __init__(
        self,
        server: viser.ViserServer,
        trainer_ref: "SplatTrainer | None" = None,
        viewer: "ViewerApp | None" = None,
        out_root: Path | None = None,
    ) -> None:
        self.server = server
        self._trainer_ref = trainer_ref
        self._viewer = viewer
        self._out_root = Path(out_root) if out_root is not None else Path("segment_outputs")

        # State
        self.object_masks: torch.Tensor | None = None   # (N, H, W) bool
        self._seed: dict | None = None                    # {'P','frame_idx','uv'}
        self._pointer_armed = False

        self._build_gui()

    # ------------------------------------------------------------------ GUI

    def _build_gui(self) -> None:
        placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.server.gui.add_folder("Segment Object"):
            self.gui_status = self.server.gui.add_markdown(
                "_Pick an object, then segment it across all frames._"
            )
            self.gui_pick_btn = self.server.gui.add_button(
                "Pick Object (click in 3D view)",
                hint="Arms a one-shot click handler. Click the object in the "
                     "live 3D viewport; the clicked point is back-projected to "
                     "3D and SAM 2 previews the mask on the best training frame.",
            )
            self.gui_seed_preview = self.server.gui.add_image(
                placeholder, label="seed-frame mask"
            )
            self.gui_segment_btn = self.server.gui.add_button(
                "Segment All Frames",
                hint="Propagate the picked object across every training frame "
                     "with the SAM 2 video predictor, then save the per-frame "
                     "masks and show a montage.",
            )
            self.gui_montage = self.server.gui.add_image(
                placeholder, label="per-frame masks"
            )
            self.gui_select_btn = self.server.gui.add_button(
                "Select Object Splats",
                hint="Find the 3D gaussian splats that make up the segmented "
                     "object (front-surface multi-view vote) and highlight them "
                     "as a yellow overlay. Stores trainer.selected_splat_mask.",
            )
            self.gui_additive = self.server.gui.add_checkbox(
                "add to current selection",
                initial_value=True,
                hint="On: each 'Select Object Splats' UNIONs the object's splats "
                     "into the running selection — segment + select several "
                     "objects in turn. Off: replace the selection each time.",
            )
            self.gui_select_status = self.server.gui.add_markdown(
                "_no selection yet_"
            )
            self.gui_reset_btn = self.server.gui.add_button(
                "Reset Selection",
                hint="Clear the accumulated splat selection and remove the "
                     "yellow overlay.",
            )

        self.gui_pick_btn.on_click(lambda _ev: self._on_pick_click())
        self.gui_segment_btn.on_click(lambda _ev: self._on_segment_click())
        self.gui_select_btn.on_click(lambda _ev: self._on_select_click())
        self.gui_reset_btn.on_click(lambda _ev: self._on_reset_click())

    # ------------------------------------------------------------- helpers

    def _out_dir(self) -> Path:
        """Resolve the output directory based on the trainer's name."""
        name = getattr(self._trainer_ref, "name", None) or "default"
        return self._out_root / name

    def _camera_from_client(self, cam, d):
        """Build (w2c_np, c2w_np, K_np, W, H) from a viser camera, matching the
        inpainter's capture convention (square pixels, vfov from cam.fov)."""
        w2c_np = viser_camera_to_opencv_viewmat(cam.position, cam.wxyz)
        c2w_np = np.linalg.inv(w2c_np)
        try:
            W = int(cam.image_width); H = int(cam.image_height)
            if W <= 0 or H <= 0:
                raise ValueError
        except (TypeError, ValueError):
            H, W = int(d.H), int(d.W)
        # Cap to training resolution, preserve aspect.
        max_side = max(int(d.H), int(d.W))
        if max(H, W) > max_side:
            s = max_side / max(H, W)
            H = max(1, int(round(H * s)))
            W = max(1, int(round(W * s)))
        vfov = float(getattr(cam, "fov", 1.047))  # radians; ~60° fallback
        fy = H / (2.0 * float(np.tan(vfov * 0.5)))
        fx = fy
        K_np = np.array([[fx, 0.0, W / 2.0],
                         [0.0, fy, H / 2.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
        return w2c_np, c2w_np, K_np, W, H

    def _frame_rgb_u8(self, d, i: int) -> np.ndarray:
        """Training frame i as (H, W, 3) uint8 RGB."""
        f = d.rgb[i].detach().cpu().numpy()  # (H, W, 3) float [0,1]
        return (f * 255).clip(0, 255).astype(np.uint8)

    def _select_seed_frame(self, P: np.ndarray, d):
        """Pick the training frame that best sees world point P.

        Returns (frame_idx, u, v) for the in-bounds frame whose camera is
        closest to P (largest apparent size, most likely unoccluded), or None.
        """
        c2w = d.c2w.detach().cpu().numpy()      # (N, 4, 4)
        K = d.K.detach().cpu().numpy()          # (N, 3, 3)
        H, W = int(d.H), int(d.W)
        best = None
        best_dist = float("inf")
        for i in range(int(d.N)):
            u, v, z = _project_world_to_pixel(P, K[i], c2w[i])
            if not (z > 0) or np.isnan(u):
                continue
            if not (0 <= u < W and 0 <= v < H):
                continue
            dist = float(np.linalg.norm(c2w[i][:3, 3] - P))
            if dist < best_dist:
                best_dist = dist
                best = (i, float(u), float(v))
        return best

    def _sam_image_mask(self, rgb_u8: np.ndarray, u: float, v: float, device) -> np.ndarray:
        """Single-frame SAM 2 mask from a positive point prompt → bool (H, W)."""
        predictor = _get_predictor(device=str(device))
        with torch.inference_mode():
            predictor.set_image(rgb_u8)
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[u, v]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    # --------------------------------------------------------------- pick

    def _on_pick_click(self) -> None:
        """Arm a one-shot scene click handler to pick the object."""
        try:
            t = self._trainer_ref
            if t is None or getattr(t, "data", None) is None or getattr(t, "train", None) is None:
                self.gui_status.content = "_Init training first (no splat model)._"
                return
            if self._pointer_armed:
                self.gui_status.content = "_Already waiting for a click — click the object._"
                return

            self._pointer_armed = True
            self.gui_status.content = "_Click the object in the 3D viewport…_"

            @self.server.scene.on_pointer_event(event_type="click")
            def _handle(event: viser.ScenePointerEvent) -> None:
                try:
                    self._handle_click(event)
                finally:
                    # One-shot: disarm regardless of outcome.
                    self.server.scene.remove_pointer_callback()
                    self._pointer_armed = False

        except Exception as e:
            import traceback
            self._pointer_armed = False
            self.gui_status.content = f"_Pick error: {e}_"
            traceback.print_exc()

    def _handle_click(self, event: "viser.ScenePointerEvent") -> None:
        """Turn a 3D-view click into a world point P and preview the seed mask."""
        t = self._trainer_ref
        d = t.data
        device = t.means_t.device

        cam = event.client.camera
        w2c_np, c2w_np, K_np, W, H = self._camera_from_client(cam, d)

        # Render splat depth at this camera and sample at the clicked pixel.
        _, alpha, depth = t.render_rgbd_at(c2w_np, K_np, H, W)
        sx, sy = event.screen_pos[0]            # normalized [0,1] over the viewport
        u = int(round(sx * (W - 1)))
        v = int(round(sy * (H - 1)))
        u = max(0, min(W - 1, u)); v = max(0, min(H - 1, v))
        z = float(depth[v, u])
        if not (z > 0) or float(alpha[v, u]) < 0.5:
            self.gui_status.content = (
                "_Clicked empty space (no splat surface there) — aim at the "
                "object and click again._"
            )
            return

        P = _backproject_to_world(u, v, z, K_np, c2w_np)

        seed = self._select_seed_frame(P, d)
        if seed is None:
            self.gui_status.content = (
                "_Picked a 3D point, but no training frame sees it. Try a point "
                "more central to the scene._"
            )
            return
        fi, su, sv = seed

        # Preview the SAM 2 mask on the seed frame so the user can confirm.
        rgb_seed = self._frame_rgb_u8(d, fi)
        mask = self._sam_image_mask(rgb_seed, su, sv, device)
        self._seed = {"P": P, "frame_idx": fi, "uv": (su, sv)}

        overlay = _overlay_mask(rgb_seed, mask)
        self.gui_seed_preview.image = _thumbnail(overlay, max_side=384)
        cov = 100.0 * float(mask.mean())
        self.gui_status.content = (
            f"_Picked object → seed frame {fi} (mask {cov:.1f}% of frame). "
            f"Click 'Segment All Frames' to propagate._"
        )
        print(f"  segmenter: picked P={P.tolist()} seed_frame={fi} uv=({su:.0f},{sv:.0f})",
              file=sys.stderr, flush=True)

    # ------------------------------------------------------------ segment

    def _on_segment_click(self) -> None:
        """Propagate the picked object across every training frame and save."""
        tmpdir = None
        try:
            t = self._trainer_ref
            if t is None or getattr(t, "data", None) is None:
                self.gui_status.content = "_Init training first._"
                return
            if self._seed is None:
                self.gui_status.content = "_Pick an object first (click in the 3D view)._"
                return

            d = t.data
            device = t.means_t.device
            N, H, W = int(d.N), int(d.H), int(d.W)
            seed_i = int(self._seed["frame_idx"])
            su, sv = self._seed["uv"]

            self.gui_status.content = f"_Dumping {N} frames + loading SAM 2 video…_"
            t0 = time.perf_counter()

            # 1. Dump training frames to a temp dir of zero-padded JPEGs.
            from PIL import Image
            tmpdir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
            for i in range(N):
                Image.fromarray(self._frame_rgb_u8(d, i)).save(
                    str(tmpdir / f"{i:05d}.jpg"), quality=95
                )

            # 2. Init video predictor, seed at the picked frame, propagate both ways.
            predictor = _get_video_predictor(device=str(device))
            masks = np.zeros((N, H, W), dtype=bool)
            autocast = torch.autocast(
                device_type="cuda" if str(device).startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
            )
            with torch.inference_mode(), autocast:
                state = predictor.init_state(
                    video_path=str(tmpdir), offload_video_to_cpu=True
                )
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=seed_i,
                    obj_id=1,
                    points=np.array([[su, sv]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
                for reverse in (False, True):
                    for fidx, _obj_ids, logits in predictor.propagate_in_video(
                        state, reverse=reverse
                    ):
                        m = (logits[0, 0] > 0.0).cpu().numpy()
                        if m.shape != (H, W):
                            m = np.asarray(
                                Image.fromarray(m.astype(np.uint8) * 255).resize(
                                    (W, H), Image.NEAREST
                                )
                            ) > 127
                        masks[fidx] = m

            dt = time.perf_counter() - t0

            # 3. Store on the trainer + save PNGs + montage.
            self.object_masks = torch.from_numpy(masks)
            setattr(t, "object_masks", self.object_masks)

            out_dir = self._out_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(N):
                Image.fromarray((masks[i].astype(np.uint8) * 255)).save(
                    str(out_dir / f"objmask_{i:04d}.png")
                )

            overlays = [
                _overlay_mask(self._frame_rgb_u8(d, i), masks[i]) for i in range(N)
            ]
            self.gui_montage.image = _thumbnail(_tile_images(overlays), max_side=512)

            n_nonempty = int((masks.reshape(N, -1).any(axis=1)).sum())
            cov = 100.0 * float(masks.mean())
            self.gui_status.content = (
                f"_Segmented object across {n_nonempty}/{N} frames "
                f"({cov:.1f}% pixels) in {dt:.1f}s._  \n"
                f"_Saved → `{out_dir.resolve()}/objmask_*.png`; "
                f"`trainer.object_masks` set ({N}×{H}×{W}). "
                f"Note: stale if frames are appended later._"
            )
            print(f"  segmenter: masks {n_nonempty}/{N} frames → {out_dir.resolve()}",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_status.content = f"_Segment error: {e}_"
            traceback.print_exc()
        finally:
            if tmpdir is not None:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------- select

    def _on_select_click(self) -> None:
        """Select the 3D splats making up the object and highlight them."""
        try:
            t = self._trainer_ref
            if t is None or getattr(t, "train", None) is None:
                self.gui_select_status.content = "_Init training first._"
                return
            masks = self.object_masks
            if masks is None:
                masks = getattr(t, "object_masks", None)
            if masks is None:
                self.gui_select_status.content = (
                    "_Segment an object first (Pick → Segment All Frames)._"
                )
                return

            self.gui_select_status.content = "_Selecting splats…_"
            t0 = time.perf_counter()
            stats = t.select_splats_by_masks(
                masks, accumulate=bool(self.gui_additive.value))
            dt = time.perf_counter() - t0

            sel = t.selected_splat_mask
            n_sel = int(stats["n_selected"])
            n_new = int(stats.get("n_new", n_sel))
            M = int(stats["M"])

            if n_sel == 0:
                # Clear any prior overlay and report.
                if self._viewer is not None:
                    self._viewer._teardown_debug_layer("selected_object")
                self.gui_select_status.content = (
                    f"_Selected 0 / {M} splats — try lowering the vote "
                    f"fraction / depth tolerance in `select_splats_by_masks`._"
                )
                return

            all_means = t.train.params["means"].detach().cpu().numpy().astype(np.float32)
            pts = all_means[sel.detach().cpu().numpy()]
            # Scale the point size to the scene's bbox diagonal so the highlight
            # is visible at whatever world scale the splats live in — a fixed
            # size vanishes at DA3 / COLMAP scales (mirrors
            # derive_splat_centers_layer). Slightly larger than the base
            # splat_centers layer (5e-3) so the selection stands out.
            diag = float(np.linalg.norm(all_means.max(axis=0) - all_means.min(axis=0)))
            pt_size = max(diag * 1.2e-2, 1e-4)
            layer = PointCloudLayer(
                name="selected_object",
                points=pts,
                colors_rgb=np.zeros((len(pts), 3), dtype=np.uint8),
                color_mode="uniform",
                uniform_color=(255, 255, 0),
                point_size=pt_size,
            )
            mode_note = ""
            if self._viewer is not None:
                self._viewer._add_or_refresh_debug_layer(layer)
                # Point-cloud overlays only render in 'points' display mode
                # (splats mode removes all point clouds), so switch to it
                # automatically — otherwise the highlight is silently hidden.
                try:
                    if (getattr(self._viewer, "gui_display", None) is not None
                            and self._viewer.display_mode != "points"):
                        self._viewer.gui_display.value = "points"
                        self._viewer._apply_display_mode()
                        mode_note = " Switched Display → 'points' to show it."
                except Exception:
                    import traceback
                    traceback.print_exc()

            added_note = f"+{n_new} new → " if bool(self.gui_additive.value) else ""
            self.gui_select_status.content = (
                f"_{added_note}{n_sel} / {M} splats selected "
                f"({100.0 * n_sel / max(M,1):.1f}%) in {dt:.1f}s — yellow overlay "
                f"(size {pt_size:.4f}).{mode_note}_  \n"
                f"_Overlays show in the **points** Display mode only; switch back "
                f"to 'splats' for the full render. `trainer.selected_splat_mask` set._"
            )
            print(f"  segmenter: +{n_new} new, total {n_sel}/{M} splats, "
                  f"pt_size={pt_size:.4f}", file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_select_status.content = f"_Select error: {e}_"
            traceback.print_exc()

    def _on_reset_click(self) -> None:
        """Clear the accumulated splat selection and remove the overlay."""
        try:
            t = self._trainer_ref
            if t is not None:
                t.selected_splat_mask = None
            if self._viewer is not None:
                self._viewer._teardown_debug_layer("selected_object")
            self.gui_select_status.content = "_Selection cleared._"
            print("  segmenter: selection reset", file=sys.stderr, flush=True)
        except Exception as e:
            import traceback
            self.gui_select_status.content = f"_Reset error: {e}_"
            traceback.print_exc()
