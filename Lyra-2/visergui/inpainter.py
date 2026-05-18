"""
Inpainter panel — scene editing via inpaint → depth → splat injection.

Phased build (each phase is independently testable):
- Phase 1: Capture current viser view → splat render → thumbnail + PNG  ← IMPLEMENTED
- Phase 2a: Disocclusion mask + nearest training-frame selection         ← TODO
- Phase 2b: Multi-view-conditioned inpaint (FLUX-Kontext)                ← TODO
- Phase 2c: Save + display                                                ← TODO
- Phase 3: DepthAnything-3 on the inpainted image                         ← TODO
- Phase 4: Unproject NEW-region pixels → inject as gaussians              ← TODO
- Phase 5: Append the view to trainer.data + continue optimization        ← TODO

Layout: a single `InpainterPanel` class owns all state + GUI. The viewer
instantiates it once during boot:

    from inpainter import InpainterPanel
    self.inpainter = InpainterPanel(server=self.server, trainer_ref=trainer,
                                    viewer=self)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import viser

# Reuse the existing viser→OpenCV w2c helper
from viewer import viser_camera_to_opencv_viewmat

if TYPE_CHECKING:
    from splat_trainer import SplatTrainer
    from viewer import ViewerApp


class InpainterPanel:
    """All state + GUI for the inpaint pipeline. One per viewer."""

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
        self._out_root = Path(out_root) if out_root is not None else Path("inpaint_outputs")

        # Phase 1 state
        self._capture_counter = 0
        self.last_screenshot: dict | None = None  # {'rgb','alpha','K','c2w','H','W','path'}

        self._build_gui()

    # ------------------------------------------------------------------ GUI

    def _build_gui(self) -> None:
        placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.server.gui.add_folder("Inpaint"):
            # ---- Phase 1: capture
            self.gui_capture_btn = self.server.gui.add_button("Capture View")
            self.gui_status = self.server.gui.add_markdown("_no capture yet_")
            self.gui_screenshot = self.server.gui.add_image(placeholder, label="screenshot")

            # ---- Phase 2a: mask + neighbors
            self.gui_mode = self.server.gui.add_dropdown(
                "mode",
                options=["disocclusion_only", "masked_inpaint", "outpaint"],
                initial_value="disocclusion_only",
                hint=(
                    "disocclusion_only = mask from low-alpha pixels of splat render. "
                    "masked_inpaint = union of low-alpha and user rectangle. "
                    "outpaint = pad the image + mask the padding."
                ),
            )
            self.gui_alpha_thresh = self.server.gui.add_slider(
                "alpha_thresh", initial_value=0.5, min=0.05, max=0.95, step=0.05,
                hint="Pixels with splat alpha below this go into the inpaint mask.",
            )
            self.gui_n_neighbors = self.server.gui.add_slider(
                "n_neighbors", initial_value=4, min=1, max=8, step=1,
                hint="How many nearest training cameras to feed the inpaint model as reference.",
            )
            self.gui_pad_frac = self.server.gui.add_slider(
                "pad_frac (outpaint)", initial_value=0.25, min=0.05, max=0.5, step=0.05,
                hint="Outpaint mode: pad each side by this fraction of image size.",
            )
            self.gui_mask_x0 = self.server.gui.add_slider("mask x0", initial_value=0.30, min=0.0, max=1.0, step=0.01)
            self.gui_mask_y0 = self.server.gui.add_slider("mask y0", initial_value=0.30, min=0.0, max=1.0, step=0.01)
            self.gui_mask_x1 = self.server.gui.add_slider("mask x1", initial_value=0.70, min=0.0, max=1.0, step=0.01)
            self.gui_mask_y1 = self.server.gui.add_slider("mask y1", initial_value=0.70, min=0.0, max=1.0, step=0.01)
            self.gui_neighbors_btn = self.server.gui.add_button("Find Neighbors + Build Mask")
            self.gui_mask_preview = self.server.gui.add_image(placeholder, label="mask preview")
            self.gui_neighbors_preview = self.server.gui.add_image(placeholder, label="neighbor frames")

            # ---- Phase 2b: inpaint
            self.gui_prompt = self.server.gui.add_text(
                "prompt",
                initial_value=(
                    "Fill in the missing regions consistent with the reference view. "
                    "Maintain scene geometry, lighting, and architecture."
                ),
            )
            self.gui_model_id = self.server.gui.add_text(
                "model_id",
                initial_value="black-forest-labs/FLUX.2-klein-base-9B",
                hint=(
                    "Default = FLUX.2-Klein-base-9B (BFL's 'interactive visual intelligence' inpaint "
                    "model with native reference-image conditioning, 9B params, ~25GB). "
                    "KV variant: black-forest-labs/FLUX.2-klein-9b-kv. "
                    "Alternatives: black-forest-labs/FLUX.1-Fill-dev (no refs), "
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (SDXL inpaint, no refs)."
                ),
            )
            self.gui_steps = self.server.gui.add_slider(
                "steps", initial_value=20, min=1, max=50, step=1,
                hint="FLUX2-Klein: 20-30. FLUX-Fill: 28+. SDXL: 12-20. SD-1.5: 20-30.",
            )
            self.gui_guidance = self.server.gui.add_slider(
                "guidance", initial_value=8.0, min=0.0, max=15.0, step=0.5,
                hint="FLUX2-Klein: 8.0. FLUX-Fill: 3-5. SDXL: 5-8. SD: 7-9.",
            )
            self.gui_strength = self.server.gui.add_slider(
                "strength", initial_value=0.85, min=0.1, max=1.0, step=0.05,
                hint="How much noise to add to the masked region (1.0=full inpaint, lower=preserve "
                     "more of the original). FLUX2-Klein default: 0.8.",
            )
            self.gui_inpaint_btn = self.server.gui.add_button("Inpaint")
            self.gui_inpaint_preview = self.server.gui.add_image(placeholder, label="inpaint result")

        self.gui_capture_btn.on_click(lambda _ev: self._on_capture_click())
        self.gui_neighbors_btn.on_click(lambda _ev: self._on_neighbors_click())
        self.gui_inpaint_btn.on_click(lambda _ev: self._on_inpaint_click())

        # Phase 2 state
        self.last_neighbors: dict | None = None     # {'mask','indices','rgbs','c2ws','K','H','W'}
        self.last_inpaint: dict | None = None        # {'rgb','mask','K','c2w','H','W','path','meta'}
        self._kontext_pipeline = None                # lazy-loaded diffusers pipeline
        self._kontext_model_id = None                # which model_id the cached pipeline is for

    # ----------------------------------------------------------- Phase 1

    def _on_capture_click(self) -> None:
        """Capture a splat render from the viewer's current camera."""
        try:
            t0 = time.perf_counter()

            t = self._trainer_ref
            if t is None or getattr(t, "data", None) is None or getattr(t, "train", None) is None:
                self.gui_status.content = "_Init training first (no splat model)_ "
                return

            # 1. Pick a client camera. Viewer keeps a list internally; just
            #    grab the first connected one (single-user local case).
            clients = self.server.get_clients()
            if not clients:
                self.gui_status.content = "_No client connected_"
                return
            client = next(iter(clients.values()))
            cam = client.camera

            # 2. Build w2c (OpenCV) from the viser camera pose.
            w2c_np = viser_camera_to_opencv_viewmat(cam.position, cam.wxyz)
            c2w_np = np.linalg.inv(w2c_np)
            device = t.means_t.device
            w2c = torch.from_numpy(w2c_np).to(device, dtype=torch.float32)

            # 3. Use the same intrinsics + resolution as the training frames.
            d = t.data
            H, W = int(d.H), int(d.W)
            K = d.K[0].to(device, dtype=torch.float32)

            # 4. Render via gsplat.
            from gsplat import rasterization
            p = t.train.params
            with torch.no_grad():
                means    = p["means"]
                quats    = F.normalize(p["quats"], dim=-1)
                scales   = torch.exp(p["scales"])
                opac     = torch.sigmoid(p["opacities"])
                # SH: cat sh0 + shN (shN may be (M, 0, 3) if sh_max_deg=0)
                sh = torch.cat([p["sh0"], p["shN"]], dim=1)
                sh_deg = int(round(sh.shape[1] ** 0.5)) - 1
                out, alpha, _ = rasterization(
                    means=means, quats=quats, scales=scales, opacities=opac,
                    colors=sh,
                    viewmats=w2c[None], Ks=K[None],
                    width=W, height=H,
                    sh_degree=sh_deg, packed=False,
                    render_mode="RGB+ED",
                )
                # out: (1, H, W, 4) → RGB+depth ; alpha: (1, H, W, 1)
                rgb = out[0, :, :, :3].clamp(0, 1).cpu().numpy()
                alpha_np = alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()

            rgb_u8 = (rgb * 255).astype(np.uint8)

            # 5. Save to disk.
            out_dir = self._out_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"screenshot_{self._capture_counter:04d}.png"
            from PIL import Image
            Image.fromarray(rgb_u8).save(str(png_path))
            self._capture_counter += 1

            # 6. Stash state for downstream phases.
            self.last_screenshot = {
                "rgb": rgb_u8, "alpha": alpha_np,
                "K": K.cpu().numpy(), "c2w": c2w_np,
                "H": H, "W": W, "path": png_path,
            }

            # 7. Show a downscaled thumbnail in the panel.
            self.gui_screenshot.image = _thumbnail(rgb_u8, max_side=384)

            dt = time.perf_counter() - t0
            cov = float((alpha_np > 0.5).mean()) * 100
            self.gui_status.content = (
                f"_Captured {W}×{H}, alpha>0.5 coverage {cov:.0f}% in {dt:.2f}s_  \n"
                f"`{png_path.resolve()}`"
            )
            print(f"  inpainter: captured → {png_path.resolve()}",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_status.content = f"_Capture error: {e}_"
            traceback.print_exc()

    # ---------------------------------------------------------- Phase 2a

    def _on_neighbors_click(self) -> None:
        """Build the inpaint mask + pick nearest training cameras as references."""
        try:
            if self.last_screenshot is None:
                self.gui_status.content = "_Capture a view first (Phase 1)_"
                return
            t = self._trainer_ref
            if t is None or getattr(t, "data", None) is None:
                self.gui_status.content = "_Trainer not initialized_"
                return

            d = t.data
            ss = self.last_screenshot
            H, W = ss["H"], ss["W"]
            c2w_curr = ss["c2w"]                       # (4, 4) numpy float

            # 1. Build the mask. Always start with the alpha-disocclusion mask
            #    from the splat render captured in Phase 1.
            alpha_thresh = float(self.gui_alpha_thresh.value)
            mask = (ss["alpha"] < alpha_thresh).astype(np.uint8)  # (H, W)

            mode = self.gui_mode.value
            if mode == "masked_inpaint":
                # Union with user rectangle (in normalized [0,1] image coords).
                x0 = max(0.0, min(1.0, float(self.gui_mask_x0.value)))
                y0 = max(0.0, min(1.0, float(self.gui_mask_y0.value)))
                x1 = max(0.0, min(1.0, float(self.gui_mask_x1.value)))
                y1 = max(0.0, min(1.0, float(self.gui_mask_y1.value)))
                if x1 <= x0 or y1 <= y0:
                    self.gui_status.content = "_invalid mask rectangle_"
                    return
                ix0, ix1 = int(round(x0 * W)), int(round(x1 * W))
                iy0, iy1 = int(round(y0 * H)), int(round(y1 * H))
                mask[iy0:iy1, ix0:ix1] = 1
            # For 'disocclusion_only' we use the alpha mask as-is.
            # For 'outpaint' we keep mask as the alpha disocclusion of the
            # original image; the padding is applied at inpaint time (Phase 2b).

            # 2. Find nearest training cameras (composite of position dist + fwd dot).
            K_neighbors = int(self.gui_n_neighbors.value)
            c2w_train = d.c2w.detach().cpu().numpy()    # (N, 4, 4)
            t_curr = c2w_curr[:3, 3]
            t_train = c2w_train[:, :3, 3]               # (N, 3)
            # OpenCV camera +Z is forward.
            fwd_curr = c2w_curr[:3, :3] @ np.array([0.0, 0.0, 1.0])
            fwd_train = c2w_train[:, :3, :3] @ np.array([0.0, 0.0, 1.0])
            pos_dist = np.linalg.norm(t_train - t_curr[None], axis=1)            # (N,)
            fwd_dot = (fwd_train * fwd_curr[None]).sum(axis=1)                   # (N,)
            # Higher score = better. Position closeness contributes more than orientation.
            score = (1.0 / (pos_dist + 1e-6)) + 0.5 * fwd_dot
            order = np.argsort(-score)
            neighbor_idx = order[:K_neighbors].tolist()

            # 3. Pull the neighbor RGB frames (training data is uint-ish float [0,1]).
            rgbs = []
            for i in neighbor_idx:
                frame = d.rgb[i].detach().cpu().numpy()  # (H, W, 3) float [0,1]
                rgbs.append((frame * 255).clip(0, 255).astype(np.uint8))

            # 4. Stash + display.
            self.last_neighbors = {
                "mask": mask,
                "indices": neighbor_idx,
                "rgbs": rgbs,
                "c2ws": [c2w_train[i] for i in neighbor_idx],
                "K": ss["K"],
                "H": H, "W": W,
                "mode": mode,
            }

            mask_overlay = _overlay_mask(ss["rgb"], mask)
            self.gui_mask_preview.image = _thumbnail(mask_overlay, max_side=384)
            self.gui_neighbors_preview.image = _thumbnail(_tile_images(rgbs), max_side=512)

            n_masked = int(mask.sum())
            mask_pct = 100.0 * n_masked / mask.size
            self.gui_status.content = (
                f"_mode={mode}_  \n"
                f"_mask covers {mask_pct:.1f}% of pixels_  \n"
                f"_picked neighbors: {neighbor_idx} (closest pos {pos_dist[neighbor_idx[0]]:.3f}m)_"
            )
            print(f"  inpainter: mode={mode} mask_px={n_masked} neighbors={neighbor_idx}",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_status.content = f"_Neighbors error: {e}_"
            traceback.print_exc()

    # ---------------------------------------------------------- Phase 2b

    def _on_inpaint_click(self) -> None:
        """Run the inpainting model on the captured render with neighbor references."""
        try:
            if self.last_neighbors is None:
                self.gui_status.content = "_Click 'Find Neighbors + Build Mask' first_"
                return
            ss = self.last_screenshot
            nb = self.last_neighbors
            mode = nb["mode"]

            # Prepare the image + mask to feed the model.
            from PIL import Image
            image_u8 = ss["rgb"]                                      # (H, W, 3)
            mask_u8 = (nb["mask"] * 255).astype(np.uint8)              # (H, W) — 255=fill, 0=keep
            H, W = ss["H"], ss["W"]
            K_out = ss["K"].copy()
            c2w_out = ss["c2w"].copy()

            if mode == "outpaint":
                # Pad image + mask, shift principal point in K.
                pad_frac = float(self.gui_pad_frac.value)
                pad_x = int(round(W * pad_frac))
                pad_y = int(round(H * pad_frac))
                padded_rgb = np.zeros((H + 2 * pad_y, W + 2 * pad_x, 3), dtype=np.uint8)
                padded_rgb[pad_y:pad_y + H, pad_x:pad_x + W] = image_u8
                # Mask is 255 everywhere EXCEPT the inner region (which we union
                # with the existing alpha mask).
                padded_mask = np.full((H + 2 * pad_y, W + 2 * pad_x), 255, dtype=np.uint8)
                inner = mask_u8.copy()                                # inner disocclusion
                padded_mask[pad_y:pad_y + H, pad_x:pad_x + W] = inner
                image_pil = Image.fromarray(padded_rgb)
                mask_pil = Image.fromarray(padded_mask)
                # Updated intrinsics + resolution
                K_out[0, 2] += pad_x
                K_out[1, 2] += pad_y
                W = padded_rgb.shape[1]
                H = padded_rgb.shape[0]
            else:
                image_pil = Image.fromarray(image_u8)
                mask_pil = Image.fromarray(mask_u8)

            ref_pils = [Image.fromarray(r) for r in nb["rgbs"]]
            prompt = self.gui_prompt.value
            steps = int(self.gui_steps.value)
            guidance = float(self.gui_guidance.value)
            model_id = self.gui_model_id.value

            # SDXL/FLUX require height & width divisible by 8 (FLUX often 16).
            # Pad the image+mask up to the next multiple, run, then crop back.
            mult = 16 if "flux" in model_id.lower() else 8
            target_W = ((image_pil.width + mult - 1) // mult) * mult
            target_H = ((image_pil.height + mult - 1) // mult) * mult
            pad_r = target_W - image_pil.width
            pad_b = target_H - image_pil.height
            if pad_r or pad_b:
                im_np = np.asarray(image_pil)
                mk_np = np.asarray(mask_pil)
                im_padded = np.pad(im_np, ((0, pad_b), (0, pad_r), (0, 0)),
                                   mode="edge")
                # Mask the padded strip = 0 (don't inpaint there), so output
                # padding region is undefined but we'll crop it out anyway.
                mk_padded = np.pad(mk_np, ((0, pad_b), (0, pad_r)),
                                   mode="constant", constant_values=0)
                image_for_pipe = Image.fromarray(im_padded)
                mask_for_pipe = Image.fromarray(mk_padded)
            else:
                image_for_pipe = image_pil
                mask_for_pipe = mask_pil

            self.gui_status.content = f"_Loading {model_id}…_"
            pipeline = self._get_pipeline(model_id)
            self.gui_status.content = f"_Running {model_id} ({steps} steps)…_"
            t0 = time.perf_counter()

            # Debug: save the EXACT image+mask going to the pipeline so we can
            # check that the mask is meaningful (white = inpaint, black = keep)
            # and the image looks right.
            dbg_dir = self._out_dir()
            dbg_dir.mkdir(parents=True, exist_ok=True)
            image_for_pipe.save(str(dbg_dir / "_dbg_pipe_input.png"))
            mask_for_pipe.save(str(dbg_dir / "_dbg_pipe_mask.png"))
            print(f"  inpainter: dbg input → {dbg_dir / '_dbg_pipe_input.png'}",
                  file=sys.stderr, flush=True)
            print(f"  inpainter: dbg mask  → {dbg_dir / '_dbg_pipe_mask.png'}",
                  file=sys.stderr, flush=True)
            mk_arr = np.asarray(mask_for_pipe)
            print(f"  inpainter: mask stats: min={mk_arr.min()} max={mk_arr.max()} "
                  f"mean={mk_arr.mean():.1f} white_pixels={(mk_arr > 127).sum()}",
                  file=sys.stderr, flush=True)

            with torch.no_grad():
                kwargs = dict(
                    prompt=prompt,
                    image=image_for_pipe,
                    mask_image=mask_for_pipe,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    strength=float(self.gui_strength.value),
                    height=image_for_pipe.height,
                    width=image_for_pipe.width,
                )
                # FLUX-schnell's T5 caps at 256 tokens; other FLUX variants 512.
                pipeline_kind = type(pipeline).__name__
                if "Flux" in pipeline_kind and "Flux2" not in pipeline_kind:
                    kwargs["max_sequence_length"] = 256 if "schnell" in model_id.lower() else 512
                # Reference-image arg per pipeline:
                #   - Flux2KleinInpaintPipeline → image_reference (single PIL)
                #   - FluxKontextInpaintPipeline → ip_adapter_image (needs adapter loaded)
                #   - FluxInpaintPipeline → ip_adapter_image (needs adapter loaded)
                #   - Others → no reference support; skip silently
                if ref_pils and "Klein" in pipeline_kind:
                    # Klein takes a single best-neighbor reference (the first one,
                    # which we already ranked highest in Phase 2a).
                    kwargs["image_reference"] = ref_pils[0]
                out = pipeline(**kwargs).images[0]                    # PIL

            dt = time.perf_counter() - t0

            inpaint_u8 = np.asarray(out.convert("RGB"))
            # Crop off the padding so the output matches the saved K/c2w/H/W.
            if inpaint_u8.shape[:2] != (H, W):
                inpaint_u8 = inpaint_u8[:H, :W]
            if inpaint_u8.shape[:2] != (H, W):
                # Fallback if the model rescaled (rare, but covered)
                inpaint_u8 = np.asarray(Image.fromarray(inpaint_u8).resize((W, H), Image.BILINEAR))

            # Phase 2c — save artifacts.
            out_dir = self._out_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"inpaint_{len(list(out_dir.glob('inpaint_*.png'))):04d}"
            png_path = out_dir / f"{stem}.png"
            mask_path = out_dir / f"{stem}.mask.png"
            meta_path = out_dir / f"{stem}.meta.json"
            Image.fromarray(inpaint_u8).save(str(png_path))
            Image.fromarray(_resize_to(mask_u8 if mode != "outpaint" else padded_mask,
                                       (H, W))).save(str(mask_path))
            import json
            meta_path.write_text(json.dumps({
                "mode": mode,
                "model_id": model_id,
                "prompt": prompt,
                "steps": steps,
                "guidance": guidance,
                "neighbor_indices": nb["indices"],
                "K": K_out.tolist(),
                "c2w": c2w_out.tolist(),
                "H": H, "W": W,
                "took_seconds": dt,
            }, indent=2))

            self.last_inpaint = {
                "rgb": inpaint_u8,
                "mask": (np.asarray(mask_pil) > 127).astype(np.uint8),
                "K": K_out, "c2w": c2w_out, "H": H, "W": W,
                "path": png_path, "meta_path": meta_path,
            }
            self.gui_inpaint_preview.image = _thumbnail(inpaint_u8, max_side=384)
            self.gui_status.content = (
                f"_Inpainted in {dt:.1f}s ({model_id})_  \n"
                f"`{png_path.resolve()}`"
            )
            print(f"  inpainter: inpaint saved → {png_path.resolve()}",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_status.content = f"_Inpaint error: {e}_"
            traceback.print_exc()

    def _get_pipeline(self, model_id: str):
        """Lazy-load + cache the diffusers pipeline. Picks the right class
        from the model id (case-insensitive) and the right dtype per family."""
        if self._kontext_pipeline is not None and self._kontext_model_id == model_id:
            return self._kontext_pipeline
        import diffusers
        mid = model_id.lower()
        # Choose pipeline class:
        if "klein" in mid:
            cls = diffusers.Flux2KleinInpaintPipeline   # inpaint + image_reference
        elif "kontext" in mid:
            cls = diffusers.FluxKontextInpaintPipeline  # whole-image edit + mask
        elif "flux" in mid and "fill" in mid:
            cls = diffusers.FluxFillPipeline
        elif "flux" in mid:
            cls = diffusers.FluxInpaintPipeline         # schnell + dev
        else:
            cls = diffusers.AutoPipelineForInpainting
        # FLUX prefers bf16; SD/SDXL run fine in fp16.
        dtype = torch.bfloat16 if "flux" in mid else torch.float16
        pipeline = cls.from_pretrained(model_id, torch_dtype=dtype).to("cuda")
        self._kontext_pipeline = pipeline
        self._kontext_model_id = model_id
        return pipeline

    # ------------------------------------------------------------ helpers

    def _out_dir(self) -> Path:
        """Resolve the output directory based on the trainer's name."""
        name = getattr(self._trainer_ref, "name", None) or "default"
        return self._out_root / name


def _thumbnail(rgb_u8: np.ndarray, max_side: int = 384) -> np.ndarray:
    """Downscale an (H, W, 3) uint8 image so its longest side ≤ max_side."""
    from PIL import Image
    h, w = rgb_u8.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb_u8
    s = max_side / longest
    new_w, new_h = int(w * s), int(h * s)
    return np.asarray(Image.fromarray(rgb_u8).resize((new_w, new_h), Image.BILINEAR))


def _overlay_mask(rgb_u8: np.ndarray, mask: np.ndarray,
                  tint=(255, 60, 60), alpha: float = 0.45) -> np.ndarray:
    """Blend a red tint onto rgb where mask == 1."""
    out = rgb_u8.astype(np.float32)
    m = (mask > 0).astype(np.float32)[..., None]
    tint_arr = np.array(tint, dtype=np.float32).reshape(1, 1, 3)
    out = out * (1 - alpha * m) + tint_arr * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _tile_images(images: list[np.ndarray], cols: int | None = None) -> np.ndarray:
    """Tile a list of (H, W, 3) uint8 images into a grid for display."""
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    n = len(images)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # Resize all to the first image's shape (or some common small size).
    from PIL import Image
    target_w = 240
    resized = []
    for im in images:
        h, w = im.shape[:2]
        s = target_w / w
        th = int(round(h * s))
        resized.append(np.asarray(Image.fromarray(im).resize((target_w, th), Image.BILINEAR)))
    target_h = max(im.shape[0] for im in resized)
    canvas = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    for i, im in enumerate(resized):
        r, c = i // cols, i % cols
        ih, iw = im.shape[:2]
        canvas[r * target_h: r * target_h + ih, c * target_w: c * target_w + iw] = im
    return canvas


def _resize_to(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize (H, W) or (H, W, C) uint8 to target (H, W)."""
    from PIL import Image
    H, W = hw
    pil = Image.fromarray(img)
    return np.asarray(pil.resize((W, H), Image.NEAREST))
