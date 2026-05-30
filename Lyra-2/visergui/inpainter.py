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
import threading
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
        preload: bool = True,
    ) -> None:
        self.server = server
        self._trainer_ref = trainer_ref
        self._viewer = viewer
        self._out_root = Path(out_root) if out_root is not None else Path("inpaint_outputs")
        self._preload = bool(preload)

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
                options=["disocclusion_only", "outpaint"],
                initial_value="disocclusion_only",
                hint=(
                    "disocclusion_only = mask from low-alpha pixels of splat render. "
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
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (SDXL inpaint, no refs). "
                    "The model is NOT auto-loaded (it's ~25GB and OOMs alongside training). "
                    "Click 'Load model' to warm it once you've initialized a scene, or just "
                    "click Inpaint and the load happens lazily on first use."
                ),
            )
            self.gui_load_model_btn = self.server.gui.add_button(
                "Load model",
                hint="Load the diffusers pipeline for the model_id above into "
                     "GPU memory now (~25GB, ~30-60 s), so the first Inpaint "
                     "click is instant. Initialize a scene first — FLUX must "
                     "load AFTER DA3 or DA3's weight-init breaks. Off by "
                     "default; this is the manual replacement for boot preload.",
            )
            self.gui_model_status = self.server.gui.add_markdown(
                "**model:** not loaded"
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

            # ---- Phase 3: InstaInpaint backend (parallel to FLUX-Kontext)
            self.gui_backend = self.server.gui.add_dropdown(
                "backend",
                options=["flux_kontext", "instainpaint"],
                initial_value="flux_kontext",
                hint=(
                    "flux_kontext = existing FLUX path (~30 s, 2D inpaint + DA3 depth + injection). "
                    "instainpaint = feed-forward 3DGS inpaint (~0.4 s, direct splat output)."
                ),
            )
            self.gui_instainpaint_btn = self.server.gui.add_button("Run InstaInpaint")
            self.gui_instainpaint_status = self.server.gui.add_markdown("_no instainpaint run yet_")

            # ---- Phase 4: append captured view as a training frame
            self.gui_add_frame_btn = self.server.gui.add_button("Add as Training Frame")
            self.gui_add_frame_status = self.server.gui.add_markdown("_no frame appended_")

        self.gui_capture_btn.on_click(lambda _ev: self._on_capture_click())
        self.gui_neighbors_btn.on_click(lambda _ev: self._on_neighbors_click())
        self.gui_load_model_btn.on_click(lambda _ev: self._on_load_model_click())
        self.gui_inpaint_btn.on_click(lambda _ev: self._on_inpaint_click())
        self.gui_instainpaint_btn.on_click(lambda _ev: self._on_instainpaint_click())
        self.gui_add_frame_btn.on_click(lambda _ev: self._on_add_frame_click())

        # Phase 2 state
        self.last_neighbors: dict | None = None     # {'mask','indices','rgbs','c2ws','K','H','W'}
        self.last_inpaint: dict | None = None        # {'rgb','mask','K','c2w','H','W','path','meta'}
        self._kontext_pipeline = None                # lazy-loaded diffusers pipeline
        self._kontext_model_id = None                # which model_id the cached pipeline is for
        # Lock around the pipeline cache: preload runs in a daemon thread
        # and the user can still click Inpaint mid-load. The lock serialises
        # them so we never double-load, and the second caller sees the
        # already-cached pipeline once the first finishes.
        self._pipeline_lock = threading.Lock()
        # Phase 3 state — provenance ledger keyed on splat index ranges added by each run.
        self.gaussian_provenance: list[dict] = []    # [{'backend':..., 'start':int, 'end':int}, ...]

        # Inpaint pipeline preload is deferred — see `start_preload()`. We
        # used to spawn it from here, but loading FLUX-Klein (diffusers /
        # accelerate) before DA3 puts `from_pretrained` into the
        # init_empty_weights pattern globally, which leaves subsequent
        # DA3 params as meta tensors and breaks `.to(device)`. Waiting
        # until after the first prepare_and_init lets DA3 load cleanly
        # first, then FLUX preloads in the background while training runs.
        self._preload_started = False
        self._model_loading = False  # guards the manual 'Load model' button
        if not self._preload:
            print(
                "[inpainter] auto-preload disabled — click 'Load model' in the "
                "Inpaint tab to warm FLUX, or the first Inpaint click loads it "
                "lazily.",
                file=sys.stderr,
            )

    # ----------------------------------------------------------- Phase 1

    def _on_capture_click(self) -> None:
        """Capture a render from the viewer's current camera, honoring its display mode."""
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

            # 3. Resolution + intrinsics: match the viser viewport so the
            #    capture is WYSIWYG. Fall back to training shape if viser
            #    hasn't reported a viewport size yet.
            d = t.data
            try:
                W = int(cam.image_width); H = int(cam.image_height)
                if W <= 0 or H <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                H, W = int(d.H), int(d.W)
            # Cap resolution to keep capture time reasonable; preserve aspect.
            max_side = max(int(d.H), int(d.W))
            if max(H, W) > max_side:
                s = max_side / max(H, W)
                H = max(1, int(round(H * s)))
                W = max(1, int(round(W * s)))
            vfov = float(getattr(cam, "fov", 1.047))  # radians; ~60° fallback
            fy = H / (2.0 * float(np.tan(vfov * 0.5)))
            fx = fy                                                    # square pixels
            K_np = np.array([[fx, 0.0, W / 2.0],
                             [0.0, fy, H / 2.0],
                             [0.0, 0.0, 1.0]], dtype=np.float32)
            K = torch.from_numpy(K_np).to(device)

            # 4. Pick renderer based on the viewer's Display dropdown.
            #    Splat capture yields per-pixel depth; the other modes leave it None.
            display_mode = (self._viewer.display_mode if self._viewer is not None else "splats")
            depth_np = None
            if display_mode == "points":
                rgb_u8, alpha_np = self._render_points(w2c, K, W, H, device)
            elif display_mode == "mesh":
                rgb_u8, alpha_np = self._render_mesh(w2c_np, K.cpu().numpy(), W, H)
            else:
                rgb_u8, alpha_np, depth_np = self._render_splats(w2c, K, W, H, t)

            # 5. Save to disk.
            out_dir = self._out_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"screenshot_{self._capture_counter:04d}.png"
            from PIL import Image
            Image.fromarray(rgb_u8).save(str(png_path))
            self._capture_counter += 1

            # 6. Stash state for downstream phases.
            self.last_screenshot = {
                "rgb": rgb_u8, "alpha": alpha_np, "depth": depth_np,
                "K": K.cpu().numpy(), "c2w": c2w_np,
                "H": H, "W": W, "path": png_path,
            }

            # 7. Show a downscaled thumbnail in the panel.
            self.gui_screenshot.image = _thumbnail(rgb_u8, max_side=384)

            dt = time.perf_counter() - t0
            cov = float((alpha_np > 0.5).mean()) * 100
            self.gui_status.content = (
                f"_Captured ({display_mode}) {W}×{H}, alpha>0.5 coverage "
                f"{cov:.0f}% in {dt:.2f}s_  \n"
                f"`{png_path.resolve()}`"
            )
            print(f"  inpainter: captured → {png_path.resolve()}",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_status.content = f"_Capture error: {e}_"
            traceback.print_exc()

    # ---------------------------------------------- per-display-mode renderers

    def _render_splats(self, w2c, K, W, H, t):
        """Returns (rgb_u8, alpha_np, depth_np). Depth is in world units; pixels
        with alpha=0 will have depth=0 from gsplat's accumulated-depth output."""
        from gsplat import rasterization
        p = t.train.params
        with torch.no_grad():
            sh = torch.cat([p["sh0"], p["shN"]], dim=1)
            sh_deg = int(round(sh.shape[1] ** 0.5)) - 1
            out, alpha, _ = rasterization(
                means=p["means"],
                quats=F.normalize(p["quats"], dim=-1),
                scales=torch.exp(p["scales"]),
                opacities=torch.sigmoid(p["opacities"]),
                colors=sh,
                viewmats=w2c[None], Ks=K[None],
                width=W, height=H,
                sh_degree=sh_deg, packed=False,
                render_mode="RGB+ED",
            )
            rgb = out[0, :, :, :3].clamp(0, 1).cpu().numpy()
            alpha_np = alpha[0, :, :, 0].clamp(0, 1).cpu().numpy()
            # gsplat ED accumulates expected depth; divide by alpha to get expected depth.
            a = alpha[0, :, :, 0].clamp_min(1e-6)
            depth_np = (out[0, :, :, 3] / a).cpu().numpy()
        return (rgb * 255).astype(np.uint8), alpha_np, depth_np

    def _render_points(self, w2c, K, W, H, device) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize visible PointCloudLayers as screen-space sprites with a
        z-buffer — mirrors three.js point rendering used by viser."""
        from dataclasses import replace as dc_replace
        from viewer import compute_colors

        handles = getattr(self._viewer, "_handles", {})
        pts_list, col_list, size_list = [], [], []
        scene = self._viewer.scene
        with scene.read() as s:
            layer_items = list(s.point_clouds.items())
        for name, layer in layer_items:
            h = handles.get(name)
            if h is not None:
                if not bool(h["visible"].value):
                    continue
                resolved = dc_replace(
                    layer,
                    point_size=float(h["size"].value),
                    color_mode=str(h["color_mode"].value),
                    uniform_color=tuple(int(c) for c in h["uniform_color"].value),
                )
            else:
                if not layer.visible:
                    continue
                resolved = layer
            pts_list.append(resolved.points.astype(np.float32))
            col_list.append(compute_colors(resolved).astype(np.float32) / 255.0)
            size_list.append(np.full(len(resolved.points), float(resolved.point_size), dtype=np.float32))

        if not pts_list:
            return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32)

        pts_w = torch.from_numpy(np.concatenate(pts_list)).to(device)
        cols  = torch.from_numpy(np.concatenate(col_list)).to(device)
        sizes = torch.from_numpy(np.concatenate(size_list)).to(device)

        # World → camera → image
        R = w2c[:3, :3]; tvec = w2c[:3, 3]
        pts_cam = pts_w @ R.T + tvec                                  # (N, 3)
        z = pts_cam[:, 2]
        in_front = z > 1e-3
        if not bool(in_front.any()):
            return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32)
        pts_cam = pts_cam[in_front]; cols = cols[in_front]
        sizes  = sizes[in_front];   z = z[in_front]

        proj = pts_cam @ K.T / z.unsqueeze(-1)                        # (N, 3)
        ix = proj[:, 0]; iy = proj[:, 1]

        # Match viser's custom point shader (ThreeAssets.tsx PointCloudMaterial):
        #   gl_PointSize_diameter_DB = point_size / tan(fov/2) * H_css * DPR / z
        # In CSS pixels visible to the user that simplifies to
        #   diameter_px = 2 * fy * size / z  →  radius_px = fy * size / z.
        r_px = (K[1, 1] * sizes / z).clamp(min=0.5)

        # Cull points whose projected disc lies entirely off-screen.
        on_screen = (
            (ix + r_px >= 0) & (ix - r_px < W) &
            (iy + r_px >= 0) & (iy - r_px < H)
        )
        ix = ix[on_screen]; iy = iy[on_screen]; r_px = r_px[on_screen]
        cols = cols[on_screen]; z = z[on_screen]
        if ix.numel() == 0:
            return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32)

        # Sort back-to-front so nearer points overwrite farther ones in the scatter.
        order = torch.argsort(z, descending=True)
        ix = ix[order]; iy = iy[order]; r_px = r_px[order]
        cols = cols[order]; z = z[order]

        img = torch.zeros((H, W, 3), device=device)
        zbuf = torch.full((H, W), float("inf"), device=device)

        # Scatter into each pixel within the on-screen disc. The disc cap (R) is
        # taken from the max projected radius — typically 1–4 px for viser point
        # sizes, so the (2R+1)^2 inner loop is tiny.
        R_max = int(r_px.max().ceil().item())
        for dy in range(-R_max, R_max + 1):
            for dx in range(-R_max, R_max + 1):
                if dx * dx + dy * dy > R_max * R_max:
                    continue
                # Per-point: in-radius for THIS offset?
                d2 = float(dx * dx + dy * dy)
                inside = d2 <= r_px * r_px
                if not bool(inside.any()):
                    continue
                px = (ix[inside] + dx).round().long()
                py = (iy[inside] + dy).round().long()
                bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                px = px[bounds]; py = py[bounds]
                cz = z[inside][bounds]; cc = cols[inside][bounds]
                if px.numel() == 0:
                    continue
                # Per-pixel z-buffer compare. Use scatter_reduce to take min depth.
                flat = py * W + px
                # For simplicity: iterate in back-to-front order means later writes win.
                # Filter to pixels where this point is in front of current zbuf.
                current_z = zbuf.view(-1)[flat]
                closer = cz < current_z
                if not bool(closer.any()):
                    continue
                flat_c = flat[closer]; cc_c = cc[closer]; cz_c = cz[closer]
                zbuf.view(-1).scatter_(0, flat_c, cz_c)
                img.view(-1, 3).index_copy_(0, flat_c, cc_c)

        alpha = (zbuf < float("inf")).to(torch.float32)
        rgb_u8 = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return rgb_u8, alpha.cpu().numpy()

    def _render_mesh(self, w2c_np, K_np, W, H) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize the current mesh via Open3D's offscreen renderer."""
        import open3d as o3d
        scene = self._viewer.scene
        with scene.read() as s:
            verts = s.mesh_verts
            faces = s.mesh_faces
            colors = s.mesh_colors
        if verts is None or faces is None:
            raise RuntimeError("No mesh available — run Generate Mesh first.")

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        mesh.compute_vertex_normals()

        renderer = o3d.visualization.rendering.OffscreenRenderer(int(W), int(H))
        renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit" if colors is not None else "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)

        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        intr = o3d.camera.PinholeCameraIntrinsic(int(W), int(H), fx, fy, cx, cy)
        renderer.setup_camera(intr, w2c_np.astype(np.float64))

        img = np.asarray(renderer.render_to_image())                  # (H, W, 3) uint8
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        alpha = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
        return img, alpha

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

    # ----------------------------------------------------------- Phase 3

    def _on_instainpaint_click(self) -> None:
        """Run the InstaInpaint backend: feed-forward 3DGS inpainting.

        Uses the captured viewport (view 0) + 3 nearest training views with
        SAM-2 per-view masks, then merges the resulting gaussian dict directly
        into trainer.train.params.
        """
        try:
            t0 = time.perf_counter()

            if self.last_screenshot is None:
                self.gui_instainpaint_status.content = "_Capture a view first_"
                return
            if self.last_neighbors is None:
                self.gui_instainpaint_status.content = "_Click 'Find Neighbors + Build Mask' first_"
                return
            ss = self.last_screenshot
            if ss.get("depth") is None:
                self.gui_instainpaint_status.content = (
                    "_InstaInpaint needs splat-mode capture for depth — "
                    "switch viewer Display to 'splats' and re-capture._"
                )
                return

            t = self._trainer_ref
            if t is None or getattr(t, "train", None) is None:
                self.gui_instainpaint_status.content = "_Trainer not initialized_"
                return

            nb = self.last_neighbors
            captured_rgb_u8 = ss["rgb"]
            captured_mask = nb["mask"].astype(bool)
            captured_depth = ss["depth"]
            captured_K = np.asarray(ss["K"], dtype=np.float32)
            captured_c2w = np.asarray(ss["c2w"], dtype=np.float32)

            # 1. SAM-2 per-neighbor masks via back-projection through depth.
            from multiview_mask import prepare_multiview_masks
            neighbor_rgbs = nb["rgbs"][:3]                # take exactly 3
            neighbor_c2ws = [np.asarray(c, dtype=np.float32) for c in nb["c2ws"][:3]]
            neighbor_Ks   = [np.asarray(nb["K"], dtype=np.float32)] * len(neighbor_rgbs)
            if len(neighbor_rgbs) < 3:
                self.gui_instainpaint_status.content = (
                    f"_Need 3 neighbors, got {len(neighbor_rgbs)}. Raise 'n_neighbors' slider._"
                )
                return
            self.gui_instainpaint_status.content = "_Running SAM 2 per-view masks…_"
            _, nb_masks, nb_prompts = prepare_multiview_masks(
                captured_rgb_u8, captured_mask, captured_depth,
                captured_K, captured_c2w,
                neighbor_rgbs, neighbor_Ks, neighbor_c2ws,
            )
            valid_nb = [(rgb, m, K_, c2w_) for rgb, m, K_, c2w_ in
                        zip(neighbor_rgbs, nb_masks, neighbor_Ks, neighbor_c2ws)
                        if m is not None]
            if len(valid_nb) < 3:
                self.gui_instainpaint_status.content = (
                    f"_SAM 2 projected only {len(valid_nb)}/3 valid neighbors. "
                    f"Pick a different click; the 3D point may have fallen out of the others' frusta._"
                )
                return

            # 2. InstaInpaint expects ALL 4 views at the same H×W, multiples of 8.
            #    Use the captured resolution; resize neighbors to match.
            H_c, W_c = int(ss["H"]), int(ss["W"])
            H_t = (H_c // 8) * 8
            W_t = (W_c // 8) * 8
            if (H_t, W_t) != (H_c, W_c):
                print(f"  instainpaint: cropping {H_c}x{W_c} → {H_t}x{W_t} (multiple of 8)",
                      file=sys.stderr, flush=True)

            def _crop_to_8(img: np.ndarray) -> np.ndarray:
                return img[:H_t, :W_t]

            def _resize_to(img: np.ndarray, mode_bilinear: bool) -> np.ndarray:
                import cv2
                if img.shape[:2] == (H_t, W_t):
                    return img
                interp = cv2.INTER_AREA if mode_bilinear else cv2.INTER_NEAREST
                return cv2.resize(img, (W_t, H_t), interpolation=interp)

            captured_rgb_f = _crop_to_8(captured_rgb_u8).astype(np.float32) / 255.0
            captured_mask_f = _crop_to_8(captured_mask).astype(np.float32)
            ref_rgbs_f, ref_masks_f = [], []
            for rgb, m, _, _ in valid_nb:
                ref_rgbs_f.append(_resize_to(rgb, True).astype(np.float32) / 255.0)
                ref_masks_f.append(_resize_to(m.astype(np.uint8), False).astype(np.float32))

            rgbs   = torch.from_numpy(np.stack([captured_rgb_f]  + ref_rgbs_f))     # (4, H, W, 3)
            masks  = torch.from_numpy(np.stack([captured_mask_f] + ref_masks_f))    # (4, H, W)
            c2ws_cv = torch.from_numpy(np.stack([captured_c2w] + [c2w for _, _, _, c2w in valid_nb]))
            Ks     = torch.from_numpy(np.stack([captured_K]   + [K_ for _, _, K_, _ in valid_nb]))

            # 3. Call the wrapper.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "InstaInpaint"))
            from wrapper import inpaint as _instainpaint
            self.gui_instainpaint_status.content = f"_Running InstaInpaint on {H_t}x{W_t}…_"
            out = _instainpaint(rgbs, masks, c2ws_cv, Ks)

            # 4. Filter to the inpaint mask of the captured view (view 0) only.
            #    Output gaussians are (1, 4, C, H, W); pick view 0 and mask
            #    where the captured mask is 1 (the region we asked to inpaint).
            xyz_v0     = out["xyz"][0, 0]        # (3, H, W)
            rgb_v0     = out["rgb"][0, 0]        # (3, H, W)
            opacity_v0 = out["opacity"][0, 0, 0] # (H, W)
            scale_v0   = out["scale"][0, 0]      # (3, H, W)
            rot_v0     = out["rotation"][0, 0]   # (4, H, W)

            m_t = torch.from_numpy(captured_mask_f > 0.5).to(xyz_v0.device)
            sel = m_t.flatten()
            n_new = int(sel.sum().item())
            if n_new == 0:
                self.gui_instainpaint_status.content = "_Mask was empty after crop — nothing to add._"
                return

            def _flat(t):  # (C, H, W) → (H*W, C)
                return t.permute(1, 2, 0).reshape(-1, t.shape[0])

            means_new    = _flat(xyz_v0)[sel]                                   # (n_new, 3)
            rgb_new      = _flat(rgb_v0).clamp(0, 1)[sel]                       # (n_new, 3) — assume already in [0,1] after fp32 cast
            opacity_new  = opacity_v0.flatten()[sel].clamp(1e-6, 1.0 - 1e-6)    # (n_new,)
            scale_new    = _flat(scale_v0).clamp_min(1e-6)[sel]                 # (n_new, 3)
            rot_new      = _flat(rot_v0)[sel]                                   # (n_new, 4) quaternion (w, x, y, z)

            # 5. Merge into trainer.train.params (same in-place resize pattern
            #    used elsewhere). RGB → sh0 via inverse SH-DC: sh0 = (rgb-0.5)/C0.
            C0 = 0.282094791
            sh0_new = ((rgb_new - 0.5) / C0).unsqueeze(1)                       # (n_new, 1, 3)

            p = t.train.params
            existing_means = p["means"]
            shN_existing = p["shN"]
            shN_bands = shN_existing.shape[1]
            shN_new = torch.zeros((n_new, shN_bands, 3), device=means_new.device, dtype=existing_means.dtype)

            # Inverse activations: trainer stores log(scale), logit(opacity), unnormalized quats.
            log_scale_new = torch.log(scale_new.to(existing_means.dtype))
            logit_opacity_new = torch.log(opacity_new / (1 - opacity_new)).to(existing_means.dtype)
            quats_new = rot_new.to(existing_means.dtype)
            quats_new = quats_new / quats_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            start_idx = int(existing_means.shape[0])

            def _append(name: str, new_tensor: torch.Tensor):
                old = p[name].data
                new_tensor = new_tensor.to(device=old.device, dtype=old.dtype)
                p[name].data = torch.cat([old, new_tensor], dim=0)

            _append("means",     means_new.to(existing_means.dtype))
            _append("quats",     quats_new)
            _append("scales",    log_scale_new)
            _append("opacities", logit_opacity_new)
            _append("sh0",       sh0_new.to(existing_means.dtype))
            _append("shN",       shN_new)

            end_idx = int(p["means"].shape[0])
            self.gaussian_provenance.append({
                "backend": "instainpaint", "start": start_idx, "end": end_idx,
                "captured_path": str(ss["path"]),
            })

            # Trigger re-render.
            if hasattr(self._viewer, "scene") and hasattr(self._viewer.scene, "_lock"):
                with self._viewer.scene._lock:
                    self._viewer.scene.splat_version += 1
                    self._viewer.scene.num_splats = end_idx
            elif hasattr(t, "scene") and getattr(t, "scene", None) is not None:
                with t.scene._lock:
                    t.scene.splat_version += 1
                    t.scene.num_splats = end_idx

            dt = time.perf_counter() - t0
            self.gui_instainpaint_status.content = (
                f"_InstaInpaint merged {n_new:,} new gaussians "
                f"(total {end_idx:,}) in {dt:.2f}s_"
            )
            print(f"  instainpaint: merged {n_new} gaussians → total {end_idx} in {dt:.2f}s",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_instainpaint_status.content = f"_InstaInpaint error: {e}_"
            traceback.print_exc()

    # ----------------------------------------------------------- Phase 4

    def _on_add_frame_click(self) -> None:
        """Append the current capture to trainer.data so the photometric loop
        can refine the newly merged region. Prefers the FLUX-inpainted RGB
        when available; falls back to the raw splat render. Requires depth
        (i.e., splat-mode capture)."""
        try:
            if self.last_screenshot is None:
                self.gui_add_frame_status.content = "_Capture a view first_"
                return
            t = self._trainer_ref
            if t is None or getattr(t, "data", None) is None:
                self.gui_add_frame_status.content = "_Trainer not initialized_"
                return
            ss = self.last_screenshot
            depth_np = ss.get("depth")
            if depth_np is None:
                self.gui_add_frame_status.content = (
                    "_Need splat-mode capture for depth — switch viewer Display to 'splats'._"
                )
                return

            # Prefer FLUX-inpainted RGB if a recent inpaint was run; else raw.
            if self.last_inpaint is not None and self.last_inpaint.get("rgb") is not None:
                rgb_src = self.last_inpaint["rgb"]
                src_tag = "flux-inpainted"
            else:
                rgb_src = ss["rgb"]
                src_tag = "raw splat render"

            d = t.data
            H, W = int(d.H), int(d.W)
            # Resize to training resolution if needed (data tensors are fixed H×W).
            import cv2 as _cv2
            if rgb_src.shape[:2] != (H, W):
                rgb_src = _cv2.resize(rgb_src, (W, H), interpolation=_cv2.INTER_AREA)
                depth_np = _cv2.resize(depth_np, (W, H), interpolation=_cv2.INTER_NEAREST)
            # Build K at the training resolution from the captured K, scaling by H/W ratio.
            K_cap = np.asarray(ss["K"], dtype=np.float32)
            scale_x = W / float(ss["W"])
            scale_y = H / float(ss["H"])
            K_new = K_cap.copy()
            K_new[0, 0] *= scale_x
            K_new[1, 1] *= scale_y
            K_new[0, 2] *= scale_x
            K_new[1, 2] *= scale_y

            rgb_t   = torch.from_numpy(rgb_src.astype(np.float32) / 255.0).to(d.rgb.device, d.rgb.dtype)
            depth_t = torch.from_numpy(depth_np.astype(np.float32)).to(d.depth.device, d.depth.dtype)
            K_t     = torch.from_numpy(K_new).to(d.K.device, d.K.dtype)
            c2w_t   = torch.from_numpy(np.asarray(ss["c2w"], dtype=np.float32)).to(d.c2w.device, d.c2w.dtype)

            new_idx = d.append_frame(rgb_t, depth_t, K_t, c2w_t)

            # Extend the per-frame loss mask. Supervise the whole frame — depth
            # is from the splat render itself so the "valid depth" gate isn't
            # meaningful here; the inpainted region is the part we care about.
            init = getattr(t, "init", None)
            if init is not None:
                mask = torch.ones((H, W), device=init.train_mask.device, dtype=init.train_mask.dtype)
                init.append_train_mask(mask)

            self.gui_add_frame_status.content = (
                f"_Appended frame #{new_idx} ({src_tag}) → trainer.data now has {d.N} frames._"
            )
            print(f"  inpainter: appended frame #{new_idx} (src={src_tag}); data.N={d.N}",
                  file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            self.gui_add_frame_status.content = f"_Add-frame error: {e}_"
            traceback.print_exc()

    def _get_pipeline(self, model_id: str):
        """Lazy-load + cache the diffusers pipeline. Picks the right class
        from the model id (case-insensitive) and the right dtype per family.
        Thread-safe: the cache + load are guarded by `_pipeline_lock` so
        the boot-time preload and an early user click can't double-load."""
        with self._pipeline_lock:
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

    def reset(self) -> None:
        """Clear all captured / generated images + status lines + on-disk
        artifacts so the Inpaint tab matches the freshly-booted state.
        Called from `train_and_view.resetter()` when the user clicks
        Reset on the Setup panel — keeping stale captures around would
        lie about which scene the inpaint references belong to.

        Does NOT touch the cached diffusers pipeline (`_kontext_pipeline`)
        — that's session-scoped and reloading would defeat the preload.
        The provenance ledger is also cleared since the splats those
        entries point to no longer exist after trainer.reset()."""
        placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
        # Image fields
        for h in (
            getattr(self, "gui_screenshot", None),
            getattr(self, "gui_mask_preview", None),
            getattr(self, "gui_neighbors_preview", None),
            getattr(self, "gui_inpaint_preview", None),
        ):
            if h is not None:
                try:
                    h.image = placeholder
                except Exception:
                    pass
        # Status lines back to boot defaults
        for h, msg in (
            (getattr(self, "gui_status", None), "_no capture yet_"),
            (getattr(self, "gui_instainpaint_status", None),
                "_no instainpaint run yet_"),
            (getattr(self, "gui_add_frame_status", None),
                "_no frame appended_"),
        ):
            if h is not None:
                try:
                    h.content = msg
                except Exception:
                    pass
        # Captured / generated state
        self.last_screenshot = None
        self.last_neighbors = None
        self.last_inpaint = None
        self.gaussian_provenance = []
        self._capture_counter = 0

        # On-disk artifacts written by capture + inpaint runs. We only
        # delete files matching our own glob patterns so user-placed
        # files in the same directory survive. Captured + masked +
        # generated images + the debug pipe intermediates + per-run
        # JSON metadata sidecars.
        try:
            out_dir = self._out_dir()
            if out_dir.is_dir():
                patterns = (
                    "screenshot_*.png",
                    "inpaint_*.png",
                    "inpaint_*.mask.png",
                    "inpaint_*.meta.json",
                    "_dbg_pipe_*.png",
                )
                removed = 0
                for pat in patterns:
                    for p in out_dir.glob(pat):
                        try:
                            p.unlink()
                            removed += 1
                        except Exception as e:
                            print(f"[inpainter.reset] unlink {p} failed: {e}",
                                  file=sys.stderr)
                print(f"[inpainter.reset] removed {removed} on-disk "
                      f"artifact(s) from {out_dir}",
                      file=sys.stderr)
        except Exception as e:
            print(f"[inpainter.reset] disk cleanup skipped: {e}",
                  file=sys.stderr)

    def _on_load_model_click(self) -> None:
        """Manual 'Load model' button: warm the diffusers pipeline on demand.

        Unlike `start_preload`, this is NOT gated on the `_preload` flag, so it
        works when auto-preload is off (the default — FLUX is ~25GB and OOMs
        alongside training, so we no longer load it at boot). Loads in a daemon
        thread so the UI stays responsive, and mirrors progress into the status
        markdown. Requires DA3 to have loaded first (initialize a scene),
        because loading FLUX before DA3 leaves accelerate's init_empty_weights
        active and breaks DA3's `.to(device)` — see `start_preload`."""
        model_id = str(self.gui_model_id.value)
        if not getattr(self._trainer_ref, "_initialized", False):
            self.gui_model_status.content = (
                "**model:** initialize a scene first — FLUX must load after "
                "DA3 or DA3's weight-init breaks"
            )
            return
        with self._pipeline_lock:
            already = (self._kontext_pipeline is not None
                       and self._kontext_model_id == model_id)
        if already:
            self.gui_model_status.content = f"**model:** already loaded ({model_id})"
            return
        if self._model_loading:
            self.gui_model_status.content = "**model:** load already in progress…"
            return
        self._model_loading = True
        self.gui_model_status.content = (
            f"**model:** loading {model_id} (~25GB, ~30-60 s)…"
        )

        def _worker() -> None:
            try:
                t0 = time.time()
                self._get_pipeline(model_id)  # lock-guarded cache + load
                self.gui_model_status.content = (
                    f"**model:** loaded ({model_id}) in {time.time() - t0:.0f}s"
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.gui_model_status.content = (
                    f"**model:** load failed — {type(e).__name__}: {e}"
                )
            finally:
                self._model_loading = False

        threading.Thread(target=_worker, daemon=True, name="inpainter-load").start()

    def start_preload(self) -> None:
        """Public hook to kick off the inpainter pipeline preload.
        Idempotent — only spawns the daemon thread on the first call (per
        session). No-op when `preload=False` was passed at construction.

        Why deferred: loading FLUX-Klein via diffusers before DA3 leaves
        accelerate's `init_empty_weights` global state active for the next
        `from_pretrained`; DA3 then constructs with meta params that fail
        the subsequent `.to(device)` call. Letting DA3 load first
        (inside the first `trainer.prepare_and_init`) keeps DA3 well-
        formed. Call this from the host after Initialize completes."""
        if not self._preload:
            return
        if self._preload_started:
            return
        self._preload_started = True
        try:
            self._spawn_preload(str(self.gui_model_id.value))
        except Exception as e:
            print(f"[inpainter] preload spawn skipped: {e}", file=sys.stderr)
            self._preload_started = False  # let caller retry on next init

    def _spawn_preload(self, model_id: str) -> None:
        """Run `_get_pipeline(model_id)` in a daemon thread so the user's
        first click on Inpaint doesn't block on the ~30-60 s model load.
        If the user changes `gui_model_id` between boot and the first
        click, the click pays for the new model — preload only warms the
        default. Idempotent: if the lock-protected cache already matches,
        the worker returns immediately."""
        def _worker() -> None:
            try:
                t0 = time.time()
                print(f"[inpainter] preloading {model_id!r} on a background thread…",
                      file=sys.stderr, flush=True)
                self._get_pipeline(model_id)
                dt = time.time() - t0
                print(f"[inpainter] preload of {model_id!r} done in {dt:.1f}s",
                      file=sys.stderr, flush=True)
            except Exception as e:
                import traceback
                print(f"[inpainter] preload failed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                traceback.print_exc()
        t = threading.Thread(target=_worker, daemon=True, name="inpainter-preload")
        t.start()

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
