# Splat Viewer — Phased Implementation Plan

A viser-based 3D Gaussian Splatting viewer with GUI controls and a path to
live training visualization. Built in phases so each can be validated before
the next.

## How to use these with Claude Code

One phase per session. Hand Claude Code the relevant `phase_N_*.md` and let
it implement against that single document. Don't paste multiple phases in
one go — they reference each other (later phases assume earlier
infrastructure exists), but the implementation discipline of completing one
before starting the next is what keeps the design clean.

After each phase: run the "Done means" checklist before moving on. The next
phase's "Prerequisites" section will tell you what it expects.

## Phases

- **`phase_0_scaffolding.md`** — viser server, GUI loop, synthetic image,
  threading-ready `SceneState`. No splats yet. ~1 evening.
- **`phase_1_static_ply.md`** — load a real `.ply`, render it correctly,
  nail down the camera convention. ~1 day with the convention sanity-check.
- **`phase_2_gui_controls.md`** — SH degree, render mode (RGB/depth), FOV,
  FPS readout, reset camera. Brings it to feature parity with the native
  ImGui plan. ~1 day.
- **`phase_3_adaptive_resolution.md`** — motion detection, low-res during
  motion, full-res at idle. Skip if viser's defaults already work for your
  scenes. ~1 evening.
- **`phase_4_niceties.md`** — multi-scene loading, bookmarks, screenshots,
  click-to-inspect, crop box. Pick what you actually need. À la carte.
- **`phase_5_live_training.md`** — background training thread, lock-protected
  state swap, loss plot. The whole point of the threading-ready design from
  Phase 0. ~2 days.

## Design choices made up front (reference)

- **viser** owns the window, camera, and GUI. We don't own a `FlyCamera` class.
- **gsplat** (Apache 2.0) for rasterization. Inria's `diff-gaussian-rasterization`
  is research-license only and is blocked.
- **OpenCV camera convention** at the gsplat boundary; the viser→OpenCV
  conversion is the one piece of math worth unit-testing.
- **`SceneState` with an `RLock`** from Phase 0, even when nothing else
  contends for it. Phase 5 plugs into it without touching render code.
- **Single render entry point** (`Renderer.render`). All features (depth,
  adaptive res, crop, training-mode-render) flow through it.
- **No CPU roundtrip on the per-frame path** except the final
  `.cpu().numpy()` at return.

## What's deliberately not in the plan

- A native (glfw/imgui) viewer. We considered it, decided viser was the right
  call for an internal tool that needs to work over SSH on a GPU box.
- CUDA-GL interop. Viser handles transport; the interop machinery doesn't
  apply here.
- Multi-user collaborative editing. Viser supports multiple clients with
  independent cameras over a shared scene; that's enough.
- Any web frontend code. Viser provides the frontend; we only write Python.

## Stack summary

- Python 3.10+
- `viser` (Apache 2.0)
- `gsplat` (Apache 2.0)
- `torch` (CUDA)
- `numpy`, `plyfile`
- (Phase 5) optional: a dataset loader matching your training recipe
