# Changelog

All notable changes to this project are documented here.

## [0.1.0] — 2026-09-06

First release. A CLI to a running ComfyUI, built with the cli-anything harness
methodology against ComfyUI Desktop 0.34.5 (Windows, RTX 3090, 1805 node types).

- **`workflow convert`** — a saved canvas becomes a runnable API graph, converted
  against the LIVE `/object_info` so widget names and order match the node packs
  actually installed. Handles the `control_after_generate` off-by-one that gives
  a KSampler seven `widgets_values` for six inputs, collapses bypassed (mode 4)
  and Reroute nodes and rewires their consumers, drops muted and canvas-only
  nodes with a report, and guards against link cycles.
- **`workflow deps`** — which node packs a workflow needs that this ComfyUI does
  not have. Names every missing type in one run, and distinguishes a subgraph
  (bare-UUID `class_type`) from a missing pack.
- **`workflow info` / `validate` / `find` / `set`** — output-node detection
  (a graph with none writes nothing), pre-flight validation, node lookup by type
  or title, and single-input patching with auto-save.
- **`run`** — queue, wait, and report outputs from every bucket: images, gifs,
  videos and audio.
- **`server`, `nodes`, `queue`, `history`, `assets`, `models`, `traps`, `status`**
- REPL by default; `--json` on every command; session with locked writes,
  auto-save and `--dry-run`.
- 50 tests, all passing: 31 unit and 19 E2E against a real ComfyUI, two of which
  render an image and verify PNG magic bytes on the downloaded file.

### Not implemented

Subgraph expansion, `review` (contact sheets from produced video), and a CLI
surface for windowed rendering. Stated in the README and TEST.md rather than
implied.
