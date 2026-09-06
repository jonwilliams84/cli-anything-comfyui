# Changelog

All notable changes to this project are documented here.

## [0.2.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (7 files changed, 556 insertions(+), 8 deletions(-))

## [Unreleased]

Refine pass — coverage driven from the archived driver scripts' needs:

- **`windows`** — the OOM-guarded windowed render loop is now a command, not
  just a library function: several workflow files run in order, VRAM is freed
  between windows (`--keep-vram` to suppress), a failed window does not abort
  the rest, and `--download` collects every produced file. Fails with a
  non-zero exit if any window failed.
- **`server features` / `server embeddings`** — the backend's `features()` and
  `embeddings()` calls, which had no CLI surface, are now exposed.
- Unit suite expanded from 31 to 60 tests: `core/run.py` (`submit_and_wait`,
  `fetch` byte verification, `run_windows` failure/free semantics) and the
  backend's HTTP transport (`_http_only` scheme pinning, HTTP-error bodies,
  dead-server `ComfyUnavailable`, non-JSON replies, submit rejection,
  `wait` timeout and queue-position ticks, download) are now covered without a
  server, as are the CLI commands themselves via `CliRunner`.
  Overall coverage: 34% → 70%.

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
