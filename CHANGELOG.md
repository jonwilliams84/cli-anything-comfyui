# Changelog

All notable changes to this project are documented here.

## [0.15.0] — 2026-09-06

One new command, closing the third leg of pre-flight before a render.

- **`workflow models [PATH]`** — whether the model FILES a graph names are on
  this server. `workflow deps` checks node TYPES and `workflow validate` checks
  graph SHAPE; both pass a graph whose CheckpointLoaderSimple names a checkpoint
  that is not on disk, and that graph queues clean and dies seconds into
  execution. Every widget input ending in `_name` (ckpt_name, lora_name,
  vae_name, clip_name, unet_name, control_net_name, model_name, …) is checked
  against the server's own `GET /models` listings across ALL folders —
  membership anywhere, not in a predicted folder, because the folder a node
  pack files its models under is not guessable. Exits 1 when something is
  missing. A graph with no model inputs never polls /models at all.
- New core functions in `core/workflow.py`: `model_refs`, `index_models`,
  `check_models`.
- 9 new unit tests in `test_core.py` (202 total, all passing; 100% statements
  and branches), 3 new E2E subprocess tests in `test_full_e2e.py`.
- Docs: README.md, cli_anything/comfyui/README.md, COMFYUI.md, TEST.md.

## [0.14.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py` and 2 more. (9 files changed, 314 insertions(+), 4 deletions(-))

## [0.13.0] — 2026-09-06

- Updated `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (3 files changed, 211 insertions(+))

## [Unreleased] — refine round 14

One new command, aimed at the question every failed render asks and no existing
command could answer: WHY did it fail.

- **`history show [PROMPT_ID] [--graph PATH]`** — one history entry in full:
  status, the server's execution messages (the `execution_error` rows with node
  id, node type, exception type and message — recorded ONLY in the entry's
  status.messages, which `history list` and `history outputs` both hide), and
  its output files. Defaults to the session's `last_prompt_id`; exits 1 when
  the entry reports an error so `history show && …` chains stop at the failure.
- `--graph PATH` writes the exact API-format graph the server ran — re-runnable
  with `run --path`, diffable with `workflow diff` — even for a failed entry.
- 5 new unit tests in `test_core.py` (failed-entry payload + exit code, the
  session-id default, `--graph` write/refuse, no-id/no-entry errors, malformed
  and overflowing status.messages rows). 184 total, all passing. Coverage:
  100% statements, 100% branches. No existing test weakened, skipped or deleted.

## [Unreleased] — refine round 13

Coverage-honesty pass — no new commands, no CLI behaviour change.

- **8 new unit tests in `test_core.py`** pinning the 11 partially-covered
  branches that were the only dark paths left (99.36% → 100% statement AND
  branch coverage, 178 tests, all passing): `workflow convert` without `-o`
  and with `--no-load`; `workflow set --raw`; `workflow validate` on a graph
  that passes (CLI exit code 0); junk link rows in `link_map`; a bypassed node
  passing its first WIRED input through past unwired/unknown-link inputs; a
  widget that is also wired (the wire wins and the positional cursor still
  advances past the `control_after_generate` companion); `upload_mask` with a
  dict `original_ref` and a None `kind`; `userdata_put` with raw bytes.
- No existing test or check was weakened, skipped or deleted.

## [0.12.0] — 2026-09-06

Maintenance release — no new commands, no CLI behaviour change.

- **E2E test fix: `assets upload --overwrite false`** — the v0.11.0 test
  asserted that a second upload of the same filename raises (a 409-style
  refusal). ComfyUI has no such response: it renames the new file
  (`name (1).png`, `name (2).png`, …) — except when the bytes are identical to
  what is already on disk, in which case it keeps the existing name and writes
  nothing. The test now pins that real contract against a live server: same
  bytes keep the original name, different bytes come back as
  `cli-anything-e2e (1).png`. Found by the post-merge E2E check.
- Version bumped to 0.12.0 (`cli_anything/comfyui/__init__.py`, `setup.py`).

## [0.11.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/core/run.py`, `cli_anything/comfyui/skills/skill.md` and 5 more. (11 files changed, 264 insertions(+), 17 deletions(-))

## [0.10.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/core/run.py`, `cli_anything/comfyui/skills/skill.md`, `cli_anything/comfyui/tests/test.md` and 1 more. (8 files changed, 156 insertions(+), 2 deletions(-))

## [0.9.0] — 2026-09-06

- Updated `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (2 files changed, 268 insertions(+))

## [0.8.0] — 2026-09-06

- Updated `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (2 files changed, 233 insertions(+))

## [0.7.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/tests/test_core.py`, `cli_anything/comfyui/utils/comfyui_backend.py`. (5 files changed, 203 insertions(+), 4 deletions(-))

## [0.6.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`, `cli_anything/comfyui/tests/test_full_e2e.py` and 1 more. (7 files changed, 338 insertions(+), 2 deletions(-))

## [0.5.0] — 2026-09-06

- Updated `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (3 files changed, 390 insertions(+), 2 deletions(-))

## [0.4.0] — 2026-09-06

- Updated `comfyui.md`, `cli_anything/comfyui/readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/core/workflow.py`, `cli_anything/comfyui/skills/skill.md`, `cli_anything/comfyui/tests/test.md` and 4 more. (11 files changed, 656 insertions(+), 14 deletions(-))

## [0.3.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (7 files changed, 396 insertions(+), 2 deletions(-))

## [0.2.0] — 2026-09-06

- Updated `comfyui.md`, `readme.md`, `cli_anything/comfyui/readme.md`, `cli_anything/comfyui/comfyui_cli.py`, `cli_anything/comfyui/tests/test.md`, `cli_anything/comfyui/tests/test_core.py`. (7 files changed, 556 insertions(+), 8 deletions(-))

## [Unreleased]

Fifth refine pass — the wait half of the render loop, standalone:

- **`queue wait PROMPT_ID [--timeout S] [--poll S] [--download DIR]`** — await a
  prompt that is ALREADY on the queue (queued from the canvas, another agent or
  an earlier shell) and report its outputs from every bucket. `run` only waits
  for prompts it submitted itself; this closes that gap with new core function
  `run_core.wait_and_collect`, composed with `fetch` for `--download`.
- `__main__.py` — the last dark statement in the repo — now covered.
- Unit suite expanded from 158 to 163 tests.

Third refine pass — the patch loop and the two missing server surfaces:

- **`workflow unset`** — the honest inverse of `workflow set`: removes an input
  override so the node falls back to its schema default or its link, instead of
  baking a guessed replacement value in. Auto-saves unless `--dry-run`.
- **`workflow export -o FILE`** — writes the CURRENT session graph (every
  `workflow set` included) to disk. `workflow convert -o` only wrote the freshly
  converted canvas; patches used to be trapped in the session.
- **`workflow diff [A] [B]`** — what changed between two graphs, node by node
  and input by input (a rewire is just an input whose `[id, slot]` value
  changed). One path diffs that file, as baseline, against the patched session
  graph: "what did I change since the render that worked".
- **`assets mask FILE ORIGINAL_REF`** — `POST /upload/mask`, the inpainting
  path. Without `original_ref` the server files the mask as a loose picture and
  the inpaint graph masks nothing.
- **`server logs`** — `GET /internal/logs` (newer ComfyUI builds): a rejected
  prompt says WHAT was refused; the log says what happened around it.
- Unit suite expanded from 75 to 92 tests; total coverage 85.52% → 86.64% with
  ~120 new statements. E2E suite gained the `/internal/logs` read, a real mask
  upload against a rendered image, and a set → export → diff → unset
  subprocess round trip that must end identical to its source.

Fourth refine pass — no new commands; the converter's edge paths went from
dark lines to pinned behaviour:

- **Dict-shaped link rows** (`{id, origin_id, ...}`, the newer canvas
  serialisation) are read by `link_map`; previously such a workflow converted
  with every input silently unwired.
- **Dangling and dead links are warned, never silent**: a link whose origin
  node no longer exists, an input whose link id the map never heard of, and a
  bypassed/reroute chain with nothing behind it each drop the input WITH a
  reason. An input with `link: null` is skipped quietly.
- **`widgets_values` by name** (`{seed: 5}`, some packs) skips the positional
  pass; **surplus widget values** (frontend-only widget, pack drift) are
  ignored but reported with `extra_values` and the likely cause.
- **Node titles** travel into `_meta` (what `workflow find --title` reads), and
  `find_nodes` sorts digit ids numerically (`2` before `10`).
- Backend edges: a NON-JSON HTTP error body (a proxy's HTML 502) still surfaces
  its text; an empty reply is `{}`; `wait` without `on_tick` still polls and
  times out; `session.load(url=...)` lets `--url` beat the stored URL.
- CLI edges pinned: `server status` in both renderers, `workflow convert`
  end-to-end over a fake `/object_info`, a rejected `run` exits 1 and records
  no prompt id, and `workflow validate` exits 1 on an invalid graph.
- Unit suite expanded from 92 to 114 tests; total coverage 86.64% → 92.24%,
  with `core/workflow.py` at 99%, `core/session.py` and `core/run.py` at 100%.

Second refine pass — CLI surface completeness and coverage of what already shipped:

- **`nodes categories`** — the discovery half of `nodes list --category`: the
  category prefixes installed on the server, with node counts, biggest first.
- **`workflow outputs`** — the nodes in the loaded graph (or a `--path` file)
  that actually write files; a graph with none runs, succeeds and saves
  nothing. Composes with `run --download`.
- Unit suite expanded from 60 to 75 tests; total coverage 69.98% → 85.52%
  (`comfyui_cli.py` 52% → 77%, `comfyui_backend.py` 79% → 97%, `core/run.py`
  to 100%). Previously untested handlers now exercised: `queue`, `history`,
  `assets`, `models`, `nodes`, `server free/interrupt`, `workflow
  validate/info/deps`; backend multipart upload, mutating endpoints, and
  request-path construction.

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
