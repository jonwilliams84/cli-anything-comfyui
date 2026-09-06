# Test plan — cli-anything-comfyui

## What makes this harness hard to test honestly

The backend is a **running ComfyUI holding models in VRAM**. There is no
headless convert-a-file mode to shell out to, so an E2E test either drives a
real server or proves nothing. Per HARNESS.md there is **no graceful
degradation**: if ComfyUI is not reachable the E2E tests FAIL, they do not skip.

The instance these were written against: ComfyUI Desktop 0.34.5 on Windows,
RTX 3090, reachable from WSL at `127.0.0.1:8188`, 1805 node types installed.

## Inventory plan

- `test_core.py` — ~92 unit tests, synthetic graphs and a fake client, no server
- `test_full_e2e.py` — ~25 tests against the real server, including subprocess

## Unit test plan (`test_core.py`)

### `core/workflow.py` — the converter, where the value is

- `detect_format` — UI, API, and neither
- **`control_after_generate`**: a KSampler-shaped node whose 7 `widgets_values`
  must land on 6 named inputs with the phantom companion swallowed. This is the
  off-by-one that four hand-rolled `wf2api` scripts got wrong; if this test
  passes and the others fail, the harness is still worth having.
- widget vs link classification: INT/FLOAT/STRING/BOOLEAN/COMBO are widgets,
  everything else arrives over a link
- link resolution: `[origin_id, origin_slot]` built from the positional link rows
- **bypassed (mode 4) nodes** are collapsed and their consumers rewired to the
  real source
- **Reroute** nodes collapsed the same way
- a **link cycle** raises rather than recursing for ever
- muted (mode 2) dropped by default, kept with `keep_muted`
- canvas-only nodes (Note, MarkdownNote) dropped and REPORTED
- missing node types collected into ONE error naming all of them, not the first
- subgraph instances (UUID class_type) named from `definitions.subgraphs`
- a dangling link to a dropped node is removed and warned about
- **report shape is identical** on the already-API path (the bug that made a
  caller KeyError depending on input format)
- `validate` finds: unknown class_type, missing required input, dangling wire
- `outputs` finds output nodes; empty means the graph writes nothing
- `set_input` on a missing node raises with the ids that DO exist
- `find_nodes` by class_type and by title
- `unset_input` removes an override and errors on a missing node OR input,
  naming what exists
- `diff_graphs` reports added/removed nodes and per-input changes (a rewire is
  just an input whose `[id, slot]` value changed); identical graphs are `same`

### `core/session.py`

- round-trip save/load; `_path` survives
- `--dry-run` does not write
- a corrupt session file recovers rather than wedging every command
- concurrent saves do not interleave (locked write)

### `utils/comfyui_backend.py` (no server)

- `ComfyUnavailable` names the URL and how to start one
- `ComfyPromptRejected` flattens nested `node_errors` into readable lines
- `outputs_of` reads **every** bucket — images, gifs, videos, audio — not just
  images, which is the bug that makes video workflows look empty
- queue row helpers tolerate short/garbage rows
- `_http_only` pins the scheme to http/https — `file://` and `ftp://` refused
  at the one place a URL enters the object
- the HTTP error paths, with `urlopen` faked: an HTTP error surfaces the body,
  a 400 on `POST /prompt` becomes `ComfyPromptRejected`, a dead server becomes
  `ComfyUnavailable`, and a non-JSON reply is an error, not a crash
- `submit` also raises `ComfyPromptRejected` when the 200 reply carries
  `node_errors`
- `wait` times out with a message that says so, and reports the live queue
  position through `on_tick`
- `upload_mask` builds the multipart body with `original_ref` (a mask without
  it masks nothing) and refuses a missing file
- `logs` builds `/internal/logs` with and without a limit

### `core/run.py` (fake client, no server)

- `submit_and_wait` flattens outputs from every bucket; a reply with no
  `prompt_id` raises rather than silently "succeeding"
- `fetch` re-stats every file on THIS disk: a zero-byte download lands in
  `empty`, not in the downloaded count
- `run_windows` keeps going when a window fails, frees VRAM between windows
  (even after a failure), honours `free_between=False`, and reports each window
  through `on_window`

### the CLI (`CliRunner`, no server)

- `--version`; `traps` list, one in full, and an unknown id naming the known ones
- `status` reports session state with the server down, without dying
- `workflow set` patches the session and `workflow find` sees it; with no
  workflow loaded it says so and exits 1
- `workflow convert` with no server fails loudly, naming the URL
- `run` queues the session graph, downloads, and records the `prompt_id`
- `windows` runs every graph, frees between windows, downloads the outputs,
  and exits non-zero when a window failed
- `server features` / `server embeddings` reach the server; with no server they
  name the problem
- `workflow export -o` writes the PATCHED session graph to disk; without a
  workflow loaded it says so and exits 1
- `workflow diff FILE` compares the file (baseline) against the patched session
  graph; two paths diff file A against file B; with nothing to compare, or no
  loaded graph, it says so
- `workflow unset` removes the override from the session and errors name what
  exists
- `assets mask` uploads with the original ref recorded
- `server logs` reports the recent log lines

## E2E plan (`test_full_e2e.py`) — real server required

1. `server status` reports a version and at least one device
2. `/object_info` returns >100 node types; KSampler's schema has the
   `control_after_generate` flag this harness depends on
3. **The 48-canvas corpus**: every saved workflow in the archive either converts,
   or reports missing packs/subgraphs — none may raise an unexpected exception
4. A converted graph **validates against the live schema**
5. **A real render**: build a minimal graph from installed nodes, submit it,
   wait, and verify the produced file — non-zero bytes and PNG magic `\x89PNG`
6. The queue is visible during and empty after
7. `server free` returns cleanly
8. Upload a generated PNG and get the server's stored name back
9. Download a produced file and verify bytes on disk
10. `/internal/logs` answers (newer builds)
11. Render, fetch the PNG, upload it back as a MASK for itself — the inpainting
    path is exercised against the real endpoint

### Subprocess tests (the installed command)

`_resolve_cli("cli-anything-comfyui")`, no `cwd` set, so the installed command
must work from anywhere:

- `--help`, `--version`
- `--json server status` parses and carries `comfyui_version`
- `--json nodes schema KSampler` shows the control_after_generate flag
- `--json workflow convert <real canvas>` then `workflow info` finds the output node
- `--json traps` lists the recorded failure modes
- full workflow: convert -> set an input -> validate -> run -> outputs exist
- round trip: set -> export -> diff -> unset, ending identical to the source
- `server logs` parses
- `workflow unset` on an unknown input fails loudly, naming what exists

## Realistic workflow scenarios

**Scenario A — "make this canvas runnable"**
Simulates: the migration this harness was written during. Convert a saved canvas,
discover which node packs are missing, convert the rest, validate.
Verified: the missing-pack list matches what the archive corpus reports.

**Scenario B — "render and check it"**
Simulates: an agent producing an image and confirming it exists.
Chained: convert -> set seed -> run -> history outputs -> download.
Verified: PNG magic bytes, non-zero size, path printed for inspection.

**Scenario C — "the graph that writes nothing"**
Simulates: the quietest failure in ComfyUI.
Chained: strip the output node -> `workflow info`.
Verified: reports zero output nodes and warns.

---

# Test results

Run: `CLI_ANYTHING_FORCE_INSTALLED=1 python -m pytest cli_anything/comfyui/tests/ -v --tb=no`
Against: ComfyUI Desktop 0.34.5 on Windows (RTX 3090), 1805 node types, from WSL.
Date: 2026-09-06.

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/jon/.venvs/ccu/bin/python
cachedir: .pytest_cache
rootdir: /home/jon/projects/github/jonwilliams84/cli-anything-comfyui
collecting ... collected 50 items

cli_anything/comfyui/tests/test_core.py::test_detect_format[data0-ui] PASSED [  2%]
cli_anything/comfyui/tests/test_core.py::test_detect_format[data1-api] PASSED [  4%]
cli_anything/comfyui/tests/test_core.py::test_detect_format[data2-unknown] PASSED [  6%]
cli_anything/comfyui/tests/test_core.py::test_detect_format[data3-unknown] PASSED [  8%]
cli_anything/comfyui/tests/test_core.py::test_control_after_generate_does_not_shift_every_later_widget PASSED [ 10%]
cli_anything/comfyui/tests/test_core.py::test_widgets_and_links_are_classified_from_the_schema PASSED [ 12%]
cli_anything/comfyui/tests/test_core.py::test_a_link_becomes_an_origin_pair PASSED [ 14%]
cli_anything/comfyui/tests/test_core.py::test_a_bypassed_node_is_collapsed_and_its_consumer_rewired PASSED [ 16%]
cli_anything/comfyui/tests/test_core.py::test_a_reroute_is_collapsed_into_its_consumer PASSED [ 18%]
cli_anything/comfyui/tests/test_core.py::test_a_link_cycle_raises_instead_of_recursing_for_ever PASSED [ 20%]
cli_anything/comfyui/tests/test_core.py::test_muted_nodes_are_dropped_by_default_and_kept_on_request PASSED [ 22%]
cli_anything/comfyui/tests/test_core.py::test_canvas_only_nodes_are_dropped_and_reported PASSED [ 24%]
cli_anything/comfyui/tests/test_core.py::test_every_missing_node_type_is_named_at_once PASSED [ 26%]
cli_anything/comfyui/tests/test_core.py::test_a_subgraph_is_named_not_called_a_missing_pack PASSED [ 28%]
cli_anything/comfyui/tests/test_core.py::test_a_wire_to_a_dropped_node_is_removed_and_warned_about PASSED [ 30%]
cli_anything/comfyui/tests/test_core.py::test_the_already_api_path_returns_the_same_report_shape PASSED [ 32%]
cli_anything/comfyui/tests/test_core.py::test_a_workflow_that_is_neither_format_says_so PASSED [ 34%]
cli_anything/comfyui/tests/test_core.py::test_validate_finds_unknown_types_missing_inputs_and_dangling_wires PASSED [ 36%]
cli_anything/comfyui/tests/test_core.py::test_validate_passes_a_complete_graph PASSED [ 38%]
cli_anything/comfyui/tests/test_core.py::test_outputs_finds_the_nodes_that_write_files PASSED [ 40%]
cli_anything/comfyui/tests/test_core.py::test_set_input_on_a_missing_node_names_the_ones_that_exist PASSED [ 42%]
cli_anything/comfyui/tests/test_core.py::test_find_nodes_by_type_and_title PASSED [ 44%]
cli_anything/comfyui/tests/test_core.py::test_session_round_trips PASSED [ 46%]
cli_anything/comfyui/tests/test_core.py::test_dry_run_writes_nothing PASSED [ 48%]
cli_anything/comfyui/tests/test_core.py::test_a_corrupt_session_recovers_rather_than_wedging_every_command PASSED [ 50%]
cli_anything/comfyui/tests/test_core.py::test_concurrent_saves_do_not_interleave PASSED [ 52%]
cli_anything/comfyui/tests/test_core.py::test_unavailable_names_the_url_and_how_to_start_one PASSED [ 54%]
cli_anything/comfyui/tests/test_core.py::test_a_rejected_prompt_is_flattened_into_readable_lines PASSED [ 56%]
cli_anything/comfyui/tests/test_core.py::test_outputs_of_reads_every_bucket_not_just_images PASSED [ 58%]
cli_anything/comfyui/tests/test_core.py::test_outputs_of_survives_an_empty_or_odd_entry PASSED [ 60%]
cli_anything/comfyui/tests/test_core.py::test_queue_row_helpers_tolerate_short_rows PASSED [ 62%]
cli_anything/comfyui/tests/test_full_e2e.py::test_the_server_reports_a_version_and_a_device PASSED [ 64%]
cli_anything/comfyui/tests/test_full_e2e.py::test_object_info_is_populated_and_ksampler_carries_the_flag PASSED [ 66%]
cli_anything/comfyui/tests/test_full_e2e.py::test_free_vram_returns_cleanly PASSED [ 68%]
cli_anything/comfyui/tests/test_full_e2e.py::test_every_archived_canvas_converts_or_explains_itself PASSED [ 70%]
cli_anything/comfyui/tests/test_full_e2e.py::test_a_real_render_produces_a_real_png PASSED [ 72%]
cli_anything/comfyui/tests/test_full_e2e.py::test_the_queue_is_readable_and_history_carries_the_prompt PASSED [ 74%]
cli_anything/comfyui/tests/test_full_e2e.py::test_a_graph_with_no_output_node_is_identified_before_it_wastes_a_render PASSED [ 76%]
cli_anything/comfyui/tests/test_full_e2e.py::test_upload_then_download_round_trips PASSED [ 78%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_help PASSED [ 80%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_version PASSED [ 82%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_server_status_json PASSED [ 84%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_nodes_schema_marks_widgets_and_the_flag PASSED [ 86%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_nodes_search_finds_something PASSED [ 88%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_traps_are_listed PASSED [ 90%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_models_lists_folders PASSED [ 92%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_convert_then_info_on_a_real_canvas PASSED [ 94%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_full_workflow_convert_set_validate_run PASSED [ 96%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_dry_run_does_not_write_the_session PASSED [ 98%]
cli_anything/comfyui/tests/test_full_e2e.py::TestCLISubprocess::test_a_missing_node_type_fails_loudly PASSED [100%]

============================== 50 passed in 8.94s ==============================
```

## Summary

- **50 tests, 100% pass** — 31 unit, 19 E2E (of which 11 drive the installed command)
- Two of them perform a **real render** and verify PNG magic bytes on the
  downloaded file, not just a zero exit code
- The 48-canvas corpus converts as: **15 clean, 25 need node packs, 5 use
  subgraphs, 3 were already API format** — every one an explained outcome, none
  an unhandled exception

## Coverage notes

Covered: the converter in depth (control_after_generate, bypass/reroute
collapsing, cycles, muting, missing packs, subgraphs, dangling wires, report
shape), validation, session persistence including locking and corruption
recovery, the backend's error surfaces, and the full convert → set → validate →
run → download path through the installed command.

Not covered, and honestly so:

- **Subgraph expansion** is not implemented. Five archived canvases use
  subgraphs; the harness names them and refuses rather than converting them
  wrongly.
- **`review`** (contact sheets / frame sampling of produced video) is designed
  in COMFYUI.md but not built. 45 of the 116 archived scripts shell out to
  ffmpeg, so this is the largest remaining gap.
- **Windowed rendering** (`windows`) is exposed as a command and covered by
  unit tests against a fake client (failure isolation, VRAM freeing between
  windows, download), but has no E2E test — a truthful one needs a
  multi-minute video model load.
- No test asserts behaviour when ComfyUI dies *mid-render*.

---

# Refine pass — 2026-09-06

Ran the CI gate exactly as the pipeline does:

```
python -m pytest cli_anything/comfyui/tests/test_core.py --cov=cli_anything
  --cov-fail-under=30 -q --durations=10
ruff check cli_anything/ --output-format=github
ruff format --check --diff cli_anything/
bandit -r cli_anything/ -ll -x '*/tests/*,*/test_*.py,*/conftest.py'
```

Result: **exit 0** — 60 unit tests pass, coverage **69.98%** (was 34.00%),
lint clean, format clean, bandit clean.

New since the 0.1.0 build: unit coverage of `core/run.py` (96%), the backend's
HTTP transport (79%), and the CLI command layer (52%); the `windows`,
`server features` and `server embeddings` commands.

## Refine pass — 2026-09-06

Same gate, after the refine pass:

```
75 passed, coverage 85.52% total
```

- `comfyui_cli.py` 52% → 77%: every command group's handlers are now exercised
  — `queue list/cancel/clear`, `history list/outputs` (including the session
  fallback and the unknown-prompt error), `assets upload/download`, `models`
  (folders and one folder), `nodes list/search/schema`, `server free
  (--keep-models)/interrupt`, `workflow validate/info/deps`.
- `comfyui_backend.py` 79% → 97%: the hand-built multipart body in
  `upload_image` (fields, filename, boundary, the skipped empty-subfolder
  field), `submit`'s `front`/`extra_data`, the exact verb/path/body of
  `free`/`clear_queue`/`cancel`/`interrupt`, and the query paths built by
  `history`/`models`/`object_info(node_class)`.
- `core/run.py` 96% → 100%: a failed `free()` between windows is reported as a
  warning on the window result instead of killing the run.
- Two new commands, unit-tested: `nodes categories` (discovery half of
  `nodes list --category`) and `workflow outputs` (the file-writing nodes of
  the loaded graph or a `--path`, composes with `run --download`).
