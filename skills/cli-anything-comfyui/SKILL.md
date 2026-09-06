---
name: "cli-anything-comfyui"
description: "Drive a running ComfyUI from the shell: convert a saved canvas into a runnable API graph (handling the control_after_generate off-by-one), find which node packs a workflow needs, patch inputs, queue renders, poll, and fetch outputs from every bucket — images, gifs, videos and audio. Use when an agent must operate ComfyUI without the browser canvas."
---

# cli-anything-comfyui

A command-line interface **to** a running ComfyUI. It renders nothing itself:
ComfyUI is a hard dependency and must be reachable.

## Before anything else

```bash
cli-anything-comfyui --json server status
```

`up: false` or a non-zero exit means there is no server. Start ComfyUI (the
Desktop app, or `python main.py --listen 0.0.0.0 --port 8188`) or set
`COMFYUI_URL`. Do not proceed — nothing else in this CLI can work.

## Command groups

- `server` — `status`, `free` (unload models / free VRAM), `interrupt`, `logs`
- `nodes` — `list`, `search <text>`, `schema <ClassType>`
- `workflow` — `convert`, `deps`, `info`, `find`, `set`, `unset`, `validate`,
  `export`, `diff`
- `run` — queue the loaded graph, wait, report outputs
- `queue` — `list`, `cancel <prompt_id>`, `clear`
- `history` — `list`, `outputs [prompt_id]`
- `assets` — `upload <file>`, `mask <file> <original_ref>`, `download <filename> <dest>`
- `models` — model folders and their contents
- `traps` — recorded failure modes for this software
- `status` — session + server state

Every command accepts `--json`. Run with no subcommand for a REPL.

## The workflow that matters

```bash
cli-anything-comfyui --json workflow deps canvas.json      # can this even run here?
cli-anything-comfyui --json workflow convert canvas.json   # UI -> API, loads into session
cli-anything-comfyui --json workflow info                  # does it have an output node?
cli-anything-comfyui --json workflow set 3 seed 42
cli-anything-comfyui --json run --download ./out
```

`workflow convert` reads a saved canvas (`user/default/workflows/*.json`) and
produces the API graph `POST /prompt` accepts. It converts against the LIVE
`/object_info`, because widget names and their order come from the node packs
installed on the machine that will run the graph.

## Guidance for agents

**Check `output_nodes` before running.** `workflow info` reports them. A graph
with none executes, reports success, and writes no file — the quietest way to
waste a render.

**Read every output bucket.** `run` and `history outputs` return `outputs[]`
where each entry has a `bucket` of `images`, `gifs`, `videos` or `audio`. Code
that looks only at images finds nothing for video workflows.

**`workflow deps` before a migration.** It answers "which node packs must I
install" from the workflow itself. Missing types come back in
`missing_node_types`; a bare-UUID `class_type` is a SUBGRAPH, reported under
`subgraphs`, not a missing pack.

**`server free` between long renders.** Models stay resident between prompts. A
windowed render that never frees VRAM fills the card and dies partway.

**Errors are the useful output.** A rejected prompt names the node, the input
and the reason. Read it rather than retrying blind.

**`--dry-run`** suppresses the session save for any mutation.

**`workflow diff FILE`** answers "what did I change since the render that
worked" — the file is the baseline, the patched session graph is the current
state. `workflow unset NODE INPUT` removes an override instead of guessing a
replacement; `workflow export -o out.json` writes the patched graph to disk.

**`assets mask FILE ORIGINAL_REF`** uploads an inpainting mask tied to the
image it belongs to. Without ORIGINAL_REF the server files the mask as a loose
picture and the inpaint graph masks nothing.

**`traps`** lists what has actually gone wrong on this estate, with the fix for
each. Read it before debugging from first principles.

## JSON shapes worth knowing

- `server status` → `{up, comfyui_version, os, devices[{name, vram_free_gb}], argv}`
- `workflow convert` → `{nodes, dropped[], warnings[], missing_node_types[], subgraphs[]}`
- `workflow info` → `{nodes, output_nodes[], by_class_type{}}`
- `workflow diff` → `{baseline, same, added[], removed[], changed[{node, class_type, changes[{input, from, to}]}]}`
- `run` → `{prompt_id, elapsed_s, output_count, outputs[{bucket, filename, subfolder, type}]}`
- `nodes schema X` → `inputs[{name, type, widget, control_after_generate, default, choices}]`

## Not implemented

Subgraph expansion (named and refused, never converted wrongly), `review`
(contact sheets from produced video), and a CLI surface for windowed rendering.
No preview bundle support.
