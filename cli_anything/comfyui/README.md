# cli-anything-comfyui

A command-line interface to a **running ComfyUI**. It converts saved canvases
into runnable graphs, tells you which node packs a workflow needs, queues
renders, and verifies what came back.

It renders nothing itself. ComfyUI is a hard dependency.

## Install the software first

ComfyUI must be running and reachable. Either the Desktop app, or:

```bash
python main.py --listen 0.0.0.0 --port 8188
```

The harness defaults to `http://127.0.0.1:8188`. Point it elsewhere with
`--url` or `COMFYUI_URL`. A WSL shell reaches a Windows ComfyUI on
`127.0.0.1:8188` when WSL2 mirrored networking is on.

## Install the CLI

```bash
pip install -e .
cli-anything-comfyui --version
```

## Use it

```bash
cli-anything-comfyui server status              # version, devices, free VRAM
cli-anything-comfyui server features            # the server's feature flags
cli-anything-comfyui server logs                # the server's own recent log lines
cli-anything-comfyui nodes schema KSampler      # inputs: which are widgets, which are links
cli-anything-comfyui nodes categories           # category prefixes installed here, with counts
cli-anything-comfyui workflow deps my.json      # which node packs this canvas needs
cli-anything-comfyui workflow convert my.json   # canvas -> runnable API graph
cli-anything-comfyui workflow outputs my-api.json  # which nodes in a graph write files
cli-anything-comfyui workflow set 3 seed 42     # patch one input
cli-anything-comfyui workflow unset 3 seed      # remove one input override
cli-anything-comfyui workflow export -o p.json  # write the patched session graph to disk
cli-anything-comfyui workflow diff my.json      # what changed since that file was converted
cli-anything-comfyui run --download ./out       # queue, wait, fetch what it made
cli-anything-comfyui run --extra-data '{"filename": "job1"}'   # metadata with the prompt
cli-anything-comfyui history show canvas-pid                   # why a render failed: the server's execution messages
cli-anything-comfyui history show canvas-pid --graph ran.json  # the exact graph that ran, to re-run or diff
cli-anything-comfyui history clear --id canvas-pid             # prune one finished prompt
cli-anything-comfyui assets upload f.png --kind temp --no-overwrite
cli-anything-comfyui windows w1.json w2.json    # a windowed render: VRAM freed between
cli-anything-comfyui traps                      # the recorded ways this goes wrong
```

Run with no arguments for the REPL. Every command takes `--json`.

## What it is for

ComfyUI has three MCP servers already, and this harness deliberately does not
re-wrap them. It covers what they do not:

**Converting a saved canvas into something you can run.** A canvas stores widget
values POSITIONALLY; `POST /prompt` wants them under their real names, and the
names come from the node packs installed on the machine that will run it. The
harness converts against the live `/object_info` for that reason.

The conversion has a trap that four hand-rolled converters in this estate got
wrong: a widget with `control_after_generate` occupies **two** slots, so a real
KSampler stores seven values for six inputs. Consume one slot each and the graph
runs with `steps="randomize"`.

**Telling you what a workflow needs.** `workflow deps` reads the requirement off
the workflow instead of off your memory. Run across 48 archived canvases it
reported: 22 want rgthree, 5 want KJNodes' Get/SetNode, 2 want GGUF.

**Not lying about what happened.** Outputs are read from every bucket — images,
gifs, videos, audio — because code that reads only `images` finds nothing for
every video workflow. A graph with no output node is flagged before it wastes a
render.

**Keeping the patch loop honest.** `workflow set` patches the session graph;
`workflow diff` answers "what did I change since the render that worked",
`workflow unset` removes an override instead of guessing a replacement value,
and `workflow export` writes the patched graph to disk so it can be handed on.

## Session

State lives in `~/.config/cli-anything-comfyui/session.json`: the server URL, the
loaded graph, the last prompt id. One-shot mutations auto-save; `--dry-run`
suppresses it. Saves take an exclusive lock.

## Tests

```bash
CLI_ANYTHING_FORCE_INSTALLED=1 python -m pytest cli_anything/comfyui/tests/ -v -s
```

E2E tests drive a real ComfyUI and **fail** rather than skip when one is absent.
Two of them render an image and check the PNG magic bytes on the downloaded
file. See `tests/TEST.md`.

## Not built yet

- **Subgraph expansion.** A node whose `class_type` is a bare UUID is a subgraph
  instance; the harness names it and refuses rather than converting it wrongly.
- **`review`** — contact sheets and frame sampling of produced video.
- Windowed rendering (`windows`) is a CLI command, unit-tested against a fake
  client, but has no E2E test yet.
