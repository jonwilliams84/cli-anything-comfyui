# ComfyUI — codebase analysis and CLI SOP

Source analysed: `Comfy-Org/ComfyUI` @ `15eb748` (the repo `comfyanonymous/ComfyUI`
and `comfy-org/comfyui` both redirect here). Live instance used throughout:
ComfyUI Desktop **0.34.5** on Windows, reachable from WSL at `127.0.0.1:8188`.

## Phase 1 — what the software actually is

### The backend engine is the running server, not a library

ComfyUI has no headless "convert this file" mode. Everything a human does in the
canvas becomes an HTTP call to a server that holds the models in VRAM. So the
harness is an HTTP client to the REAL ComfyUI, exactly as the browser is:

    POST /prompt              queue a graph, returns {prompt_id, number, node_errors}
    GET  /history/{id}        what a finished prompt produced
    POST /history             clear history, or delete named entries from it
    GET  /queue               running + pending
    POST /queue               clear, or delete by id
    POST /interrupt           stop the current execution
    POST /free                unload models / free VRAM  ← the OOM lever
    GET  /object_info         EVERY node type and its input schema (1805 on this box)
    GET  /models/{folder}     installed models per folder
    POST /upload/image        put a file in the input dir
    POST /upload/mask         put an inpainting MASK in, tied to its original image
    GET  /view                fetch an output by filename/subfolder/type
    GET  /system_stats        versions, argv, devices, VRAM
    GET  /internal/logs       the server's own log lines (newer builds)
    GET  /ws                  progress events while a prompt executes
    GET  /userdata?dir=...    list the server's user tree (saved canvases)
    GET|POST|DELETE /userdata/{path}   read / save / delete one file in it

`comfy-cli` exists but installs and manages ComfyUI; it does not drive a running
instance. There is no `melt`/`libreoffice --headless` equivalent — **the server is
the backend**, and it is a hard dependency.

### The data model is TWO workflow formats, and the gap between them is the work

A saved canvas (`user/default/workflows/*.json`, 48 of them in Jon's archive) is
the **UI format**:

    {id, revision, last_node_id, last_link_id, nodes[], links[], groups, extra, version}

with each node carrying `widgets_values` — a **POSITIONAL** list — plus `inputs`
holding link references.

`POST /prompt` accepts none of that. It wants the **API format**:

    {"<node_id>": {"class_type": "KSampler",
                   "inputs": {"seed": 1, "model": ["4", 0], ...},
                   "_meta": {"title": "..."}}}

Converting one to the other means mapping each node's positional
`widgets_values` onto NAMED inputs, and the names and their order come from
`/object_info` — which depends on the exact node-pack versions installed on that
machine. **Nothing on the server does this conversion**; the frontend does it in
TypeScript. That is why four separate scripts in `~/ai-video` had their own
`wf2api`, and why this harness does it against the LIVE `/object_info` rather
than a hardcoded table.

### There is no undo system

The queue is the mutation log. `/queue` and `/history` are the introspection
surface, and `/interrupt` is the only "stop".

## Phase 2 — CLI architecture

### Why this harness and not the three MCP servers

comfy-mcp, comfyui-mcp and comfy-pilot already cover queue/poll/fetch, node
install, model download and live-canvas editing. This harness deliberately does
NOT re-wrap those. It covers what they do not, chosen from evidence: the 116
driver scripts archived at `E:\ai-video-archive\workbench\scripts` are 18,930
lines in which **66 re-implement `queue_prompt`, 62 re-implement history polling,
45 shell out to ffmpeg and 27 hand-roll PIL**.

### Command groups

    server     status / features / embeddings / free / interrupt / logs
    nodes      list, search, schema, categories — the 1805 types and what each
               input is called, discoverable by category prefix
    workflow   convert (UI→API), validate, info, outputs, deps, get/set/unset a
               node input, export the patched graph, diff it against a file
    run        submit a graph and wait, with progress
    windows    the OOM-guarded window loop: several graphs, VRAM freed between,
               a failed window does not abort the rest
    queue      list / wait <id> / cancel / clear
    history    list / outputs / clear (the canvas's Clear-history button, as an
               API call: POST /history with {"clear": true} or {"delete": [ids]})
    assets     upload an input (--kind input/temp/output, --no-overwrite makes
               the server answer 409 instead of silently replacing),
               upload an inpainting mask, download an output
    userdata   list / get / put / move / copy / delete the server's user tree
               (move is the server's POST /userdata/{path}/move/{to}; copy is a
               GET and a PUT, because the API has no copy route)
    models     what is installed, per folder
    review     sample frames from produced media into a contact sheet (NOT BUILT
               yet — needs an imaging dependency the harness deliberately lacks)
    traps      the recorded failure catalogue for this software

### State model

A session JSON holds: server URL, the loaded workflow (API format) and its
source path, the last `prompt_id`, and the output directory. One-shot mutations
auto-save; `--dry-run` suppresses the save. Saves take an exclusive lock.

### Rendering

There is no rendering gap to bridge here: the server IS the renderer. The
harness's job is to hand it a valid API graph and verify what came back — file
exists, non-zero, correct magic bytes, expected frame count/duration.
