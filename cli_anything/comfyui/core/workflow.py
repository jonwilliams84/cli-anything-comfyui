"""The two workflow formats, and the gap between them.

A saved canvas is the UI format: `nodes[]` with a POSITIONAL `widgets_values`
list, plus `links[]` joining node outputs to node inputs. `POST /prompt` accepts
none of that — it wants the API format, `{node_id: {class_type, inputs, _meta}}`,
where every widget value is under its REAL NAME.

Nothing on the server converts between them. The frontend does it in TypeScript,
which is why four separate scripts in `~/ai-video` each grew their own `wf2api`
and why this one is done against the LIVE `/object_info`: the names, their order,
and which of them even IS a widget all depend on the node-pack versions installed
on the machine that will run the graph.

THE OFF-BY-ONE THAT BREAKS EVERY HAND-ROLLED CONVERTER
    A widget whose schema carries `control_after_generate` gets a SECOND,
    invisible entry in `widgets_values` holding "randomize"/"fixed"/etc. A real
    KSampler on this box has SIX widget inputs and SEVEN values:

        [71177, "randomize", 4, 1.0, "euler", "simple", 1.0]
         seed   ^^^^^^^^^^^  steps cfg sampler scheduler denoise

    Consume the value and then consume the companion, or every widget after the
    seed silently shifts by one and the graph runs with `steps="randomize"`.
"""

from __future__ import annotations

import json
import os

from cli_anything.comfyui.utils.comfyui_backend import ComfyError

#: Types that are entered in the canvas rather than wired. Everything else
#: (MODEL, LATENT, IMAGE, CONDITIONING, VAE, CLIP, custom pack types...) can
#: only arrive over a link, so it never consumes a positional slot.
WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN"}

#: Present in the canvas, meaningless to the executor.
UI_ONLY_TYPES = {"Note", "MarkdownNote", "PrimitiveNode"}

#: Pure pass-through in the canvas — collapsed rather than emitted.
REROUTE_TYPES = {"Reroute", "RerouteNode", "Reroute (rgthree)"}

MODE_MUTED = 2
MODE_BYPASSED = 4


def subgraph_defs(ui):
    """uuid -> name, for the subgraphs this workflow carries.

    A subgraph instance appears in `nodes` with its `type` set to a bare UUID,
    and the body lives in `definitions.subgraphs[]`. Reporting that as "node
    type not installed" sends people hunting for a node pack that does not
    exist, so it is named separately.
    """
    defs = (ui.get("definitions") or {}).get("subgraphs") or []
    return {
        str(d.get("id")): d.get("name") or "unnamed subgraph"
        for d in defs
        if isinstance(d, dict) and d.get("id")
    }


class WorkflowError(ValueError):
    """A workflow that cannot be converted, with the node that caused it."""


def load(path):
    """Read a workflow file. Accepts either format and says which it is."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise WorkflowError(f"no such workflow: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def detect_format(data):
    """`ui`, `api`, or `unknown`.

    The UI format always has a `nodes` LIST. The API format is a bare mapping of
    id -> {class_type}. Anything else is neither, and saying so early beats a
    KeyError six frames down.
    """
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return "ui"
    if (
        isinstance(data, dict)
        and data
        and all(isinstance(v, dict) and "class_type" in v for v in data.values())
    ):
        return "api"
    return "unknown"


def _is_widget(spec):
    """Whether this input spec is entered in the canvas rather than wired."""
    t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    if isinstance(t, list):
        return True  # COMBO — a dropdown of choices
    return t in WIDGET_SCALARS


def _has_control_after_generate(spec):
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
    return bool(isinstance(opts, dict) and opts.get("control_after_generate"))


def schema_inputs(object_info, class_type):
    """(name, spec) for a node's inputs, required first, in declaration order.

    Order is the contract: `widgets_values` is positional against exactly this
    sequence. Python dicts preserve insertion order and the server serialises
    them in schema order, so this is stable — but it is the reason the live
    `/object_info` must be used rather than a snapshot from another machine.
    """
    info = object_info.get(class_type)
    if not info:
        return None
    spec = info.get("input") or {}
    out = []
    for group in ("required", "optional"):
        for name, s in (spec.get(group) or {}).items():
            out.append((name, s))
    return out


def link_map(ui):
    """link_id -> (origin_node_id, origin_slot).

    A link row is positional: [id, origin_node, origin_slot, target_node,
    target_slot, type].
    """
    out = {}
    for row in ui.get("links") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 5:
            out[row[0]] = (str(row[1]), int(row[2]))
        elif isinstance(row, dict) and "id" in row:  # newer schema variant
            out[row["id"]] = (str(row.get("origin_id")), int(row.get("origin_slot") or 0))
    return out


def _resolve_through_bypass(node_id, slot, nodes_by_id, links, seen=None):
    """Follow a link back through Reroute and BYPASSED nodes to a real source.

    A bypassed node (mode 4) stays in the canvas and passes its input straight
    through, so the API graph must point at whatever feeds it. Not following
    that is why "I muted a node and the whole graph broke" happens. Cycles are
    guarded because a canvas can contain one and this would otherwise recurse
    for ever.
    """
    seen = seen or set()
    key = (node_id, slot)
    if key in seen:
        raise WorkflowError(f"link cycle through node {node_id}")
    seen.add(key)
    node = nodes_by_id.get(str(node_id))
    if node is None:
        return None
    is_reroute = node.get("type") in REROUTE_TYPES
    is_bypassed = node.get("mode") == MODE_BYPASSED
    if not (is_reroute or is_bypassed):
        return (str(node_id), slot)
    # Take the first wired input — Reroute has exactly one; a bypassed node
    # passes the matching slot through, and slot 0 is the correct fallback.
    for inp in node.get("inputs") or []:
        lid = inp.get("link")
        if lid is not None and lid in links:
            src_id, src_slot = links[lid]
            return _resolve_through_bypass(src_id, src_slot, nodes_by_id, links, seen)
    return None


def to_api(ui, object_info, keep_muted=False, strict=True):
    """UI workflow -> API graph the server will accept.

    Returns `(api_graph, report)`. The report names every node that was dropped
    and why, because a silent drop is indistinguishable from a graph that never
    had the node — and that is exactly how a render comes back missing its
    upscale pass.
    """
    fmt = detect_format(ui)
    if fmt == "api":
        # SAME SHAPE as the conversion path. A caller reading
        # report["missing_node_types"] must not KeyError because the input
        # happened to be an API graph already — that turns a no-op into a crash.
        return dict(ui), {
            "already_api": True,
            "dropped": [],
            "warnings": [],
            "missing_node_types": [],
            "missing": [],
            "subgraphs": [],
            "nodes_in": len(ui),
            "nodes_out": len(ui),
        }
    if fmt != "ui":
        raise WorkflowError("not a ComfyUI workflow: no `nodes` list and no class_type map")

    nodes = ui.get("nodes") or []
    nodes_by_id = {str(n.get("id")): n for n in nodes}
    links = link_map(ui)
    subgraphs = subgraph_defs(ui)
    api, dropped, warnings, missing, unsupported = {}, [], [], [], []

    for node in nodes:
        nid = str(node.get("id"))
        ctype = node.get("type")
        mode = node.get("mode") or 0
        if ctype in UI_ONLY_TYPES:
            dropped.append({"node": nid, "class_type": ctype, "why": "canvas-only node"})
            continue
        if ctype in REROUTE_TYPES:
            dropped.append(
                {"node": nid, "class_type": ctype, "why": "reroute collapsed into its consumers"}
            )
            continue
        if mode == MODE_BYPASSED:
            dropped.append(
                {"node": nid, "class_type": ctype, "why": "bypassed (mode 4); inputs pass through"}
            )
            continue
        if mode == MODE_MUTED and not keep_muted:
            dropped.append(
                {
                    "node": nid,
                    "class_type": ctype,
                    "why": "muted (mode 2); pass --keep-muted to include",
                }
            )
            continue

        schema = schema_inputs(object_info, ctype)
        if schema is None:
            # Collected, not raised: one run must name EVERY missing type, or
            # installing packs becomes a guessing game one node at a time.
            if ctype in subgraphs:
                unsupported.append(
                    {"node": nid, "class_type": ctype, "name": subgraphs[ctype], "kind": "subgraph"}
                )
            else:
                missing.append({"node": nid, "class_type": ctype, "kind": "not installed"})
            continue

        # Inputs arriving over a link, by input NAME.
        wired = {}
        for inp in node.get("inputs") or []:
            lid = inp.get("link")
            name = inp.get("name")
            if lid is None or name is None:
                continue
            src = links.get(lid)
            if src is None:
                warnings.append(
                    {"node": nid, "input": name, "why": f"link {lid} has no origin — dropped"}
                )
                continue
            resolved = _resolve_through_bypass(src[0], src[1], nodes_by_id, links)
            if resolved is None:
                warnings.append(
                    {
                        "node": nid,
                        "input": name,
                        "why": "link resolves only to bypassed/reroute nodes with no source",
                    }
                )
                continue
            wired[name] = [resolved[0], resolved[1]]

        # Positional widgets, in schema order, honouring control_after_generate.
        values = list(node.get("widgets_values") or [])
        if isinstance(node.get("widgets_values"), dict):
            # Some packs serialise widgets by name. Nothing positional to do.
            wired.update({k: v for k, v in node["widgets_values"].items() if k not in wired})
            values = []
        cursor = 0
        inputs = {}
        for name, spec in schema:
            if not _is_widget(spec):
                continue
            if cursor < len(values):
                if name not in wired:
                    inputs[name] = values[cursor]
                cursor += 1
                if _has_control_after_generate(spec):
                    cursor += 1  # swallow the phantom companion slot
        if cursor < len(values):
            # Two innocent causes and one dangerous one: a FRONTEND-ONLY widget
            # the node's Python schema never declares (PixaromaPortraitLandscape
            # carries a preset dropdown like this), a note node's editor state,
            # or genuine version drift between the canvas and the installed pack.
            # The extra values are ignored — every NAMED input is still correct —
            # but it is reported because the third cause is worth knowing about.
            warnings.append(
                {
                    "node": nid,
                    "class_type": ctype,
                    "extra_values": len(values) - cursor,
                    "why": f"the canvas holds {len(values)} widget values but "
                    f"{ctype} declares {cursor}; the extras are ignored "
                    f"(frontend-only widget, or node-pack drift)",
                }
            )
        inputs.update(wired)
        entry = {"class_type": ctype, "inputs": inputs}
        title = node.get("title")
        if title:
            entry["_meta"] = {"title": title}
        api[nid] = entry

    # A link pointing at a node that was dropped is a dangling edge.
    live = set(api)
    for nid, entry in api.items():
        for name, val in list(entry["inputs"].items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) not in live:
                warnings.append(
                    {
                        "node": nid,
                        "input": name,
                        "why": f"wired to node {val[0]}, which is not in the graph — removed",
                    }
                )
                entry["inputs"].pop(name)

    report = {
        "already_api": False,
        "dropped": dropped,
        "warnings": warnings,
        "missing_node_types": sorted({m["class_type"] for m in missing}),
        "missing": missing,
        "subgraphs": unsupported,
        "nodes_in": len(nodes),
        "nodes_out": len(api),
    }
    if (missing or unsupported) and strict:
        lines = []
        if missing:
            lines.append("This ComfyUI does not have these node types installed:")
            for t in sorted({m["class_type"] for m in missing}):
                ids = ", ".join(m["node"] for m in missing if m["class_type"] == t)
                lines.append(f"  {t}  (node {ids})")
            lines.append("Install the packs that provide them — `workflow deps` lists them,")
            lines.append("and `nodes search <text>` shows what IS available.")
        if unsupported:
            lines.append("This workflow uses SUBGRAPHS, which are not expanded yet:")
            for u in unsupported:
                lines.append(f"  node {u['node']}: {u['name']}  ({u['class_type']})")
            lines.append("Open it in the canvas and use Convert to Nodes, or run it from the UI.")
        lines.append("Pass --no-strict to convert the rest anyway (the graph will be incomplete).")
        raise WorkflowError("\n".join(lines))
    return api, report


def validate(api, object_info):
    """Everything the server would reject, found before submitting.

    Cheaper than a round trip and it works with ComfyUI stopped, but it is NOT a
    substitute: only the server knows whether a named checkpoint is on disk.
    """
    problems = []
    for nid, entry in (api or {}).items():
        ctype = entry.get("class_type")
        if not ctype:
            problems.append({"node": nid, "problem": "no class_type"})
            continue
        schema = schema_inputs(object_info, ctype)
        if schema is None:
            problems.append(
                {
                    "node": nid,
                    "class_type": ctype,
                    "problem": "node type not installed on this ComfyUI",
                }
            )
            continue
        required = {
            n
            for n, s in (schema_inputs(object_info, ctype) or [])
            if n in ((object_info[ctype].get("input") or {}).get("required") or {})
        }
        given = set((entry.get("inputs") or {}))
        for miss in sorted(required - given):
            problems.append(
                {
                    "node": nid,
                    "class_type": ctype,
                    "input": miss,
                    "problem": "required input is missing",
                }
            )
        for name, val in (entry.get("inputs") or {}).items():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) not in api:
                problems.append(
                    {
                        "node": nid,
                        "class_type": ctype,
                        "input": name,
                        "problem": f"wired to node {val[0]}, which is not in the graph",
                    }
                )
    return {"ok": not problems, "problems": problems, "nodes": len(api or {})}


def outputs(api, object_info):
    """The nodes that actually produce a file.

    A graph with none of these runs, reports success and writes nothing — the
    quietest way to waste a render.
    """
    out = []
    for nid, entry in (api or {}).items():
        info = object_info.get(entry.get("class_type")) or {}
        if info.get("output_node"):
            out.append({"node": nid, "class_type": entry["class_type"]})
    return out


def set_input(api, node_id, name, value):
    """Patch one input. The whole point of workflow-as-code."""
    nid = str(node_id)
    if nid not in api:
        raise WorkflowError(f"no node {nid} in this graph (have: {', '.join(sorted(api)[:12])}…)")
    api[nid].setdefault("inputs", {})[name] = value
    return api


def unset_input(api, node_id, name):
    """Remove one input override, so the node's default or link applies again.

    `workflow set` can only add or overwrite; undoing a patch with another `set`
    bakes the wrong guess in as a value. Removing the input lets the node's own
    default or its wire apply again.

    IT IS NOT A TRUE INVERSE, and must not be described as one. If `set`
    OVERWROTE an existing value, `unset` does not put the old value back — it
    removes the key, and the graph no longer matches what it was before the
    `set`. Restoring would mean remembering prior values in the session, and
    "restore" is ambiguous anyway when the input was originally a LINK rather
    than a value. A test asserting set->unset round-trips to an identical graph
    is asserting a guarantee this function does not make; one shipped on
    2026-09-06 and failed against a real server for exactly that reason.
    """
    nid = str(node_id)
    if nid not in api:
        raise WorkflowError(f"no node {nid} in this graph (have: {', '.join(sorted(api)[:12])}…)")
    inputs = api[nid].get("inputs") or {}
    if name not in inputs:
        have = ", ".join(sorted(inputs)) or "(none)"
        raise WorkflowError(f"node {nid} has no input {name!r} (have: {have})")
    del inputs[name]
    return api


def _node_sort_key(nid):
    return int(nid) if str(nid).isdigit() else 10**9


def diff_graphs(before, after):
    """What changed between two API graphs, node by node, input by input.

    The everyday loop is convert -> set -> run. When a render stops coming back
    right, the question is "what did I change since the one that worked?" —
    and the answer is a diff, not a memory exercise. Links are just inputs
    whose value is `[id, slot]`, so a rewire shows up as an ordinary change.
    """
    before = before or {}
    after = after or {}
    b_nodes, a_nodes = set(before), set(after)
    added = sorted(a_nodes - b_nodes, key=_node_sort_key)
    removed = sorted(b_nodes - a_nodes, key=_node_sort_key)
    changed = []
    for nid in sorted(b_nodes & a_nodes, key=_node_sort_key):
        b_entry, a_entry = before[nid], after[nid]
        changes = []
        if b_entry.get("class_type") != a_entry.get("class_type"):
            changes.append(
                {
                    "input": "(class_type)",
                    "from": b_entry.get("class_type"),
                    "to": a_entry.get("class_type"),
                }
            )
        b_in = b_entry.get("inputs") or {}
        a_in = a_entry.get("inputs") or {}
        for name in sorted(set(b_in) | set(a_in)):
            if b_in.get(name) != a_in.get(name):
                changes.append(
                    {
                        "input": name,
                        "from": b_in.get(name, "(absent)"),
                        "to": a_in.get(name, "(absent)"),
                    }
                )
        if changes:
            changed.append(
                {"node": nid, "class_type": a_entry.get("class_type"), "changes": changes}
            )
    return {
        "same": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


#: A widget input whose name ends in this suffix names a model FILE on the
#: server's disk. Every core loader spells it that way (`ckpt_name`, `lora_name`,
#: `vae_name`, `clip_name`, `unet_name`, `control_net_name`,
#: `style_model_name`) and so do the packs that matter (`model_name` in
#: ImageUpscaleWithModel). Inputs wired over a link or of other names are not
#: filenames.
MODEL_INPUT_SUFFIX = "_name"


def model_refs(api, object_info):
    """The widget inputs that name a model FILE, with the node that wants it.

    `validate` checks the graph's SHAPE and `workflow deps` checks the node
    TYPES; both pass a graph whose CheckpointLoaderSimple names a checkpoint
    that is not on disk, and that graph queues clean and dies seconds into
    execution. This reads every string input ending in `_name` that the schema
    also declares as a widget — the only inputs whose value is a filename on
    the server.
    """
    refs = []
    for nid, entry in (api or {}).items():
        schema = dict(schema_inputs(object_info, entry.get("class_type")) or [])
        for name, val in (entry.get("inputs") or {}).items():
            if not isinstance(val, str) or not name.endswith(MODEL_INPUT_SUFFIX):
                continue  # links ([id, slot]) and plain scalars are not filenames
            spec = schema.get(name)
            if spec is None or not _is_widget(spec):
                continue  # undeclared, or a _name input wired over a link
            refs.append(
                {
                    "node": nid,
                    "class_type": entry.get("class_type"),
                    "input": name,
                    "model": val,
                }
            )
    return sorted(refs, key=lambda r: (_node_sort_key(r["node"]), r["input"]))


def index_models(client):
    """filename -> the folders that hold it, from every /models listing.

    A file counts as installed if ANY folder carries it. Guessing which folder
    a node pack expects (`loras`? `Lora`? `diffusion_models`?) is how a check
    that should pass reports a miss; the server's own folder listing is the
    only truth. A folder that refuses to be listed is skipped, not fatal — one
    unreadable folder must not stop the other folders from being checked.
    """
    index = {}
    folders = client.models()
    if isinstance(folders, dict):
        folders = list(folders)
    for folder in folders or []:
        try:
            items = client.models(folder)
        except ComfyError:
            continue
        if isinstance(items, dict):
            items = list(items)
        for item in items or []:
            if isinstance(item, str):
                index.setdefault(item, []).append(folder)
    return index


def check_models(client, api, object_info):
    """Which model files the graph names, and whether this server has them.

    With no model refs there is nothing to ask the server, so no /models call
    is made at all — a graph with no loaders must not turn a folder outage into
    a failure.
    """
    refs = model_refs(api, object_info)
    if not refs:
        return {
            "count": 0,
            "refs": [],
            "missing": [],
            "missing_count": 0,
            "ok": True,
            "folders_scanned": 0,
        }
    index = index_models(client)
    checked = []
    for r in refs:
        where = index.get(r["model"]) or []
        checked.append({**r, "installed_in": where})
    missing = [r for r in checked if not r["installed_in"]]
    return {
        "count": len(checked),
        "refs": checked,
        "missing": missing,
        "missing_count": len(missing),
        "ok": not missing,
        "folders_scanned": len({f for v in index.values() for f in v}),
    }


def find_nodes(api, class_type=None, title=None):
    """Locate nodes by type or title, so a caller need not know numeric ids."""
    hits = []
    for nid, entry in (api or {}).items():
        if class_type and entry.get("class_type") != class_type:
            continue
        if title and (entry.get("_meta") or {}).get("title") != title:
            continue
        hits.append(
            {
                "node": nid,
                "class_type": entry.get("class_type"),
                "title": (entry.get("_meta") or {}).get("title"),
            }
        )
    return sorted(hits, key=lambda h: int(h["node"]) if h["node"].isdigit() else 0)
