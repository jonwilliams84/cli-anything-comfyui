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

import copy
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

#: How deep a subgraph may nest inside a subgraph before the harness stops
#: trusting the graph and reports the outermost instance unexpanded. Also the
#: cap on the fixpoint that rewires wires through expanded instances — a
#: canvas is acyclic, but a hostile one is not.
MAX_SUBGRAPH_DEPTH = 10


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


def subgraph_definitions(ui):
    """uuid -> the FULL subgraph definition, not just the name.

    Expansion needs the body: `inputNode`, `outputNode`, the inner `nodes` and
    `links`, the promoted `widgets`. `subgraph_defs` stays for callers that
    only want display names.
    """
    defs = (ui.get("definitions") or {}).get("subgraphs") or []
    return {str(d.get("id")): d for d in defs if isinstance(d, dict) and d.get("id")}


class _ExpansionBlocked(Exception):
    """This subgraph definition cannot be expanded; it stays reported."""


def _subgraph_block_reason(d):
    """Why this definition cannot be expanded, or None if it can."""
    if not (d.get("nodes") or []):
        return "the definition carries no nodes"
    return None


def _row_ends(row):
    """(link_id, origin_id, origin_slot, target_id, target_slot) from either link shape."""
    if isinstance(row, dict):
        return (
            row.get("id"),
            str(row.get("origin_id")),
            int(row.get("origin_slot") or 0),
            str(row.get("target_id")),
            int(row.get("target_slot") or 0),
        )
    if isinstance(row, (list, tuple)) and len(row) >= 5:
        return row[0], str(row[1]), int(row[2]), str(row[3]), int(row[4])
    return None


def _exposed_widget_value(instance, definition, name):
    """The VALUE an instance feeds an exposed WIDGET input.

    A subgraph widget is promoted to an input slot on the instance node; the
    value lives in the instance's `widgets_values`, positionally against the
    definition's `widgets` list. Some packs serialise by name (a dict), and a
    definition may carry its own default when the instance has nothing.
    """
    wname = name
    for i in instance.get("inputs") or []:
        w = i.get("widget") or {}
        if isinstance(w, dict) and w.get("name") and i.get("name") == name:
            wname = w["name"]
            break
    wv = instance.get("widgets_values")
    if isinstance(wv, dict):
        return wv.get(wname, wv.get(name))
    for i, w in enumerate(definition.get("widgets") or []):
        if isinstance(w, dict) and w.get("name") == wname:
            if isinstance(wv, list) and i < len(wv):
                return wv[i]
            return w.get("value")
    return None


def _expand_instance(
    nid, node, defs, object_info, links, nodes_by_id, keep_muted, depth, feeds, external
):
    """Expand ONE subgraph instance into the API nodes its body contains.

    The instance node's `type` is a bare UUID naming a definition in
    `definitions.subgraphs[]`. The body is a whole canvas in miniature — its own
    nodes, its own links — plus two boundary markers:

      - `inputNode`: its `inputs[]` are the slots the instance EXPOSES. Inner
        link rows whose ORIGIN is the inputNode carry an exposed input into the
        body; the value comes from the instance's own inputs (a link from the
        parent graph, or a promoted widget's value).
      - `outputNode`: inner link rows whose TARGET is the outputNode define
        what each exposed output IS; the parent graph's consumers of that
        instance output are rewired to the inner origin.

    The body is converted by calling `to_api` recursively — bypasses, reroutes,
    the control_after_generate off-by-one and nested subgraphs all just apply —
    then inner node ids are prefixed `"<instance>:<inner>"` so two instances of
    the same subgraph cannot collide, exposed inputs are patched in by NAME,
    and the exposed-output sources are returned so the caller can rewire the
    parent.

    Returns `(fragment, out_map, report_bits)`.
    """
    feeds = dict(feeds or {})
    external = set(external or ())
    d = defs[node["type"]]
    reason = _subgraph_block_reason(d)
    if reason:
        raise _ExpansionBlocked(reason)
    in_node = d.get("inputNode") or {}
    out_node = d.get("outputNode") or {}
    in_id = str(in_node.get("id") or "")
    out_id = str(out_node.get("id") or "")
    inner_nodes = [copy.deepcopy(n) for n in (d.get("nodes") or []) if isinstance(n, dict)]
    nested = (d.get("definitions") or {}).get("subgraphs") or []
    nested_defs = {str(g.get("id")) for g in nested if isinstance(g, dict) and g.get("id")}

    # Partition the body's links: boundary rows are consumed here, the rest go
    # to the inner conversion untouched.
    inner_links = []
    in_rows = {}  # inner link id -> exposed input slot index
    consumers = {}  # inner link id -> (inner target node id, input name)
    inst_feeds = []  # (inner target instance id, exposed input slot)
    out_rows = []  # (row, exposed output slot index)
    for row in d.get("links") or []:
        ends = _row_ends(row)
        if ends is None:
            continue
        lid, origin, oslot, target, tslot = ends
        if in_id and origin == in_id:
            in_rows[lid] = oslot
            tgt = next((n for n in inner_nodes if str(n.get("id")) == target), None)
            if tgt is not None and str(tgt.get("type")) in nested_defs:
                # The consumer is itself a subgraph instance: leave its exposed
                # input entry alone and inject the value after the recursive
                # conversion — stripping it would blind that instance's own
                # expansion to the wire.
                inst_feeds.append((target, oslot))
                continue
            if tgt is not None:
                kept, taken = [], None
                for i in tgt.get("inputs") or []:
                    if taken is None and i.get("link") == lid:
                        taken = i.get("name")
                    else:
                        kept.append(i)
                tgt["inputs"] = kept
                consumers[lid] = (target, taken)
            continue
        if out_id and target == out_id:
            out_rows.append((row, tslot))
            continue
        inner_links.append(copy.deepcopy(row))

    # What the instance was fed for one exposed input: a value injected by the
    # parent expansion (a nested instance's wire), else the parent link's
    # resolved source, else the promoted widget's value.
    def _feed_value(name, slot):
        if not name:
            return None
        hit = feeds.get((str(nid), name))
        if hit is not None:
            return hit
        entry = None
        for i in node.get("inputs") or []:
            if i.get("name") == name:
                entry = i
                break
        if entry is None:
            inst_inputs = node.get("inputs") or []
            if slot < len(inst_inputs):
                entry = inst_inputs[slot]
        value = None
        if entry is not None and entry.get("link") is not None:
            src = links.get(entry["link"])
            if src is not None:
                value = list(_resolve_through_bypass(src[0], src[1], nodes_by_id, links) or ())
        if value is None and entry is not None:
            value = _exposed_widget_value(node, d, entry.get("name") or "")
        return value

    # Values for boundary rows whose consumer is a NESTED instance: the nested
    # conversion cannot see this graph's links, so they travel in explicitly.
    inner_feeds = {}
    inner_external = set(external)
    exposed_inputs = in_node.get("inputs") or []
    for target, slot in inst_feeds:
        exposed = exposed_inputs[slot] if slot < len(exposed_inputs) else None
        name = exposed.get("name") if isinstance(exposed, dict) else None
        value = _feed_value(name, slot)
        if value:
            inner_feeds[(str(target), name)] = value
            if isinstance(value, list) and len(value) == 2:
                inner_external.add(str(value[0]))

    inner_ui = {"nodes": inner_nodes, "links": inner_links}
    if nested:
        inner_ui["definitions"] = {"subgraphs": nested}
    inner_api, inner_rep = to_api(
        inner_ui,
        object_info,
        keep_muted=keep_muted,
        strict=False,
        _depth=depth + 1,
        _feeds=inner_feeds or None,
        _external=inner_external or None,
    )

    # Prefix inner ids so two instances of one subgraph cannot collide.
    idmap = {k: f"{nid}:{k}" for k in inner_api}
    inner_api = {idmap[k]: v for k, v in inner_api.items()}
    for entry in inner_api.values():
        for name, val in (entry.get("inputs") or {}).items():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) in idmap:
                entry["inputs"][name] = [idmap[str(val[0])], val[1]]

    # Exposed inputs: what the instance was fed, by NAME, onto the inner node.
    for lid, slot in in_rows.items():
        exposed = exposed_inputs[slot] if slot < len(exposed_inputs) else None
        name = exposed.get("name") if isinstance(exposed, dict) else None
        tid, tname = consumers.get(lid, (None, None))
        if tid is None or tname is None or str(tid) not in idmap:
            continue
        value = _feed_value(name, slot)
        if value:
            inner_api[idmap[str(tid)]]["inputs"][tname] = value

    # Exposed outputs: the parent is rewired to the inner origin, keyed by the
    # instance's output slot — the caller fixes up every wire that pointed at
    # the instance, and chains of instance-feeding-instance settle in that pass.
    out_map = {}
    out_exposed = out_node.get("outputs") or []
    for row, exposed_slot in out_rows:
        ends = _row_ends(row)
        if ends is None:
            continue
        _lid, origin, oslot, _t, _ts = ends
        if str(origin) not in idmap:
            continue
        exposed = out_exposed[exposed_slot] if exposed_slot < len(out_exposed) else None
        oname = exposed.get("name") if isinstance(exposed, dict) else None
        inst_slot = None
        for i, o in enumerate(node.get("outputs") or []):
            if o.get("name") == oname:
                inst_slot = i
                break
        if inst_slot is None and exposed_slot < len(node.get("outputs") or []):
            inst_slot = exposed_slot
        if inst_slot is not None:
            out_map[(str(nid), inst_slot)] = (idmap[origin], oslot)

    def _prefixed(rows):
        for r in rows:
            if isinstance(r, dict) and r.get("node") is not None:
                r["node"] = f"{nid}:{r['node']}"
        return rows

    bits = {
        "warnings": _prefixed(inner_rep.get("warnings") or []),
        "dropped": _prefixed(inner_rep.get("dropped") or []),
        "missing": _prefixed(inner_rep.get("missing") or []),
        "unsupported": _prefixed(inner_rep.get("subgraphs") or []),
        "expanded": {
            "node": str(nid),
            "name": d.get("name") or "unnamed subgraph",
            "nodes_out": len(inner_api),
            "nested": bool(nested),
        },
    }
    return inner_api, out_map, bits


def subgraph_inventory(ui):
    """What each subgraph definition in a canvas contains, before conversion.

    A subgraph instance's `type` is a bare UUID, so nothing about it is
    readable without opening `definitions.subgraphs[]`. This names the body —
    inner nodes, exposed inputs/outputs, promoted widgets, which canvas nodes
    are instances of it — and whether the harness can expand it here.
    """
    defs = [
        d for d in ((ui.get("definitions") or {}).get("subgraphs") or []) if isinstance(d, dict)
    ]
    by_id = {str(d.get("id")): d for d in defs if d.get("id")}
    instances = {}
    for n in ui.get("nodes") or []:
        t = str(n.get("type") or "")
        if t in by_id:
            instances.setdefault(t, []).append(str(n.get("id")))
    inv = []
    for d in defs:
        did = str(d.get("id"))
        reason = _subgraph_block_reason(d)
        inv.append(
            {
                "id": did,
                "name": d.get("name") or "unnamed subgraph",
                "instances": instances.get(did) or [],
                "nodes": len(d.get("nodes") or []),
                "links": len(d.get("links") or []),
                "inputs": [
                    {"name": i.get("name"), "type": i.get("type")}
                    for i in ((d.get("inputNode") or {}).get("inputs") or [])
                    if isinstance(i, dict)
                ],
                "outputs": [
                    {"name": o.get("name"), "type": o.get("type")}
                    for o in ((d.get("outputNode") or {}).get("outputs") or [])
                    if isinstance(o, dict)
                ],
                "widgets": [
                    {"name": w.get("name"), "value": w.get("value")}
                    for w in (d.get("widgets") or [])
                    if isinstance(w, dict)
                ],
                "nested": bool((d.get("definitions") or {}).get("subgraphs")),
                "expandable": reason is None,
                "why_not": reason,
            }
        )
    return sorted(inv, key=lambda s: (s["name"], s["id"]))


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


def to_api(ui, object_info, keep_muted=False, strict=True, _depth=0, _feeds=None, _external=None):
    """UI workflow -> API graph the server will accept.

    Returns `(api_graph, report)`. The report names every node that was dropped
    and why, because a silent drop is indistinguishable from a graph that never
    had the node — and that is exactly how a render comes back missing its
    upscale pass.

    Subgraph instances (a node whose `type` is a bare UUID naming a definition
    in `definitions.subgraphs[]`) are EXPANDED in place: the body's nodes enter
    the graph with ids `"<instance>:<inner>"`, the instance's exposed inputs are
    fed by name, and consumers of the instance's outputs are rewired to the
    inner node that produces the value. A definition with no body (or nesting
    past :data:`MAX_SUBGRAPH_DEPTH`) stays unexpanded and is reported in
    `report["subgraphs"]` with the reason.

    `_feeds` / `_external` are expansion plumbing, not caller options: they
    carry a parent expansion's exposed-input values INTO a nested conversion,
    and the node ids those values may legally reference from outside this
    graph, so the dangling-wire cleanup cannot eat them.
    """
    feeds = dict(_feeds or {})
    external = set(_external or ())
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
            "subgraphs_expanded": [],
            "nodes_in": len(ui),
            "nodes_out": len(ui),
        }
    if fmt != "ui":
        raise WorkflowError("not a ComfyUI workflow: no `nodes` list and no class_type map")

    nodes = ui.get("nodes") or []
    nodes_by_id = {str(n.get("id")): n for n in nodes}
    links = link_map(ui)
    subgraphs = subgraph_defs(ui)
    defs = subgraph_definitions(ui)
    api, dropped, warnings, missing, unsupported = {}, [], [], [], []
    expanded, out_maps = [], {}

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
            if ctype in defs:
                if _depth >= MAX_SUBGRAPH_DEPTH:
                    unsupported.append(
                        {
                            "node": nid,
                            "class_type": ctype,
                            "name": subgraphs.get(ctype, ctype),
                            "kind": "subgraph",
                            "reason": f"nested more than {MAX_SUBGRAPH_DEPTH} deep",
                        }
                    )
                else:
                    try:
                        frag, om, bits = _expand_instance(
                            nid,
                            node,
                            defs,
                            object_info,
                            links,
                            nodes_by_id,
                            keep_muted,
                            _depth,
                            feeds,
                            external,
                        )
                    except _ExpansionBlocked as exc:
                        unsupported.append(
                            {
                                "node": nid,
                                "class_type": ctype,
                                "name": subgraphs.get(ctype, ctype),
                                "kind": "subgraph",
                                "reason": str(exc),
                            }
                        )
                    else:
                        api.update(frag)
                        out_maps.update(om)
                        warnings.extend(bits["warnings"])
                        dropped.extend(bits["dropped"])
                        missing.extend(bits["missing"])
                        unsupported.extend(bits["unsupported"])
                        expanded.append(bits["expanded"])
                        dropped.append(
                            {
                                "node": nid,
                                "class_type": ctype,
                                "why": f"subgraph instance; expanded in place into "
                                f"{len(frag)} node(s)",
                            }
                        )
                        continue
            elif ctype in subgraphs:
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

    # Rewire through expanded subgraphs. Every wire that pointed at an
    # instance's exposed output now points at the inner node that produces the
    # value; the fixpoint also settles chains of instance feeding instance.
    for _ in range(MAX_SUBGRAPH_DEPTH + 1):
        changed = False
        for entry in api.values():
            for name, val in (entry.get("inputs") or {}).items():
                if isinstance(val, list) and len(val) == 2:
                    hit = out_maps.get((str(val[0]), val[1]))
                    if hit and hit != (str(val[0]), val[1]):
                        entry["inputs"][name] = [hit[0], hit[1]]
                        changed = True
        if not changed:
            break

    # A link pointing at a node that was dropped is a dangling edge.
    live = set(api)
    for nid, entry in api.items():
        for name, val in list(entry["inputs"].items()):
            if (
                isinstance(val, list)
                and len(val) == 2
                and str(val[0]) not in live
                and str(val[0]) not in external
            ):
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
        "subgraphs_expanded": expanded,
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
            lines.append("These SUBGRAPHS could not be expanded:")
            for u in unsupported:
                reason = f" — {u['reason']}" if u.get("reason") else ""
                lines.append(f"  node {u['node']}: {u['name']}  ({u['class_type']}){reason}")
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


#: Sentinel for a widget input the schema gives no default for — including a
#: STRING widget with `forceInput`, which is declared a widget but is wired in
#: practice. `None` is a legal default value, so it cannot double as the marker.
_NO_DEFAULT = object()


def _widget_default(spec):
    """The value a fresh node carries for a widget input, or `_NO_DEFAULT`.

    A COMBO takes its first choice; a scalar takes its schema `default`. A
    widget with neither (`["INT"]`, or STRING with `forceInput`) has nothing to
    take — the caller must wire it or set it, and pretending otherwise is how a
    freshly built graph dies at the server.
    """
    if not isinstance(spec, (list, tuple)) or not spec:
        return _NO_DEFAULT  # a bare link type ("MODEL") — never a widget value
    if isinstance(spec[0], list):
        return spec[0][0] if spec[0] else _NO_DEFAULT  # COMBO: first choice
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return _NO_DEFAULT  # declared STRING, wired like any link
    default = opts.get("default")
    return default if default is not None else _NO_DEFAULT


def next_node_id(api):
    """The lowest free integer node id, as a string.

    Deterministic and collision-free against whatever the graph already holds —
    including the string ids subgraph expansion produces (`2:e`), which are not
    numeric and therefore never compete.
    """
    used = {int(n) for n in (api or {}) if str(n).isdigit()}
    n = 1
    while n in used:
        n += 1
    return str(n)


def add_node(api, class_type, object_info, node_id=None, title=None):
    """A new node, ready to wire: its widget inputs take the schema defaults.

    Building or repairing a graph was the one thing this harness still needed
    the canvas GUI for: `workflow set` can only change inputs on nodes that
    exist. This adds the node, filling every widget (COMBO or scalar) with the
    default the LIVE /object_info declares — the same source the converter
    trusts for names and order.

    Link-type inputs (and STRING widgets with `forceInput`) cannot be
    defaulted — they arrive over a wire — so they come back as `needs_wiring`
    and the node stays incomplete until `workflow wire` feeds each one. That is
    honest, not lazy: an unset required input is exactly what `workflow
    validate` reports, so the gap is checkable, never silent.
    """
    info = (object_info or {}).get(class_type)
    if not info:
        raise WorkflowError(f"node type {class_type!r} is not installed on this ComfyUI")
    nid = str(node_id) if node_id is not None else next_node_id(api)
    if nid in api:
        raise WorkflowError(f"node id {nid} is already taken by {api[nid].get('class_type')}")
    inputs = {}
    needs_wiring = []
    for name, spec in schema_inputs(object_info, class_type) or []:
        default = _widget_default(spec)
        if default is _NO_DEFAULT:
            needs_wiring.append(name)
        else:
            inputs[name] = default
    entry = {"class_type": class_type, "inputs": inputs}
    if title:
        entry["_meta"] = {"title": title}
    api[nid] = entry
    return {
        "node": nid,
        "class_type": class_type,
        "inputs": inputs,
        "needs_wiring": needs_wiring,
    }


def remove_node(api, node_id):
    """Delete a node AND every wire that pointed at it.

    Deleting a loader that feeds four consumers without clearing their links
    leaves a graph that queues and dies with `validate`'s 'wired to node N,
    which is not in the graph' — one render too late to be useful. The shape
    that counts as a wire is the same one `validate` uses: a two-element list
    whose first element names the node, so a widget that happens to be a
    two-element list is left alone unless it really names this node.
    """
    nid = str(node_id)
    if nid not in api:
        raise WorkflowError(f"no node {nid} in this graph (have: {', '.join(sorted(api)[:12])}…)")
    removed = api.pop(nid)
    cleared = []
    for other_id, entry in api.items():
        inputs = entry.get("inputs") or {}
        for name, val in list(inputs.items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == nid:
                del inputs[name]
                cleared.append(f"{other_id}.{name}")
    return {
        "removed": nid,
        "class_type": removed.get("class_type"),
        "wires_cleared": sorted(cleared),
    }


def wire_input(api, from_node, to_node, input_name, slot=0, object_info=None, force=False):
    """Wire one output of FROM into one input of TO.

    A link is just an input value of the form `[from_id, slot]`, which `workflow
    set` can also write by hand — but a bare set cannot check anything. This
    checks, against the server's own schema when both ends are known: that the
    slot exists on the source, that the target input exists at all (a typo is
    caught here instead of by the server), and that the type at each end
    matches. A MODEL wired into a CONDITIONING input queues clean and dies
    seconds into execution — exactly the class of mistake the pre-flight exists
    to catch. `--force` overrides a KNOWN mismatch, never an unknown type: when
    either node's schema is not installed the wire goes through unchecked,
    because guessing a type is how false alarms are made.
    """
    f, t = str(from_node), str(to_node)
    for nid in (f, t):
        if nid not in api:
            raise WorkflowError(
                f"no node {nid} in this graph (have: {', '.join(sorted(api)[:12])}…)"
            )
    if f == t:
        raise WorkflowError(f"cannot wire node {t} to itself")
    if slot < 0:
        raise WorkflowError(f"slot must be >= 0 (got {slot})")
    oi = object_info or {}
    src_type = None
    src_info = oi.get(api[f].get("class_type"))
    if src_info is not None and isinstance(src_info.get("output"), list):
        out_types = src_info["output"]
        if slot >= len(out_types):
            raise WorkflowError(
                f"node {f} ({api[f].get('class_type')}) has {len(out_types)} output(s); "
                f"slot {slot} does not exist"
            )
        src_type = out_types[slot]
    dst_type = None
    schema = schema_inputs(oi, api[t].get("class_type"))
    if schema is not None:
        spec = dict(schema).get(input_name)
        if spec is None:
            have = ", ".join(sorted(dict(schema)))
            raise WorkflowError(f"node {t} has no input {input_name!r} (have: {have})")
        if isinstance(spec[0], list):
            if not force:
                raise WorkflowError(
                    f"input {input_name!r} on node {t} is a widget (combo); "
                    "a wire cannot feed it — set its value with `workflow set` instead"
                )
        else:
            dst_type = spec[0]
    if not force and src_type and dst_type and src_type != dst_type:
        if "*" not in (src_type, dst_type):
            raise WorkflowError(
                f"type mismatch: node {f} output {slot} is {src_type}, "
                f"but node {t}.{input_name} wants {dst_type} (--force overrides)"
            )
    api[t].setdefault("inputs", {})[input_name] = [f, slot]
    return {
        "from": f,
        "slot": slot,
        "node": t,
        "input": input_name,
        "from_type": src_type,
        "input_type": dst_type,
    }


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
