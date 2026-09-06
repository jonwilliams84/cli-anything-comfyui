"""cli-anything-comfyui — drive a running ComfyUI from the shell.

Every command supports `--json`. With no subcommand this enters the REPL.
"""

from __future__ import annotations

import json as _json
import os
import sys

import click

from cli_anything.comfyui.core import run as run_core
from cli_anything.comfyui.core import session as session_core
from cli_anything.comfyui.core import workflow as wf
from cli_anything.comfyui.utils.comfyui_backend import ComfyError, ComfyUI, DEFAULT_URL, outputs_of

__version__ = "0.1.0"
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------- output


def emit(ctx, payload, human=None):
    """One printer. `--json` anywhere in the invocation wins."""
    if ctx.obj.get("json"):
        click.echo(_json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if human is not None:
        click.echo(human if isinstance(human, str) else "\n".join(human))
        return
    click.echo(_json.dumps(payload, indent=2, sort_keys=True, default=str))


def die(msg, code=1):
    """Fail loudly. Agents self-correct from the message, not the exit code."""
    click.echo(f"error: {msg}", err=True)
    sys.exit(code)


json_option = click.option(
    "--json", "json_", is_flag=True, default=False, help="Machine-readable output."
)


def _merge_json(ctx, json_):
    if json_:
        ctx.obj["json"] = True


def client(ctx):
    return ComfyUI(url=ctx.obj.get("url") or DEFAULT_URL, timeout=ctx.obj.get("timeout") or 60)


def _object_info(ctx):
    """Cached per invocation — /object_info is ~1800 entries and several MB."""
    if "object_info" not in ctx.obj:
        ctx.obj["object_info"] = client(ctx).object_info()
    return ctx.obj["object_info"]


def _state(ctx):
    if "state" not in ctx.obj:
        ctx.obj["state"] = session_core.load(ctx.obj.get("session"), url=ctx.obj.get("url"))
    return ctx.obj["state"]


def _autosave(ctx, state):
    """Save after a mutation unless --dry-run. See HARNESS auto-save rule."""
    return session_core.save(state, dry_run=ctx.obj.get("dry_run", False))


# ------------------------------------------------------------------ root group


@click.group(invoke_without_command=True)
@click.option(
    "--url", default=None, help=f"ComfyUI base URL (default {DEFAULT_URL}, or $COMFYUI_URL)."
)
@click.option(
    "--session",
    default=None,
    help="Session file (default ~/.config/cli-anything-comfyui/session.json).",
)
@click.option("--timeout", default=60, type=int, show_default=True, help="HTTP timeout, seconds.")
@click.option("--dry-run", is_flag=True, default=False, help="Do not save session changes.")
@click.option("--json", "json_", is_flag=True, default=False, help="Machine-readable output.")
@click.version_option(__version__, prog_name="cli-anything-comfyui")
@click.pass_context
def cli(ctx, url, session, timeout, dry_run, json_):
    """Drive a running ComfyUI: workflows, nodes, queue, renders, outputs.

    This harness talks to a REAL ComfyUI over HTTP. It renders nothing itself.
    Start one first (the Desktop app, or `python main.py --listen 0.0.0.0`).
    """
    ctx.ensure_object(dict)
    ctx.obj.update(
        {"url": url, "session": session, "timeout": timeout, "dry_run": dry_run, "json": json_}
    )
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


# ---------------------------------------------------------------------- server


@cli.group()
def server():
    """The ComfyUI instance itself: is it up, what is it, free its VRAM."""


@server.command("status")
@json_option
@click.pass_context
def server_status(ctx, json_):
    """Version, OS, devices and free VRAM."""
    _merge_json(ctx, json_)
    c = client(ctx)
    try:
        s = c.system_stats()
    except ComfyError as exc:
        die(str(exc))
    sysinfo = s.get("system") or {}
    devs = []
    for d in s.get("devices") or []:
        devs.append(
            {
                "name": d.get("name"),
                "type": d.get("type"),
                "vram_total_gb": round((d.get("vram_total") or 0) / 2**30, 1),
                "vram_free_gb": round((d.get("vram_free") or 0) / 2**30, 1),
            }
        )
    payload = {
        "url": c.url,
        "up": True,
        "comfyui_version": sysinfo.get("comfyui_version"),
        "os": sysinfo.get("os"),
        "python": sysinfo.get("python_version", "")[:12],
        "ram_free_gb": round((sysinfo.get("ram_free") or 0) / 2**30, 1),
        "devices": devs,
        "argv": sysinfo.get("argv") or [],
    }
    lines = [f"ComfyUI {payload['comfyui_version']} on {payload['os']}  ({c.url})"]
    for d in devs:
        lines.append(f"  {d['name']}  {d['vram_free_gb']}/{d['vram_total_gb']} GB VRAM free")
    lines.append(f"  host RAM free: {payload['ram_free_gb']} GB")
    emit(ctx, payload, lines)


@server.command("free")
@click.option(
    "--keep-models", is_flag=True, default=False, help="Free cache but keep models resident."
)
@json_option
@click.pass_context
def server_free(ctx, keep_models, json_):
    """Unload models and free VRAM.

    The lever for long renders. Models stay resident between prompts, and a
    windowed render that never calls this is how a card fills up mid-film.
    """
    _merge_json(ctx, json_)
    try:
        res = client(ctx).free(unload_models=not keep_models, free_memory=True)
    except ComfyError as exc:
        die(str(exc))
    emit(ctx, res, "freed VRAM" + ("" if keep_models else " and unloaded models"))


@server.command("interrupt")
@json_option
@click.pass_context
def server_interrupt(ctx, json_):
    """Stop whatever is executing now."""
    _merge_json(ctx, json_)
    try:
        emit(ctx, client(ctx).interrupt(), "interrupted the running prompt")
    except ComfyError as exc:
        die(str(exc))


@server.command("features")
@json_option
@click.pass_context
def server_features(ctx, json_):
    """The server's feature flags (/features)."""
    _merge_json(ctx, json_)
    try:
        emit(ctx, client(ctx).features())
    except ComfyError as exc:
        die(str(exc))


@server.command("logs")
@click.option("--limit", default=100, show_default=True, type=int)
@json_option
@click.pass_context
def server_logs(ctx, limit, json_):
    """The server's own recent log lines.

    A rejected prompt says WHAT was refused; the log says what happened around
    it — the OOM warning, the missing pack, the failed checkpoint load.
    """
    _merge_json(ctx, json_)
    try:
        res = client(ctx).logs(limit=limit)
    except ComfyError as exc:
        die(str(exc))
    entries = res.get("entries") or [] if isinstance(res, dict) else list(res or [])
    lines = [f"{len(entries)} log line(s)"]
    for e in entries[-limit:]:
        msg = e.get("m") if isinstance(e, dict) else e
        lines.append(f"  {msg}")
    emit(ctx, {"count": len(entries), "entries": entries[-limit:]}, lines)


@server.command("embeddings")
@json_option
@click.pass_context
def server_embeddings(ctx, json_):
    """The embeddings this server has."""
    _merge_json(ctx, json_)
    try:
        res = client(ctx).embeddings()
    except ComfyError as exc:
        die(str(exc))
    items = res if isinstance(res, list) else list(res or {})
    emit(
        ctx,
        {"count": len(items), "items": items},
        [f"{len(items)} embedding(s)"] + [f"  {i}" for i in items[:40]],
    )


# ----------------------------------------------------------------------- nodes


@cli.group()
def nodes():
    """The node types this ComfyUI has, and what their inputs are called."""


@nodes.command("list")
@click.option("--category", default=None, help="Filter by category prefix.")
@click.option("--limit", default=40, show_default=True, type=int)
@json_option
@click.pass_context
def nodes_list(ctx, category, limit, json_):
    """List installed node types."""
    _merge_json(ctx, json_)
    try:
        oi = _object_info(ctx)
    except ComfyError as exc:
        die(str(exc))
    rows = []
    for name, info in oi.items():
        cat = info.get("category") or ""
        if category and not cat.lower().startswith(category.lower()):
            continue
        rows.append({"name": name, "category": cat, "output_node": bool(info.get("output_node"))})
    rows.sort(key=lambda r: (r["category"], r["name"]))
    payload = {"count": len(rows), "shown": min(limit, len(rows)), "nodes": rows[:limit]}
    emit(
        ctx,
        payload,
        [f"{len(rows)} node types (showing {min(limit, len(rows))})"]
        + [f"  {r['name']:44s} {r['category']}" for r in rows[:limit]],
    )


@nodes.command("search")
@click.argument("text")
@click.option("--limit", default=25, show_default=True, type=int)
@json_option
@click.pass_context
def nodes_search(ctx, text, limit, json_):
    """Find a node type by name or category."""
    _merge_json(ctx, json_)
    try:
        oi = _object_info(ctx)
    except ComfyError as exc:
        die(str(exc))
    t = text.lower()
    hits = [
        {
            "name": n,
            "category": (i.get("category") or ""),
            "output_node": bool(i.get("output_node")),
        }
        for n, i in oi.items()
        if t in n.lower() or t in (i.get("category") or "").lower()
    ]
    hits.sort(key=lambda h: (len(h["name"]), h["name"]))
    emit(
        ctx,
        {"query": text, "count": len(hits), "matches": hits[:limit]},
        [f"{len(hits)} match(es) for {text!r}"]
        + [f"  {h['name']:44s} {h['category']}" for h in hits[:limit]],
    )


@nodes.command("schema")
@click.argument("class_type")
@json_option
@click.pass_context
def nodes_schema(ctx, class_type, json_):
    """A node's inputs: names, types, defaults, and which are WIDGETS.

    The widget/link distinction is what `workflow convert` depends on, so it is
    shown explicitly rather than left to be inferred.
    """
    _merge_json(ctx, json_)
    try:
        oi = _object_info(ctx)
    except ComfyError as exc:
        die(str(exc))
    if class_type not in oi:
        die(f"no node type {class_type!r} on this ComfyUI. Try: nodes search {class_type}")
    info = oi[class_type]
    rows = []
    for name, spec in wf.schema_inputs(oi, class_type):
        t = spec[0] if isinstance(spec, list) and spec else spec
        opts = spec[1] if isinstance(spec, list) and len(spec) > 1 else {}
        rows.append(
            {
                "name": name,
                "type": "COMBO" if isinstance(t, list) else t,
                "widget": wf._is_widget(spec),
                "control_after_generate": wf._has_control_after_generate(spec),
                "default": (opts or {}).get("default") if isinstance(opts, dict) else None,
                "choices": t[:8] if isinstance(t, list) else None,
            }
        )
    payload = {
        "class_type": class_type,
        "category": info.get("category"),
        "output_node": bool(info.get("output_node")),
        "outputs": info.get("output_name") or info.get("output") or [],
        "inputs": rows,
    }
    lines = [
        f"{class_type}  ({info.get('category')})",
        f"  outputs: {', '.join(str(o) for o in payload['outputs'])}",
    ]
    for r in rows:
        kind = "widget" if r["widget"] else "link"
        extra = " +control_after_generate" if r["control_after_generate"] else ""
        lines.append(f"  {r['name']:22s} {str(r['type'])[:22]:22s} {kind}{extra}")
    emit(ctx, payload, lines)


@nodes.command("categories")
@json_option
@click.pass_context
def nodes_categories(ctx, json_):
    """The category prefixes installed here, with node counts.

    `nodes list --category` filters, but filtering needs something to filter BY.
    This is the discovery half: what this server actually has, biggest first.
    """
    _merge_json(ctx, json_)
    try:
        oi = _object_info(ctx)
    except ComfyError as exc:
        die(str(exc))
    counts = {}
    for info in oi.values():
        cat = info.get("category") or "(none)"
        counts[cat] = counts.get(cat, 0) + 1
    rows = [
        {"category": cat, "nodes": n}
        for cat, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    emit(
        ctx,
        {"count": len(rows), "node_types": len(oi), "categories": rows},
        [f"{len(rows)} categor(y/ies) across {len(oi)} node types"]
        + [f"  {r['nodes']:>5d}  {r['category']}" for r in rows[:40]],
    )


# -------------------------------------------------------------------- workflow


@cli.group()
def workflow():
    """Workflows as code: convert a canvas, inspect it, patch it, check it."""


@workflow.command("convert")
@click.argument("path", type=click.Path())
@click.option("-o", "--out", default=None, help="Write the API graph here.")
@click.option("--load/--no-load", "do_load", default=True, help="Also load it into the session.")
@click.option("--strict/--no-strict", default=True, help="Fail on missing node types.")
@click.option("--keep-muted", is_flag=True, default=False, help="Include muted (mode 2) nodes.")
@json_option
@click.pass_context
def workflow_convert(ctx, path, out, do_load, strict, keep_muted, json_):
    """Canvas (UI) workflow -> API graph the server accepts.

    Converted against this machine's live /object_info, because widget names and
    their ORDER come from the installed node packs.
    """
    _merge_json(ctx, json_)
    try:
        ui = wf.load(path)
        oi = _object_info(ctx)
        api, report = wf.to_api(ui, oi, keep_muted=keep_muted, strict=strict)
    except (wf.WorkflowError, ComfyError) as exc:
        die(str(exc))
    if out:
        out = os.path.abspath(os.path.expanduser(out))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            _json.dump(api, fh, indent=2, sort_keys=True)
    saved = None
    if do_load:
        st = _state(ctx)
        st["workflow"] = api
        st["workflow_source"] = os.path.abspath(os.path.expanduser(path))
        saved = _autosave(ctx, st)
    payload = {"source": path, "out": out, "nodes": len(api), **report, "session": saved}
    lines = [f"converted {report.get('nodes_in', '?')} canvas nodes -> {len(api)} API nodes"]
    for d in report.get("dropped", [])[:8]:
        lines.append(f"  dropped node {d['node']} ({d['class_type']}): {d['why']}")
    for w in report.get("warnings", [])[:8]:
        lines.append(f"  warning: node {w.get('node')} {w.get('input', '')} — {w['why']}")
    if out:
        lines.append(f"  wrote {out}")
    emit(ctx, payload, lines)


@workflow.command("deps")
@click.argument("path", type=click.Path())
@json_option
@click.pass_context
def workflow_deps(ctx, path, json_):
    """What this workflow needs that this ComfyUI does not have.

    Run it before a migration: it answers "which node packs must I reinstall"
    from the workflows themselves rather than from memory.
    """
    _merge_json(ctx, json_)
    try:
        ui = wf.load(path)
        api, report = wf.to_api(ui, _object_info(ctx), strict=False)
    except (wf.WorkflowError, ComfyError) as exc:
        die(str(exc))
    payload = {
        "source": path,
        "missing_node_types": report["missing_node_types"],
        "subgraphs": report["subgraphs"],
        "nodes": len(api),
        "satisfied": not report["missing_node_types"],
    }
    if payload["satisfied"] and not payload["subgraphs"]:
        lines = ["every node type in this workflow is installed"]
    else:
        lines = []
        if payload["missing_node_types"]:
            lines.append("missing node types:")
            lines += [f"  {t}" for t in payload["missing_node_types"]]
        for s in payload["subgraphs"]:
            lines.append(f"  subgraph (not expanded): {s['name']}")
    emit(ctx, payload, lines)


@workflow.command("info")
@click.option(
    "--path", "path", default=None, type=click.Path(), help="Inspect a file instead of the session."
)
@json_option
@click.pass_context
def workflow_info(ctx, path, json_):
    """What the loaded graph is: node counts, and which nodes produce files."""
    _merge_json(ctx, json_)
    api = _graph(ctx, path)
    try:
        oi = _object_info(ctx)
    except ComfyError as exc:
        die(str(exc))
    outs = wf.outputs(api, oi)
    by_type = {}
    for e in api.values():
        by_type[e.get("class_type")] = by_type.get(e.get("class_type"), 0) + 1
    payload = {
        "nodes": len(api),
        "output_nodes": outs,
        "by_class_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
    }
    lines = [f"{len(api)} nodes, {len(outs)} output node(s)"]
    if not outs:
        lines.append("  WARNING: no output node — this graph will run and write nothing")
    for o in outs:
        lines.append(f"  output: node {o['node']} ({o['class_type']})")
    emit(ctx, payload, lines)


@workflow.command("find")
@click.option("--type", "class_type", default=None, help="Match class_type.")
@click.option("--title", default=None, help="Match the node's title.")
@json_option
@click.pass_context
def workflow_find(ctx, class_type, title, json_):
    """Locate nodes by type or title, so you need not know numeric ids."""
    _merge_json(ctx, json_)
    api = _graph(ctx, None)
    hits = wf.find_nodes(api, class_type=class_type, title=title)
    emit(
        ctx,
        {"count": len(hits), "matches": hits},
        [f"{len(hits)} match(es)"]
        + [
            f"  node {h['node']:>5s}  {h['class_type']}"
            + (f"  “{h['title']}”" if h["title"] else "")
            for h in hits
        ],
    )


@workflow.command("outputs")
@click.option(
    "--path", "path", default=None, type=click.Path(), help="Inspect a file instead of the session."
)
@json_option
@click.pass_context
def workflow_outputs(ctx, path, json_):
    """Which nodes in the graph actually write files.

    A graph with none of these runs, reports success and saves nothing — the
    quietest way to waste a render. Composes with `run --download`: every node
    listed here is one whose files that download will carry.
    """
    _merge_json(ctx, json_)
    api = _graph(ctx, path)
    try:
        outs = wf.outputs(api, _object_info(ctx))
    except ComfyError as exc:
        die(str(exc))
    lines = [f"{len(api)} nodes, {len(outs)} output node(s)"]
    if not outs:
        lines.append("  WARNING: no output node — this graph will run and write nothing")
    for o in outs:
        lines.append(f"  output: node {o['node']} ({o['class_type']})")
    emit(ctx, {"nodes": len(api), "output_count": len(outs), "outputs": outs}, lines)


@workflow.command("set")
@click.argument("node_id")
@click.argument("name")
@click.argument("value")
@click.option("--raw", is_flag=True, default=False, help="Keep the value as a string.")
@json_option
@click.pass_context
def workflow_set(ctx, node_id, name, value, raw, json_):
    """Patch one input on the loaded graph. Auto-saves unless --dry-run."""
    _merge_json(ctx, json_)
    st = _state(ctx)
    api = st.get("workflow")
    if not api:
        die("no workflow loaded. Run: workflow convert <canvas.json>")
    val = value
    if not raw:
        try:
            val = _json.loads(value)  # numbers, booleans, and [id, slot] links
        except _json.JSONDecodeError:
            val = value
    try:
        wf.set_input(api, node_id, name, val)
    except wf.WorkflowError as exc:
        die(str(exc))
    saved = _autosave(ctx, st)
    emit(
        ctx,
        {"node": str(node_id), "input": name, "value": val, "session": saved},
        f"node {node_id}.{name} = {val!r}"
        + ("  (dry run — not saved)" if saved.get("dry_run") else ""),
    )


@workflow.command("validate")
@click.option("--path", "path", default=None, type=click.Path())
@json_option
@click.pass_context
def workflow_validate(ctx, path, json_):
    """Everything the server would reject, found before submitting."""
    _merge_json(ctx, json_)
    api = _graph(ctx, path)
    try:
        res = wf.validate(api, _object_info(ctx))
    except ComfyError as exc:
        die(str(exc))
    lines = [
        f"{res['nodes']} nodes: " + ("valid" if res["ok"] else f"{len(res['problems'])} problem(s)")
    ]
    for p in res["problems"][:15]:
        lines.append(f"  node {p['node']} {p.get('input', '')}: {p['problem']}")
    emit(ctx, res, lines)
    if not res["ok"]:
        sys.exit(1)


@workflow.command("unset")
@click.argument("node_id")
@click.argument("name")
@json_option
@click.pass_context
def workflow_unset(ctx, node_id, name, json_):
    """Remove one input override from the loaded graph. Auto-saves unless --dry-run.

    The inverse of `workflow set`: the node falls back to its schema default or
    its link, rather than to whatever wrong guess overwrote it.
    """
    _merge_json(ctx, json_)
    st = _state(ctx)
    api = st.get("workflow")
    if not api:
        die("no workflow loaded. Run: workflow convert <canvas.json>")
    try:
        wf.unset_input(api, node_id, name)
    except wf.WorkflowError as exc:
        die(str(exc))
    saved = _autosave(ctx, st)
    emit(
        ctx,
        {"node": str(node_id), "input": name, "session": saved},
        f"node {node_id}.{name} removed"
        + ("  (dry run — not saved)" if saved.get("dry_run") else ""),
    )


@workflow.command("export")
@click.option("-o", "--out", required=True, type=click.Path(), help="Write the API graph here.")
@json_option
@click.pass_context
def workflow_export(ctx, out, json_):
    """Write the CURRENT session graph (patches included) to a file.

    `workflow convert -o` writes the freshly converted canvas; this writes what
    the graph has BECOME — every `workflow set` applied — so a patched graph can
    be handed to someone else, or diffed against its source.
    """
    _merge_json(ctx, json_)
    api = _state(ctx).get("workflow")
    if not api:
        die("no workflow loaded. Run: workflow convert <canvas.json>")
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        _json.dump(api, fh, indent=2, sort_keys=True)
    emit(ctx, {"out": out, "nodes": len(api)}, f"wrote {len(api)} nodes to {out}")


@workflow.command("diff")
@click.argument("path_a", required=False, type=click.Path())
@click.argument("path_b", required=False, type=click.Path())
@json_option
@click.pass_context
def workflow_diff(ctx, path_a, path_b, json_):
    """What changed between two graphs: `diff OTHER.json` or `diff A.json B.json`.

    With one path, the file is the BASELINE and the loaded session graph is the
    current state — "what did I change since the render that worked". With two,
    A is the baseline and B the current.
    """
    _merge_json(ctx, json_)
    if not path_a and not path_b:
        die("give a file to compare against the session graph, or two files.")
    if path_b:
        before = _graph(ctx, path_a)
        after = _graph(ctx, path_b)
        before_label, after_label = path_a, path_b
    else:
        after = _state(ctx).get("workflow")
        if not after:
            die("no workflow loaded. Run: workflow convert <canvas.json>")
        before = _graph(ctx, path_a)
        before_label, after_label = path_a, "(session)"
    d = wf.diff_graphs(before, after)
    payload = {"baseline": before_label, **d}
    lines = [
        f"{before_label} -> {after_label}: "
        + (
            "identical"
            if d["same"]
            else f"{len(d['added'])} added, "
            f"{len(d['removed'])} removed, {len(d['changed'])} changed"
        )
    ]
    for nid in d["added"]:
        lines.append(f"  + node {nid}")
    for nid in d["removed"]:
        lines.append(f"  - node {nid}")
    for c in d["changed"]:
        for ch in c["changes"]:
            lines.append(f"  ~ node {c['node']} {ch['input']}: {ch['from']!r} -> {ch['to']!r}")
    emit(ctx, payload, lines)


def _graph(ctx, path):
    """The graph to act on: an explicit file, else the session's."""
    if path:
        try:
            data = wf.load(path)
            if wf.detect_format(data) == "ui":
                api, _ = wf.to_api(data, _object_info(ctx), strict=False)
                return api
            return data
        except (wf.WorkflowError, ComfyError) as exc:
            die(str(exc))
    api = _state(ctx).get("workflow")
    if not api:
        die("no workflow loaded. Run: workflow convert <canvas.json>  (or pass --path)")
    return api


# ------------------------------------------------------------------------- run


@cli.command("run")
@click.option(
    "--path",
    "path",
    default=None,
    type=click.Path(),
    help="Run this file instead of the session graph.",
)
@click.option("--timeout", default=1800, show_default=True, type=int)
@click.option("--front", is_flag=True, default=False, help="Jump the queue.")
@click.option("--download", "dest", default=None, help="Download the outputs into this directory.")
@json_option
@click.pass_context
def run_cmd(ctx, path, timeout, front, dest, json_):
    """Queue the graph, wait for it, and report what it produced."""
    _merge_json(ctx, json_)
    api = _graph(ctx, path)
    c = client(ctx)
    try:
        res = run_core.submit_and_wait(c, api, timeout=timeout, front=front)
    except ComfyError as exc:
        die(str(exc))
    st = _state(ctx)
    st["last_prompt_id"] = res["prompt_id"]
    _autosave(ctx, st)
    if dest and res["outputs"]:
        res["download"] = run_core.fetch(c, res["outputs"], dest)
    lines = [
        f"prompt {res['prompt_id']} finished in {res['elapsed_s']}s — "
        f"{res['output_count']} output file(s)"
    ]
    for f in res["outputs"][:10]:
        lines.append(f"  [{f['bucket']}] {f['filename']}")
    if res["output_count"] == 0:
        lines.append("  WARNING: nothing was produced — does the graph have an output node?")
    if res.get("download"):
        lines.append(
            f"  downloaded {res['download']['downloaded']} file(s) to {res['download']['dir']}"
        )
    emit(ctx, res, lines)


@cli.command("windows")
@click.argument("paths", nargs=-1, required=True, type=click.Path())
@click.option("--timeout", default=1800, show_default=True, type=int)
@click.option("--keep-vram", is_flag=True, default=False, help="Do NOT free VRAM between windows.")
@click.option(
    "--download", "dest", default=None, help="Download every produced file into this directory."
)
@json_option
@click.pass_context
def windows_cmd(ctx, paths, timeout, keep_vram, dest, json_):
    """Run several graphs as ONE windowed render, freeing VRAM between them.

    The loop every long film on this estate is made with. A window that fails
    does not abort the rest — losing window 7 of 12 should still hand back the
    other eleven. Each PATH is a workflow file (canvas or API format).
    """
    _merge_json(ctx, json_)
    graphs = [_graph(ctx, p) for p in paths]
    c = client(ctx)
    try:
        res = run_core.run_windows(c, graphs, timeout=timeout, free_between=not keep_vram)
    except ComfyError as exc:
        die(str(exc))
    if dest and res["outputs"]:
        res["download"] = run_core.fetch(c, res["outputs"], dest)
    lines = [f"{res['succeeded']}/{res['windows']} window(s) finished"]
    for r in res["results"]:
        if r.get("ok"):
            lines.append(f"  window {r['window']}: {r['output_count']} file(s)")
        else:
            lines.append(f"  window {r['window']}: FAILED — {r['error']}")
    if res.get("download"):
        lines.append(
            f"  downloaded {res['download']['downloaded']} file(s) to {res['download']['dir']}"
        )
    emit(ctx, res, lines)
    if res["failed"]:
        sys.exit(1)


# ----------------------------------------------------------------- queue/history


@cli.group()
def queue():
    """The execution queue."""


@queue.command("list")
@json_option
@click.pass_context
def queue_list(ctx, json_):
    """What is running and what is waiting."""
    _merge_json(ctx, json_)
    try:
        q = client(ctx).queue()
    except ComfyError as exc:
        die(str(exc))
    running = [{"prompt_id": r[1]} for r in (q.get("queue_running") or []) if len(r) > 1]
    pending = [
        {"prompt_id": r[1], "position": i + 1}
        for i, r in enumerate(q.get("queue_pending") or [])
        if len(r) > 1
    ]
    emit(
        ctx,
        {
            "running": running,
            "pending": pending,
            "running_count": len(running),
            "pending_count": len(pending),
        },
        [f"{len(running)} running, {len(pending)} pending"]
        + [f"  running  {r['prompt_id']}" for r in running]
        + [f"  #{p['position']:<3d}     {p['prompt_id']}" for p in pending[:10]],
    )


@queue.command("cancel")
@click.argument("prompt_id")
@json_option
@click.pass_context
def queue_cancel(ctx, prompt_id, json_):
    """Remove one queued prompt."""
    _merge_json(ctx, json_)
    try:
        emit(ctx, client(ctx).cancel(prompt_id), f"cancelled {prompt_id}")
    except ComfyError as exc:
        die(str(exc))


@queue.command("clear")
@json_option
@click.pass_context
def queue_clear(ctx, json_):
    """Drop everything pending."""
    _merge_json(ctx, json_)
    try:
        emit(ctx, client(ctx).clear_queue(), "queue cleared")
    except ComfyError as exc:
        die(str(exc))


@cli.group()
def history():
    """Finished prompts and the files they produced."""


@history.command("list")
@click.option("--limit", default=10, show_default=True, type=int)
@json_option
@click.pass_context
def history_list(ctx, limit, json_):
    """Recent prompts, newest last, with their output counts."""
    _merge_json(ctx, json_)
    try:
        h = client(ctx).history(max_items=limit)
    except ComfyError as exc:
        die(str(exc))
    rows = [
        {
            "prompt_id": pid,
            "outputs": len(outputs_of(entry)),
            "status": ((entry.get("status") or {}).get("status_str") or ""),
        }
        for pid, entry in (h or {}).items()
    ]
    emit(
        ctx,
        {"count": len(rows), "prompts": rows},
        [f"{len(rows)} prompt(s)"]
        + [f"  {r['prompt_id']}  {r['outputs']:>3d} file(s)  {r['status']}" for r in rows],
    )


@history.command("outputs")
@click.argument("prompt_id", required=False)
@click.option("--download", "dest", default=None, help="Download them into this directory.")
@json_option
@click.pass_context
def history_outputs(ctx, prompt_id, dest, json_):
    """What one prompt produced. Defaults to the session's last prompt."""
    _merge_json(ctx, json_)
    pid = prompt_id or _state(ctx).get("last_prompt_id")
    if not pid:
        die("no prompt id given and none in the session. Try: history list")
    c = client(ctx)
    try:
        entry = (c.history(pid) or {}).get(pid)
    except ComfyError as exc:
        die(str(exc))
    if not entry:
        die(f"no history for prompt {pid} — it may still be queued (queue list)")
    files = outputs_of(entry)
    payload = {"prompt_id": pid, "output_count": len(files), "outputs": files}
    if dest and files:
        payload["download"] = run_core.fetch(c, files, dest)
    lines = [f"prompt {pid}: {len(files)} file(s)"] + [
        f"  [{f['bucket']}] {f['filename']}" for f in files[:20]
    ]
    if payload.get("download"):
        lines.append(
            f"  downloaded {payload['download']['downloaded']} to {payload['download']['dir']}"
        )
    emit(ctx, payload, lines)


# ---------------------------------------------------------------------- assets


@cli.group()
def assets():
    """Files in and out of the server."""


@assets.command("upload")
@click.argument("path", type=click.Path(exists=True))
@click.option("--subfolder", default="", help="Subfolder inside the input directory.")
@json_option
@click.pass_context
def assets_upload(ctx, path, subfolder, json_):
    """Put a local file in the server's input directory."""
    _merge_json(ctx, json_)
    try:
        res = client(ctx).upload_image(path, subfolder=subfolder)
    except ComfyError as exc:
        die(str(exc))
    emit(
        ctx,
        res,
        f"uploaded as {res.get('name')}"
        + (f" in {res.get('subfolder')}" if res.get("subfolder") else ""),
    )


@assets.command("mask")
@click.argument("path", type=click.Path(exists=True))
@click.argument("original_ref")
@click.option("--kind", default="temp", type=click.Choice(["temp", "output"]), show_default=True)
@json_option
@click.pass_context
def assets_mask(ctx, path, original_ref, kind, json_):
    """Upload an inpainting MASK for an image already on the server.

    ORIGINAL_REF is the filename the mask lines up with, exactly as the
    LoadImage node names it. Without it the server files the mask as a loose
    picture and the inpaint graph masks nothing.
    """
    _merge_json(ctx, json_)
    try:
        res = client(ctx).upload_mask(path, original_ref, kind=kind)
    except ComfyError as exc:
        die(str(exc))
    emit(
        ctx,
        res,
        f"mask uploaded for {original_ref}" + (f" as {res.get('name')}" if res.get("name") else ""),
    )


@assets.command("download")
@click.argument("filename")
@click.argument("dest", type=click.Path())
@click.option("--subfolder", default="")
@click.option("--type", "kind", default="output", type=click.Choice(["output", "input", "temp"]))
@json_option
@click.pass_context
def assets_download(ctx, filename, dest, subfolder, kind, json_):
    """Fetch one produced file."""
    _merge_json(ctx, json_)
    try:
        res = client(ctx).download(filename, dest, subfolder, kind)
    except ComfyError as exc:
        die(str(exc))
    emit(ctx, res, f"wrote {res['path']} ({res['bytes']:,} bytes)")


# ------------------------------------------------------------------- user data


@cli.group()
def userdata():
    """Files the server keeps for a user — the saved-workflow tree.

    The UI's `Save` button writes here (user/default/workflows/*.json). The
    API-format graph you converted and ran can be pushed back into that tree,
    so it opens in the canvas like any other saved workflow.
    """


@userdata.command("list")
@click.argument("directory", default="user")
@click.option("--no-recurse", is_flag=True, help="Do not descend into subfolders.")
@click.option("--full-info", is_flag=True, help="Size and modified time per file.")
@json_option
@click.pass_context
def userdata_list_cmd(ctx, directory, no_recurse, full_info, json_):
    """List files in the server's user data tree."""
    _merge_json(ctx, json_)
    try:
        items = client(ctx).userdata_list(directory, recurse=not no_recurse, full_info=full_info)
    except ComfyError as exc:
        die(str(exc))
    items = items if isinstance(items, list) else list(items)
    emit(
        ctx,
        {"dir": directory, "count": len(items), "items": items},
        [f"{len(items)} file(s) under {directory}"] + [f"  {i}" for i in items],
    )


@userdata.command("get")
@click.argument("path")
@click.option("--out", type=click.Path(), help="Write the raw bytes here instead of stdout.")
@json_option
@click.pass_context
def userdata_get_cmd(ctx, path, out, json_):
    """Fetch one file from the server's user data tree."""
    _merge_json(ctx, json_)
    try:
        blob = client(ctx).userdata_get(path)
    except ComfyError as exc:
        die(str(exc))
    if out:
        out = os.path.abspath(os.path.expanduser(out))
        with open(out, "wb") as fh:
            fh.write(blob)
        emit(
            ctx,
            {"path": path, "out": out, "bytes": len(blob)},
            f"wrote {out} ({len(blob):,} bytes)",
        )
        return
    emit(ctx, {"path": path, "bytes": len(blob)}, blob.decode("utf-8", "replace"))


@userdata.command("put")
@click.argument("path")
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--text", default=None, help="Save this text as the file's contents instead.")
@click.option("--no-overwrite", is_flag=True, help="Refuse if the file already exists.")
@json_option
@click.pass_context
def userdata_put_cmd(ctx, path, file, text, no_overwrite, json_):
    """Save a local file (or --text) into the server's user data tree.

    Example: push a converted graph back into the server's workflow library,
    so it opens in the canvas:

      cli-anything-comfyui userdata put user/default/workflows/rerun.json rerun.json
    """
    _merge_json(ctx, json_)
    if (file is None) == (text is None):
        die("give exactly one of FILE or --text")
    if file:
        file = os.path.abspath(os.path.expanduser(file))
        with open(file, "rb") as fh:
            data = fh.read()
    else:
        data = text.encode()
    try:
        res = client(ctx).userdata_put(path, data, overwrite=not no_overwrite)
    except ComfyError as exc:
        die(str(exc))
    emit(
        ctx,
        res or {"path": path},
        f"saved {path} ({len(data):,} bytes)" + ("" if not no_overwrite else " (no overwrite)"),
    )


@userdata.command("delete")
@click.argument("path")
@json_option
@click.pass_context
def userdata_delete_cmd(ctx, path, json_):
    """Delete one file from the server's user data tree."""
    _merge_json(ctx, json_)
    try:
        client(ctx).userdata_delete(path)
    except ComfyError as exc:
        die(str(exc))
    emit(ctx, {"deleted": path}, f"deleted {path}")


# ---------------------------------------------------------------------- models


@cli.command("models")
@click.argument("folder", required=False)
@click.option("--limit", default=30, show_default=True, type=int)
@json_option
@click.pass_context
def models_cmd(ctx, folder, limit, json_):
    """Model folders, or what is installed in one."""
    _merge_json(ctx, json_)
    try:
        data = client(ctx).models(folder)
    except ComfyError as exc:
        die(str(exc))
    items = data if isinstance(data, list) else list(data)
    emit(
        ctx,
        {"folder": folder, "count": len(items), "items": items[:limit]},
        [f"{len(items)} item(s)" + (f" in {folder}" if folder else " (folders)")]
        + [f"  {i}" for i in items[:limit]],
    )


# ----------------------------------------------------------------------- traps


@cli.command("traps")
@click.option("--id", "trap_id", default=None, help="Show one trap in full.")
@json_option
@click.pass_context
def traps_cmd(ctx, trap_id, json_):
    """The recorded ways ComfyUI work goes wrong on this estate.

    Every entry is something that actually happened, not general advice.
    """
    _merge_json(ctx, json_)
    with open(os.path.join(_DATA, "traps.json"), encoding="utf-8") as fh:
        traps = _json.load(fh)
    if trap_id:
        hit = [t for t in traps if t["id"] == trap_id]
        if not hit:
            die(f"no trap {trap_id!r}. Known: {', '.join(t['id'] for t in traps)}")
        t = hit[0]
        emit(
            ctx,
            t,
            [
                f"{t['id']}  [{t['severity']}]",
                f"  what: {t['what']}",
                f"  why:  {t['why']}",
                f"  fix:  {t['fix']}",
            ],
        )
        return
    emit(
        ctx,
        {"count": len(traps), "traps": traps},
        [f"{len(traps)} recorded traps"]
        + [f"  {t['id']:28s} [{t['severity']:7s}] {t['what'][:70]}" for t in traps],
    )


# ---------------------------------------------------------------------- session


@cli.command("status")
@json_option
@click.pass_context
def status_cmd(ctx, json_):
    """Session state: server, loaded workflow, last prompt."""
    _merge_json(ctx, json_)
    st = _state(ctx)
    s = session_core.summary(st)
    c = client(ctx)
    s["url"] = s["url"] or c.url
    s["server_up"] = c.is_up()
    emit(
        ctx,
        s,
        [
            f"server:   {s['url']}  ({'up' if s['server_up'] else 'DOWN'})",
            f"workflow: {s['workflow_source'] or '(none)'}  [{s['nodes']} nodes]",
            f"last run: {s['last_prompt_id'] or '(none)'}",
            f"session:  {s['path']}",
        ],
    )


# ------------------------------------------------------------------------- repl


@cli.command("repl")
@click.pass_context
def repl(ctx):
    """Interactive shell (the default when no subcommand is given)."""
    from cli_anything.comfyui.utils.repl_skin import ReplSkin

    skin = ReplSkin("comfyui", version=__version__)
    skin.print_banner()
    pt = skin.create_prompt_session()
    commands = {
        "status": "session + server state",
        "server": "status / features / embeddings / free / interrupt / logs",
        "nodes": "list / search / schema",
        "workflow": "convert / deps / info / find / set / unset / validate / export / diff",
        "run": "queue the loaded graph and wait",
        "windows": "run several graphs, freeing VRAM between them",
        "queue": "list / cancel / clear",
        "history": "list / outputs",
        "assets": "upload / mask / download",
        "models": "installed models",
        "traps": "recorded failure modes",
        "help": "this list",
        "exit": "leave",
    }
    while True:
        try:
            line = skin.get_input(
                pt,
                project_name=os.path.basename(_state(ctx).get("workflow_source") or "no-workflow"),
                modified=False,
            )
        except (EOFError, KeyboardInterrupt):
            break
        line = (line or "").strip()
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        if line == "help":
            skin.help(commands)
            continue
        try:
            cli.main(
                args=line.split(),
                prog_name="cli-anything-comfyui",
                standalone_mode=False,
                obj=dict(ctx.obj),
            )
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001 — REPL must survive
            skin.error(str(exc))
    skin.print_goodbye()


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
