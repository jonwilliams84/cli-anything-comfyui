"""Submitting graphs and collecting what came back.

The 116 driver scripts archived from `~/ai-video` contain 66 hand-rolled
`queue_prompt` implementations and 62 hand-rolled history polls. This is that,
once, with the parts they each got wrong: outputs read from every bucket rather
than just `images`, VRAM freed between windows, and the server's own rejection
surfaced instead of a bare HTTP 400.
"""

from __future__ import annotations

import os
import time

from cli_anything.comfyui.utils.comfyui_backend import outputs_of


def submit_and_wait(
    client, api_graph, timeout=1800, poll=1.0, on_tick=None, front=False, extra_data=None
):
    """Queue a graph and block until it finishes. The everyday path."""
    started = time.time()
    reply = client.submit(api_graph, front=front, extra_data=extra_data)
    prompt_id = reply.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"server accepted the prompt but returned no prompt_id: {reply}")
    done = client.wait(prompt_id, timeout=timeout, poll=poll, on_tick=on_tick)
    entry = done.get("history") or {}
    files = outputs_of(entry)
    return {
        "prompt_id": prompt_id,
        "number": reply.get("number"),
        "completed": done.get("completed"),
        "status": done.get("status"),
        "elapsed_s": round(time.time() - started, 2),
        "outputs": files,
        "output_count": len(files),
    }


def wait_and_collect(client, prompt_id, timeout=1800, poll=1.0, on_tick=None):
    """Wait for a prompt that is ALREADY on the queue, then collect its outputs.

    `submit_and_wait` covers the path where this harness queued the graph. This
    covers the other one: the prompt was queued from the canvas, another agent
    or an earlier shell, and `queue list` showed its id. Same wait, same
    every-bucket flattening, no submission.
    """
    done = client.wait(prompt_id, timeout=timeout, poll=poll, on_tick=on_tick)
    entry = done.get("history") or {}
    files = outputs_of(entry)
    return {
        "prompt_id": prompt_id,
        "completed": done.get("completed"),
        "status": done.get("status"),
        "waited_s": done.get("waited_s"),
        "outputs": files,
        "output_count": len(files),
    }


def fetch(client, files, dest_dir):
    """Download produced files and VERIFY each one landed.

    A render that "succeeded" and wrote a zero-byte file is the failure this
    catches; the server reports the filename before the write is durable.
    """
    dest_dir = os.path.abspath(os.path.expanduser(dest_dir))
    os.makedirs(dest_dir, exist_ok=True)
    got = []
    for f in files:
        target = os.path.join(dest_dir, f["filename"])
        client.download(f["filename"], target, f.get("subfolder", ""), f.get("type", "output"))
        # Re-stat rather than trusting the download's own byte count: the check
        # that matters is what is on THIS disk now.
        size = os.path.getsize(target) if os.path.exists(target) else 0
        got.append({**f, "path": target, "bytes": size, "ok": size > 0})
    return {
        "dir": dest_dir,
        "files": got,
        "downloaded": sum(1 for g in got if g["ok"]),
        "empty": [g["path"] for g in got if not g["ok"]],
    }


def run_windows(client, graphs, timeout=1800, free_between=True, on_window=None):
    """Run a sequence of graphs, freeing VRAM between them.

    The windowed render loop, which is how every long video on this estate is
    made. `free_between` is on by default because the alternative is documented:
    models stay resident, the card fills, and on 2026-09-04 that took the whole
    WSL VM down mid-render.

    A window that fails does not abort the rest — a 12-window film losing window
    7 should still hand back the other eleven.
    """
    results = []
    for i, graph in enumerate(graphs):
        label = f"window {i + 1}/{len(graphs)}"
        try:
            res = submit_and_wait(client, graph, timeout=timeout)
            res["window"] = i
            res["ok"] = True
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            res = {
                "window": i,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "outputs": [],
                "output_count": 0,
            }
        results.append(res)
        if on_window:
            on_window({**res, "label": label})
        if free_between and i < len(graphs) - 1:
            try:
                client.free()
            except Exception as exc:  # noqa: BLE001
                res.setdefault("warnings", []).append(f"free failed: {exc}")
    ok = [r for r in results if r.get("ok")]
    return {
        "windows": len(graphs),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "results": results,
        "outputs": [f for r in ok for f in r.get("outputs", [])],
    }
