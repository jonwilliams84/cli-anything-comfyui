"""Session state: which server, which graph, what ran last.

WHY A SESSION AT ALL
    Driving ComfyUI is a sequence — load a workflow, retarget a checkpoint,
    change a seed, queue it, look at what came out. Making every step re-read
    and re-convert the canvas would be slow (conversion needs the full 1805-node
    `/object_info`) and would lose the node ids the previous step just reported.

AUTO-SAVE
    One-shot mutations save immediately. Without that, `workflow set` in a shell
    script appears to work and changes nothing, because the process exits before
    anything is written. `--dry-run` suppresses the save and says so.
"""

from __future__ import annotations

import json
import os
import time

DEFAULT_SESSION = os.path.expanduser(
    os.environ.get("COMFYUI_SESSION", "~/.config/cli-anything-comfyui/session.json"))
VERSION = 1


def _locked_save_json(path, data):
    """Write JSON under an exclusive lock.

    Opened "r+" and truncated INSIDE the lock: opening "w" truncates before the
    lock is taken, so a concurrent reader can see an empty file and a crash
    between the two leaves nothing at all.
    """
    import fcntl
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{}")
    with open(path, "r+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            fh.truncate()
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return path


def new(url=None, path=None):
    return {"_version": VERSION, "url": url or "", "workflow": None,
            "workflow_source": "", "last_prompt_id": "", "output_dir": "",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "modified": "", "_path": os.path.abspath(os.path.expanduser(path or DEFAULT_SESSION))}


def load(path=None, url=None):
    """Read the session, or hand back a fresh one. Never raises on absence."""
    p = os.path.abspath(os.path.expanduser(path or DEFAULT_SESSION))
    if not os.path.isfile(p):
        return new(url=url, path=p)
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A torn session must not wedge every command. Start clean and say so
        # through `_recovered` rather than dying on someone's half-written file.
        data = new(url=url, path=p)
        data["_recovered"] = True
        return data
    data.setdefault("_version", VERSION)
    data["_path"] = p
    if url:
        data["url"] = url
    return data


def save(state, dry_run=False):
    """Persist unless this is a dry run. Returns what happened, for reporting."""
    if dry_run:
        return {"saved": False, "dry_run": True, "path": state.get("_path")}
    out = {k: v for k, v in state.items() if not k.startswith("_")}
    out["modified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = _locked_save_json(state.get("_path") or DEFAULT_SESSION, out)
    return {"saved": True, "dry_run": False, "path": path}


def summary(state):
    wfl = state.get("workflow") or {}
    return {"path": state.get("_path"), "url": state.get("url") or "",
            "workflow_source": state.get("workflow_source") or "",
            "nodes": len(wfl), "last_prompt_id": state.get("last_prompt_id") or "",
            "output_dir": state.get("output_dir") or "",
            "modified": state.get("modified") or ""}
