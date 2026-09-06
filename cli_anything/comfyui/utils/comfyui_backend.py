"""The real ComfyUI, over HTTP.

WHY THIS IS THE WHOLE BACKEND
    ComfyUI has no headless "convert this file" mode. There is no `melt`, no
    `libreoffice --headless`. Everything a human does in the canvas is an HTTP
    call to a server that is holding several gigabytes of model weights in VRAM,
    and that server is the only thing that can execute a graph. So this module
    is an HTTP client to the REAL ComfyUI — the same surface the browser uses —
    and the server is a HARD dependency. Nothing here renders anything itself.

WHAT IT REFUSES TO DO
    It does not start ComfyUI, and it does not fall back to anything when the
    server is absent. A missing server raises :class:`ComfyUnavailable` carrying
    the URL it tried and how to start one, because an agent that gets a polite
    empty result instead of an error will happily "succeed" at doing nothing.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
DEFAULT_TIMEOUT = float(os.environ.get("COMFYUI_TIMEOUT", "60"))


def _http_only(url):
    """Accept http/https and nothing else.

    The URL arrives from `--url`, `$COMFYUI_URL` or a session file, and every
    request below goes through `urllib.request.urlopen`, which will happily open
    `file:///etc/passwd` if asked. Pinning the scheme here closes that at the one
    place a URL enters the object.
    """
    parsed = urllib.parse.urlsplit((url or "").rstrip("/"))
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"ComfyUI URL must be http:// or https://, got {url!r}. Example: http://127.0.0.1:8188"
        )
    return (url or "").rstrip("/")


class ComfyError(RuntimeError):
    """Any failure that came back from the server itself."""


class ComfyUnavailable(ComfyError):
    """No server answered. Carries what to do about it."""

    def __init__(self, url, cause=""):
        super().__init__(
            f"No ComfyUI at {url}" + (f" ({cause})" if cause else "") + ".\n"
            "Start it, then retry:\n"
            "  Desktop app, or:  python main.py --listen 0.0.0.0 --port 8188\n"
            "If it runs on another host or port, set COMFYUI_URL "
            "(e.g. COMFYUI_URL=http://10.0.0.5:8188) or pass --url.\n"
            "This harness drives a running ComfyUI; it cannot render without one."
        )
        self.url = url


class ComfyPromptRejected(ComfyError):
    """`POST /prompt` refused the graph. Carries the per-node reasons.

    The server's rejection is the single most useful error this harness can
    surface: it names the node, the input, and what was wrong with it. Flatten
    it rather than dumping the raw envelope, because `node_errors` is nested
    three deep and agents routinely mistake it for an empty dict.
    """

    def __init__(self, payload):
        self.payload = payload or {}
        err = self.payload.get("error") or {}
        lines = [err.get("message") or "prompt rejected"]
        if err.get("details"):
            lines.append(f"  {err['details']}")
        for node_id, ne in (self.payload.get("node_errors") or {}).items():
            title = (ne.get("class_type") or "?") if isinstance(ne, dict) else "?"
            lines.append(f"  node {node_id} ({title}):")
            for e in (ne.get("errors") or []) if isinstance(ne, dict) else []:
                where = e.get("details") or e.get("extra_info", {}).get("input_name") or ""
                lines.append(f"    - {e.get('message', '?')}" + (f" [{where}]" if where else ""))
        super().__init__("\n".join(lines))

    @property
    def node_errors(self):
        return self.payload.get("node_errors") or {}


class ComfyUI:
    """A connection to one running ComfyUI.

    `client_id` is stable for the life of the object so the websocket progress
    stream and the prompts submitted over HTTP belong to the same client — the
    server keys its per-client events on it, and a fresh id per request is why
    "I queued it but saw no progress" happens.
    """

    def __init__(self, url=DEFAULT_URL, timeout=DEFAULT_TIMEOUT, client_id=None):
        self.url = _http_only(url or DEFAULT_URL)
        self.timeout = float(timeout)
        self.client_id = client_id or str(uuid.uuid4())

    # ------------------------------------------------------------- transport

    def _request(self, method, path, data=None, headers=None, raw=False, timeout=None):
        url = f"{self.url}{path}"
        body = None
        hdr = dict(headers or {})
        if data is not None and not isinstance(data, (bytes, bytearray)):
            body = json.dumps(data).encode()
            hdr.setdefault("Content-Type", "application/json")
        elif data is not None:
            body = data
        req = urllib.request.Request(url, body, hdr, method=method)  # noqa: S310
        try:
            # S310/B310: the scheme is pinned to http/https by _http_only()
            # when the client is constructed, so `file:` and custom schemes
            # cannot reach here.
            opened = urllib.request.urlopen(  # noqa: S310  # nosec B310
                req, timeout=timeout or self.timeout
            )
            with opened as r:
                payload = r.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()
            try:
                parsed = json.loads(detail)
            except Exception:
                parsed = None
            if path == "/prompt" and parsed:
                raise ComfyPromptRejected(parsed) from exc
            text = (detail or b"").decode("utf-8", "replace")[:600]
            raise ComfyError(f"{method} {path} -> HTTP {exc.code}: {text}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ComfyUnavailable(self.url, getattr(exc, "reason", exc)) from exc
        if raw:
            return payload
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ComfyError(f"{method} {path} did not return JSON: {payload[:200]!r}") from exc

    # ------------------------------------------------------------ read-only

    def system_stats(self):
        return self._request("GET", "/system_stats")

    def features(self):
        return self._request("GET", "/features")

    def object_info(self, node_class=None):
        path = f"/object_info/{urllib.parse.quote(node_class)}" if node_class else "/object_info"
        return self._request("GET", path, timeout=max(self.timeout, 120))

    def models(self, folder=None):
        return self._request(
            "GET", f"/models/{urllib.parse.quote(folder)}" if folder else "/models"
        )

    def embeddings(self):
        return self._request("GET", "/embeddings")

    def queue(self):
        return self._request("GET", "/queue")

    def history(self, prompt_id=None, max_items=None):
        if prompt_id:
            return self._request("GET", f"/history/{urllib.parse.quote(prompt_id)}")
        q = f"?max_items={int(max_items)}" if max_items else ""
        return self._request("GET", f"/history{q}")

    def is_up(self):
        try:
            self.system_stats()
            return True
        except ComfyError:
            return False

    # -------------------------------------------------------------- mutating

    def submit(self, api_graph, front=False, extra_data=None):
        """Queue an API-format graph. Returns the server's reply verbatim."""
        body = {"prompt": api_graph, "client_id": self.client_id}
        if front:
            body["front"] = True
        if extra_data:
            body["extra_data"] = extra_data
        reply = self._request("POST", "/prompt", body)
        if isinstance(reply, dict) and reply.get("node_errors"):
            raise ComfyPromptRejected(reply)
        return reply

    def interrupt(self):
        self._request("POST", "/interrupt")
        return {"interrupted": True}

    def free(self, unload_models=True, free_memory=True):
        """Unload models / free VRAM.

        The OOM lever. A long windowed render that never calls this is the
        documented way to make ComfyUI die between windows.
        """
        self._request(
            "POST",
            "/free",
            {"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
        )
        return {"unload_models": bool(unload_models), "free_memory": bool(free_memory)}

    def clear_queue(self):
        self._request("POST", "/queue", {"clear": True})
        return {"cleared": True}

    def cancel(self, prompt_id):
        self._request("POST", "/queue", {"delete": [prompt_id]})
        return {"cancelled": prompt_id}

    def upload_image(self, path, subfolder="", overwrite=True, kind="input"):
        """Multipart upload into the server's input directory.

        Built by hand rather than with `requests` so the harness keeps a
        stdlib-only dependency footprint.
        """
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise ComfyError(f"no such file: {path}")
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        boundary = f"----comfycli{uuid.uuid4().hex}"
        parts = []
        for key, val in (
            ("subfolder", subfolder),
            ("overwrite", "true" if overwrite else "false"),
            ("type", kind),
        ):
            if val == "":
                continue
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode()
            )
        with open(path, "rb") as fh:
            blob = fh.read()
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
            + blob
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        return self._request(
            "POST",
            "/upload/image",
            b"".join(parts),
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=max(self.timeout, 300),
        )

    def upload_mask(self, path, original_ref, kind="temp"):
        """Multipart upload of an inpainting MASK (POST /upload/mask).

        Not the same call as `upload_image`: the server needs `original_ref` —
        the name of the image the mask belongs to — so it can line the mask up
        with the original instead of filing it as a loose picture. Without it
        an inpaint graph silently masks nothing.
        """
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise ComfyError(f"no such file: {path}")
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        boundary = f"----comfycli{uuid.uuid4().hex}"
        parts = []
        for key, val in (("original_ref", original_ref), ("type", kind)):
            if val:
                parts.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
                    f"\r\n\r\n{val}\r\n".encode()
                )
        with open(path, "rb") as fh:
            blob = fh.read()
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
            + blob
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        return self._request(
            "POST",
            "/upload/mask",
            b"".join(parts),
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=max(self.timeout, 300),
        )

    def logs(self, limit=None):
        """The server's own log lines (GET /internal/logs), oldest first.

        A rejected prompt tells you WHAT was refused; the log tells you what
        happened around it — the OOM warning, the missing pack, the failed
        checkpoint load. Newer ComfyUI builds expose this; an older one answers
        HTTP 404, which surfaces as an ordinary ComfyError.
        """
        q = f"?limit={int(limit)}" if limit else ""
        return self._request("GET", f"/internal/logs{q}")

    def view(self, filename, subfolder="", kind="output"):
        """Raw bytes of one produced file."""
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": kind})
        return self._request("GET", f"/view?{q}", raw=True, timeout=max(self.timeout, 300))

    def download(self, filename, dest, subfolder="", kind="output"):
        blob = self.view(filename, subfolder, kind)
        dest = os.path.abspath(os.path.expanduser(dest))
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(blob)
        return {"path": dest, "bytes": len(blob)}

    # --------------------------------------------------------------- waiting

    def wait(self, prompt_id, timeout=1800, poll=1.0, on_tick=None):
        """Block until `prompt_id` leaves the queue, then return its history.

        Polls rather than reading the websocket on purpose: a poll cannot miss
        an event that fired before the listener attached, which is the failure
        mode when a caller submits and only then subscribes. `on_tick` gets the
        live queue position so a caller can show progress.
        """
        started = time.time()
        while True:
            hist = self.history(prompt_id)
            entry = hist.get(prompt_id) if isinstance(hist, dict) else None
            if entry:
                status = entry.get("status") or {}
                return {
                    "prompt_id": prompt_id,
                    "history": entry,
                    "completed": bool(status.get("completed", True)),
                    "status": status.get("status_str") or "",
                    "waited_s": round(time.time() - started, 2),
                }
            if time.time() - started > timeout:
                raise ComfyError(
                    f"prompt {prompt_id} still not finished after {timeout}s. "
                    "It may still be running — check `queue list`, and raise --timeout "
                    "for long video renders."
                )
            if on_tick:
                q = self.queue()
                running = [r for r in (q.get("queue_running") or []) if _qid(r) == prompt_id]
                on_tick(
                    {
                        "running": bool(running),
                        "pending_position": _position(q, prompt_id),
                        "waited_s": round(time.time() - started, 1),
                    }
                )
            time.sleep(poll)


def _qid(row):
    """A queue row is a positional tuple: [number, prompt_id, prompt, extra, outputs]."""
    return row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else None


def _position(queue, prompt_id):
    for i, row in enumerate(queue.get("queue_pending") or []):
        if _qid(row) == prompt_id:
            return i + 1
    return None


def outputs_of(history_entry):
    """Flatten a history entry into a list of produced files.

    ComfyUI buckets outputs by node id and then by KIND — `images`, `gifs`,
    `videos`, `audio` — and the kind depends on which SaveX node ran. A caller
    that only looks at `images` silently finds nothing for every video
    workflow, which is most of this estate's.
    """
    files = []
    for node_id, out in (history_entry.get("outputs") or {}).items():
        if not isinstance(out, dict):
            continue
        for kind, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and it.get("filename"):
                    files.append(
                        {
                            "node_id": node_id,
                            "bucket": kind,
                            "filename": it["filename"],
                            "subfolder": it.get("subfolder", ""),
                            "type": it.get("type", "output"),
                        }
                    )
    return files
