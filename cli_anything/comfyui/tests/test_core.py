"""Unit tests — synthetic graphs, no server.

The centre of gravity is `to_api`. Everything else in this harness is a thin
wrapper over an HTTP call; the converter is the part that encodes knowledge, and
it is the part four hand-rolled `wf2api` scripts in `~/ai-video` got wrong.
"""

from __future__ import annotations

import io
import json
import os
import threading
import urllib.error

import pytest
from click.testing import CliRunner

from cli_anything.comfyui import comfyui_cli as cl
from cli_anything.comfyui.core import run as run_core
from cli_anything.comfyui.core import session as sess
from cli_anything.comfyui.core import workflow as wf
from cli_anything.comfyui.utils import comfyui_backend as be


# A KSampler-shaped schema, matching the real one on this box: `seed` carries
# control_after_generate, and model/positive/negative/latent_image are links.
OI = {
    "KSampler": {
        "category": "sampling",
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "sampler_name": [["euler", "dpmpp_2m"]],
                "scheduler": [["normal", "simple"]],
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"],
                "denoise": ["FLOAT", {"default": 1.0}],
            }
        },
    },
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}},
    },
    "SaveImage": {
        "output_node": True,
        "input": {
            "required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}
        },
    },
    "Reroute": {"input": {"required": {"": ["*"]}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
}


def ui(nodes, links=(), **extra):
    return {"nodes": list(nodes), "links": [list(x) for x in links], **extra}


def node(nid, ntype, widgets=None, inputs=(), mode=0, title=None):
    n = {
        "id": nid,
        "type": ntype,
        "mode": mode,
        "widgets_values": list(widgets) if widgets is not None else [],
        "inputs": [dict(i) for i in inputs],
    }
    if title:
        n["title"] = title
    return n


# ------------------------------------------------------------- format detection


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"nodes": [], "links": []}, "ui"),
        ({"1": {"class_type": "KSampler", "inputs": {}}}, "api"),
        ({"hello": "world"}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_detect_format(data, expected):
    assert wf.detect_format(data) == expected


# ------------------------------------------------------- THE off-by-one


def test_control_after_generate_does_not_shift_every_later_widget():
    """The bug this harness exists to stop.

    A real KSampler stores SEVEN values for SIX widget inputs, because `seed`
    is followed by an invisible "randomize"/"fixed" companion. Consume one slot
    per widget and `steps` becomes "randomize".
    """
    g = ui([node(1, "KSampler", [71177, "randomize", 4, 1.0, "euler", "simple", 0.9])])
    api, _ = wf.to_api(g, OI, strict=False)
    got = api["1"]["inputs"]
    assert got["seed"] == 71177
    assert got["steps"] == 4, "the companion slot was not swallowed"
    assert got["cfg"] == 1.0
    assert got["sampler_name"] == "euler"
    assert got["scheduler"] == "simple"
    assert got["denoise"] == 0.9
    assert "randomize" not in got.values()


def test_widgets_and_links_are_classified_from_the_schema():
    spec = OI["KSampler"]["input"]["required"]
    assert wf._is_widget(spec["seed"]) and wf._is_widget(spec["sampler_name"])
    assert not wf._is_widget(spec["model"])
    assert wf._has_control_after_generate(spec["seed"])
    assert not wf._has_control_after_generate(spec["steps"])


def test_a_link_becomes_an_origin_pair():
    g = ui(
        [
            node(4, "CheckpointLoaderSimple", ["a.safetensors"]),
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 7}],
            ),
        ],
        links=[[7, 4, 0, 1, 0, "MODEL"]],
    )
    api, _ = wf.to_api(g, OI, strict=False)
    assert api["1"]["inputs"]["model"] == ["4", 0]


# --------------------------------------------------------- collapsing the canvas


def test_a_bypassed_node_is_collapsed_and_its_consumer_rewired():
    """Mode 4 passes its input through. Dropping it without rewiring leaves the
    consumer pointing at nothing, which is why "I bypassed a node and the graph
    broke" happens."""
    g = ui(
        [
            node(4, "CheckpointLoaderSimple", ["a.safetensors"]),
            node(9, "VAEDecode", inputs=[{"name": "samples", "link": 1}], mode=4),
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 2}],
            ),
        ],
        links=[[1, 4, 0, 9, 0, "MODEL"], [2, 9, 0, 1, 0, "MODEL"]],
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert "9" not in api
    assert api["1"]["inputs"]["model"] == ["4", 0], "not rewired through the bypass"
    assert any(d["node"] == "9" and "bypassed" in d["why"] for d in rep["dropped"])


def test_a_reroute_is_collapsed_into_its_consumer():
    g = ui(
        [
            node(4, "CheckpointLoaderSimple", ["a.safetensors"]),
            node(5, "Reroute", inputs=[{"name": "", "link": 1}]),
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 2}],
            ),
        ],
        links=[[1, 4, 0, 5, 0, "MODEL"], [2, 5, 0, 1, 0, "MODEL"]],
    )
    api, _ = wf.to_api(g, OI, strict=False)
    assert "5" not in api
    assert api["1"]["inputs"]["model"] == ["4", 0]


def test_a_link_cycle_raises_instead_of_recursing_for_ever():
    g = ui(
        [
            node(5, "Reroute", inputs=[{"name": "", "link": 2}]),
            node(6, "Reroute", inputs=[{"name": "", "link": 1}]),
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 3}],
            ),
        ],
        links=[[1, 5, 0, 6, 0, "*"], [2, 6, 0, 5, 0, "*"], [3, 5, 0, 1, 0, "MODEL"]],
    )
    with pytest.raises(wf.WorkflowError, match="cycle"):
        wf.to_api(g, OI, strict=False)


def test_muted_nodes_are_dropped_by_default_and_kept_on_request():
    g = ui([node(1, "KSampler", [1, "fixed", 20, 8.0, "euler", "normal", 1.0], mode=2)])
    api, rep = wf.to_api(g, OI, strict=False)
    assert api == {} and any("muted" in d["why"] for d in rep["dropped"])
    api2, _ = wf.to_api(g, OI, keep_muted=True, strict=False)
    assert "1" in api2


def test_canvas_only_nodes_are_dropped_and_reported():
    g = ui([node(1, "MarkdownNote", ["# hello"]), node(2, "Note", ["x"])])
    api, rep = wf.to_api(g, OI, strict=False)
    assert api == {}
    assert {d["class_type"] for d in rep["dropped"]} == {"MarkdownNote", "Note"}


# ------------------------------------------------------- missing types, subgraphs


def test_every_missing_node_type_is_named_at_once():
    """One run must list every pack to install, not stop at the first."""
    g = ui(
        [node(1, "NotInstalledA", []), node(2, "NotInstalledB", []), node(3, "NotInstalledA", [])]
    )
    with pytest.raises(wf.WorkflowError) as e:
        wf.to_api(g, OI, strict=True)
    msg = str(e.value)
    assert "NotInstalledA" in msg and "NotInstalledB" in msg
    api, rep = wf.to_api(g, OI, strict=False)
    assert rep["missing_node_types"] == ["NotInstalledA", "NotInstalledB"]
    assert api == {}


def test_a_subgraph_is_named_not_called_a_missing_pack():
    """A UUID class_type is a subgraph instance. Calling it "not installed"
    sends people hunting for a node pack that does not exist."""
    uuid = "cf70afc4-5a03-47ce-8210-734b1de6c6bc"
    g = ui(
        [node(1, uuid, [])],
        definitions={"subgraphs": [{"id": uuid, "name": "First & Last Frame to Video"}]},
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert rep["subgraphs"] and rep["subgraphs"][0]["name"] == "First & Last Frame to Video"
    assert rep["missing_node_types"] == [], "a subgraph is not a missing pack"
    with pytest.raises(wf.WorkflowError, match="SUBGRAPHS"):
        wf.to_api(g, OI, strict=True)


def test_a_wire_to_a_dropped_node_is_removed_and_warned_about():
    g = ui(
        [
            node(1, "MarkdownNote", ["note"]),
            node(
                2,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 5}],
            ),
        ],
        links=[[5, 1, 0, 2, 0, "MODEL"]],
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert "model" not in api["2"]["inputs"]
    assert any("not in the graph" in w["why"] for w in rep["warnings"])


def test_the_already_api_path_returns_the_same_report_shape():
    """The bug that made callers KeyError depending on input format."""
    api_in = {"1": {"class_type": "KSampler", "inputs": {}}}
    api, rep = wf.to_api(api_in, OI)
    assert rep["already_api"] is True
    for key in (
        "dropped",
        "warnings",
        "missing_node_types",
        "missing",
        "subgraphs",
        "nodes_in",
        "nodes_out",
    ):
        assert key in rep, f"{key} missing from the already-API report"


def test_a_workflow_that_is_neither_format_says_so():
    with pytest.raises(wf.WorkflowError, match="not a ComfyUI workflow"):
        wf.to_api({"hello": "world"}, OI)


# ------------------------------------------------------------------- validation


def test_validate_finds_unknown_types_missing_inputs_and_dangling_wires():
    api = {
        "1": {"class_type": "Nope", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["99", 0], "filename_prefix": "x"}},
    }
    res = wf.validate(api, OI)
    assert not res["ok"]
    kinds = {p["problem"] for p in res["problems"]}
    assert any("not installed" in k for k in kinds)
    assert any("required input is missing" in k for k in kinds)
    assert any("not in the graph" in k for k in kinds)


def test_validate_passes_a_complete_graph():
    api = {"3": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "x"}}}
    assert wf.validate(api, OI)["ok"]


def test_outputs_finds_the_nodes_that_write_files():
    api = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "SaveImage", "inputs": {}},
    }
    outs = wf.outputs(api, OI)
    assert [o["node"] for o in outs] == ["2"]
    assert wf.outputs({"1": {"class_type": "KSampler", "inputs": {}}}, OI) == [], (
        "a graph with no output node writes nothing"
    )


def test_set_input_on_a_missing_node_names_the_ones_that_exist():
    api = {"1": {"class_type": "KSampler", "inputs": {}}}
    wf.set_input(api, "1", "seed", 42)
    assert api["1"]["inputs"]["seed"] == 42
    with pytest.raises(wf.WorkflowError, match="no node 99"):
        wf.set_input(api, "99", "seed", 1)


def test_find_nodes_by_type_and_title():
    api = {
        "1": {"class_type": "KSampler", "inputs": {}, "_meta": {"title": "sampler"}},
        "2": {"class_type": "SaveImage", "inputs": {}},
    }
    assert [h["node"] for h in wf.find_nodes(api, class_type="SaveImage")] == ["2"]
    assert [h["node"] for h in wf.find_nodes(api, title="sampler")] == ["1"]


# --------------------------------------------------------------------- session


def test_session_round_trips(tmp_path):
    p = tmp_path / "s.json"
    st = sess.new(url="http://x:1", path=str(p))
    st["workflow"] = {"1": {"class_type": "KSampler", "inputs": {}}}
    assert sess.save(st)["saved"] is True
    back = sess.load(str(p))
    assert back["url"] == "http://x:1"
    assert back["workflow"]["1"]["class_type"] == "KSampler"
    assert back["_path"] == str(p)


def test_dry_run_writes_nothing(tmp_path):
    p = tmp_path / "s.json"
    st = sess.new(path=str(p))
    res = sess.save(st, dry_run=True)
    assert res["saved"] is False and res["dry_run"] is True
    assert not p.exists()


def test_a_corrupt_session_recovers_rather_than_wedging_every_command(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    st = sess.load(str(p))
    assert st.get("_recovered") is True and st["workflow"] is None


def test_concurrent_saves_do_not_interleave(tmp_path):
    p = tmp_path / "s.json"

    def writer(n):
        st = sess.new(path=str(p))
        st["workflow"] = {str(i): {"class_type": "KSampler", "inputs": {}} for i in range(n)}
        sess.save(st)

    threads = [threading.Thread(target=writer, args=(i * 7 + 1,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(p, encoding="utf-8") as fh:
        json.load(fh)  # a torn write would not parse


# --------------------------------------------------------------------- backend


def test_unavailable_names_the_url_and_how_to_start_one():
    err = be.ComfyUnavailable("http://127.0.0.1:9999", "refused")
    msg = str(err)
    assert "http://127.0.0.1:9999" in msg
    assert "COMFYUI_URL" in msg and "main.py" in msg


def test_a_rejected_prompt_is_flattened_into_readable_lines():
    """`node_errors` is nested three deep; agents routinely read the raw
    envelope as an empty dict."""
    err = be.ComfyPromptRejected(
        {
            "error": {"message": "Prompt has no outputs", "details": "check SaveImage"},
            "node_errors": {
                "7": {
                    "class_type": "KSampler",
                    "errors": [
                        {
                            "message": "value not in list",
                            "extra_info": {"input_name": "sampler_name"},
                        }
                    ],
                }
            },
        }
    )
    msg = str(err)
    assert "Prompt has no outputs" in msg
    assert "node 7 (KSampler)" in msg
    assert "value not in list" in msg and "sampler_name" in msg
    assert err.node_errors["7"]["class_type"] == "KSampler"


def test_outputs_of_reads_every_bucket_not_just_images():
    """The bug that makes every video workflow look like it produced nothing."""
    entry = {
        "outputs": {
            "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            "10": {"gifs": [{"filename": "b.webp", "subfolder": "s", "type": "output"}]},
            "11": {"videos": [{"filename": "c.mp4", "subfolder": "", "type": "output"}]},
            "12": {"audio": [{"filename": "d.flac", "subfolder": "", "type": "output"}]},
            "13": {"text": "not a list — ignored"},
        }
    }
    got = be.outputs_of(entry)
    assert {f["filename"] for f in got} == {"a.png", "b.webp", "c.mp4", "d.flac"}
    assert {f["bucket"] for f in got} == {"images", "gifs", "videos", "audio"}


def test_outputs_of_survives_an_empty_or_odd_entry():
    assert be.outputs_of({}) == []
    assert be.outputs_of({"outputs": {"1": "not a dict"}}) == []
    assert be.outputs_of({"outputs": {"1": {"images": [{"no_filename": 1}]}}}) == []


def test_queue_row_helpers_tolerate_short_rows():
    assert be._qid([1, "abc", {}]) == "abc"
    assert be._qid([1]) is None
    assert be._qid("nonsense") is None
    q = {"queue_pending": [[1, "a"], [2, "b"]]}
    assert be._position(q, "b") == 2
    assert be._position(q, "missing") is None


# ----------------------------------------------------------- run core (no server)


class FakeClient:
    """A ComfyUI that answers from memory — for the parts that are logic, not HTTP."""

    def __init__(self, outputs=None, fail_on=(), drop_prompt_id=False):
        self.outputs = outputs or {}
        self.fail_on = set(fail_on)
        self.drop_prompt_id = drop_prompt_id
        self.submitted = []
        self.freed = 0
        self.wrote = {}

    def submit(self, graph, front=False, extra_data=None):
        self.submitted.append(graph)
        if len(self.submitted) - 1 in self.fail_on:
            raise be.ComfyError("window exploded")
        if self.drop_prompt_id:
            return {"number": 1}
        return {"prompt_id": f"pid-{len(self.submitted)}", "number": len(self.submitted)}

    def wait(self, prompt_id, timeout=1800, poll=1.0, on_tick=None):
        return {
            "prompt_id": prompt_id,
            "history": dict(self.outputs),
            "completed": True,
            "status": "success",
            "waited_s": 0.0,
        }

    def free(self, **kw):
        self.freed += 1
        return {"unload_models": True}

    def download(self, filename, dest, subfolder="", kind="output"):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        data = b"" if filename == "empty.png" else b"\x89PNG\r\n\x1a\n"
        with open(dest, "wb") as fh:
            fh.write(data)
        self.wrote[dest] = data
        return {"path": dest, "bytes": len(data)}


OUT_ENTRY = {
    "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}
}


def test_submit_and_wait_flattens_the_outputs():
    res = run_core.submit_and_wait(FakeClient(outputs=OUT_ENTRY), {"1": {"class_type": "X"}})
    assert res["prompt_id"] == "pid-1"
    assert res["output_count"] == 1
    assert res["outputs"][0]["filename"] == "a.png"
    assert res["completed"] is True


def test_submit_and_wait_without_a_prompt_id_raises():
    with pytest.raises(RuntimeError, match="no prompt_id"):
        run_core.submit_and_wait(FakeClient(drop_prompt_id=True), {})


def test_fetch_verifies_each_file_landed(tmp_path):
    files = [
        {"filename": "a.png", "subfolder": "", "type": "output"},
        {"filename": "empty.png", "subfolder": "", "type": "output"},
    ]
    got = run_core.fetch(FakeClient(), files, str(tmp_path))
    assert got["downloaded"] == 1
    assert got["empty"] == [got["files"][1]["path"]], "a zero-byte file must not count as fetched"
    with open(got["files"][0]["path"], "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


def test_run_windows_keeps_going_when_a_window_fails():
    c = FakeClient(outputs=OUT_ENTRY, fail_on={1})
    seen = []
    res = run_core.run_windows(c, [{}, {}, {}], on_window=seen.append)
    assert res["succeeded"] == 2 and res["failed"] == 1
    assert "ComfyError" in res["results"][1]["error"]
    assert res["results"][1]["ok"] is False
    assert c.freed == 2, "VRAM freed between windows even after a failure"
    assert len(seen) == 3 and seen[1]["label"] == "window 2/3"
    assert len(res["outputs"]) == 2, "the failed window's outputs are absent, the rest present"


def test_run_windows_can_keep_vram_resident():
    c = FakeClient(outputs=OUT_ENTRY)
    res = run_core.run_windows(c, [{}, {}], free_between=False)
    assert res["failed"] == 0 and c.freed == 0


# --------------------------------------------------- backend transport (no server)


def test_a_url_must_be_http_or_https():
    with pytest.raises(ValueError, match="http"):
        be.ComfyUI(url="file:///etc/passwd")
    with pytest.raises(ValueError, match="http"):
        be.ComfyUI(url="ftp://host/path")
    with pytest.raises(ValueError, match="http"):
        be.ComfyUI(url="http://")  # no host
    assert be.ComfyUI(url="http://10.0.0.5:8188").url == "http://10.0.0.5:8188"


def _patch_urlopen(monkeypatch, behaviour):
    calls = []

    def fake(req, timeout=None):
        calls.append((req.get_method(), req.full_url))
        return behaviour(req)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return calls


def test_an_http_error_surfaces_the_body(monkeypatch):
    exc = urllib.error.HTTPError(
        "http://x/models", 400, "Bad", {}, io.BytesIO(b'{"detail":"nope"}')
    )
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(exc))
    with pytest.raises(be.ComfyError, match="HTTP 400"):
        be.ComfyUI().models("checkpoints")


def test_a_rejected_prompt_post_raises_prompt_rejected(monkeypatch):
    body = json.dumps(
        {"error": {"message": "Failed to validate prompt"}, "node_errors": {"3": {}}}
    ).encode()
    exc = urllib.error.HTTPError("http://x/prompt", 400, "Bad", {}, io.BytesIO(body))
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(exc))
    with pytest.raises(be.ComfyPromptRejected):
        be.ComfyUI().submit({"1": {"class_type": "X", "inputs": {}}})


def test_a_dead_server_raises_comfy_unavailable(monkeypatch):
    exc = urllib.error.URLError(OSError("connection refused"))
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(exc))
    with pytest.raises(be.ComfyUnavailable, match="No ComfyUI at"):
        be.ComfyUI().system_stats()


def test_a_non_json_reply_is_an_error_not_a_crash(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: io.BytesIO(b"<html>bad gateway</html>"))
    with pytest.raises(be.ComfyError, match="did not return JSON"):
        be.ComfyUI().queue()


def test_view_returns_raw_bytes(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: io.BytesIO(b"PNGDATA"))
    assert be.ComfyUI().view("a.png") == b"PNGDATA"


def test_is_up_reflects_reachability(monkeypatch):
    c = be.ComfyUI()
    _patch_urlopen(monkeypatch, lambda req: io.BytesIO(b"{}"))
    assert c.is_up() is True
    exc = urllib.error.URLError(OSError("down"))
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(exc))
    assert c.is_up() is False


def test_submit_raises_prompt_rejected_when_the_reply_carries_node_errors(monkeypatch):
    reply = json.dumps(
        {"prompt_id": "x", "number": 1, "node_errors": {"3": {"errors": [{"message": "bad"}]}}}
    ).encode()
    _patch_urlopen(monkeypatch, lambda req: io.BytesIO(reply))
    with pytest.raises(be.ComfyPromptRejected):
        be.ComfyUI().submit({"1": {"class_type": "X", "inputs": {}}})


def test_wait_times_out_and_says_so(monkeypatch):
    c = be.ComfyUI()
    monkeypatch.setattr(c, "history", lambda pid=None, max_items=None: {})
    with pytest.raises(be.ComfyError, match="still not finished"):
        c.wait("pid", timeout=-1, poll=0)


def test_wait_reports_the_queue_position_through_on_tick(monkeypatch):
    c = be.ComfyUI()
    pages = [{}, {"pid": {"status": {"completed": True}, "outputs": {}}}]
    monkeypatch.setattr(c, "history", lambda pid=None, max_items=None: pages.pop(0))
    q = {"queue_running": [[1, "pid"]], "queue_pending": [[2, "other"], [3, "pid"]]}
    monkeypatch.setattr(c, "queue", lambda: q)
    ticks = []
    done = c.wait("pid", timeout=5, poll=0, on_tick=ticks.append)
    assert done["completed"] is True
    assert ticks[0]["running"] is True and ticks[0]["pending_position"] == 2


def test_download_writes_the_bytes_to_disk(monkeypatch, tmp_path):
    _patch_urlopen(monkeypatch, lambda req: io.BytesIO(b"PNGDATA"))
    dest = tmp_path / "a.png"
    res = be.ComfyUI().download("a.png", str(dest))
    assert res["bytes"] == 7 and dest.read_bytes() == b"PNGDATA"


def test_upload_refuses_a_missing_file(tmp_path):
    with pytest.raises(be.ComfyError, match="no such file"):
        be.ComfyUI().upload_image(str(tmp_path / "missing.png"))


# --------------------------------------------------------------- the CLI commands


runner = CliRunner()


def _write_api_graph(path, prefix="cli"):
    graph = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64}},
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], "filename_prefix": prefix},
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh)
    return path


def _seed_session(path, graph):
    st = sess.new(path=str(path))
    st["workflow"] = graph
    sess.save(st)
    return path


def test_the_cli_reports_its_version():
    assert "0.1.0" in runner.invoke(cl.cli, ["--version"]).output


def test_traps_list_and_one_in_full():
    d = json.loads(runner.invoke(cl.cli, ["--json", "traps"]).output)
    assert d["count"] >= 8 and "control-after-generate" in {t["id"] for t in d["traps"]}
    human = runner.invoke(cl.cli, ["traps", "--id", "control-after-generate"])
    assert human.exit_code == 0 and "control-after-generate" in human.output


def test_an_unknown_trap_id_names_the_known_ones():
    r = runner.invoke(cl.cli, ["traps", "--id", "nope"])
    assert r.exit_code == 1 and "control-after-generate" in r.output


def test_status_reports_session_state_even_with_the_server_down(tmp_path, monkeypatch):
    class Down:
        def __init__(self, **kw):
            self.url = kw.get("url") or "http://x"

        def is_up(self):
            return False

    monkeypatch.setattr(cl, "ComfyUI", Down)
    p = _seed_session(tmp_path / "s.json", {"1": {"class_type": "KSampler", "inputs": {}}})
    d = json.loads(runner.invoke(cl.cli, ["--json", "--session", str(p), "status"]).output)
    assert d["server_up"] is False and d["nodes"] == 1


def test_workflow_set_patches_the_session_and_find_sees_it(tmp_path):
    p = _seed_session(
        tmp_path / "s.json",
        {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 1},
                "_meta": {"title": "sampler"},
            }
        },
    )
    r = runner.invoke(cl.cli, ["--json", "--session", str(p), "workflow", "set", "1", "seed", "42"])
    assert r.exit_code == 0, r.output
    assert sess.load(str(p))["workflow"]["1"]["inputs"]["seed"] == 42
    d = json.loads(
        runner.invoke(
            cl.cli, ["--json", "--session", str(p), "workflow", "find", "--type", "KSampler"]
        ).output
    )
    assert d["count"] == 1 and d["matches"][0]["title"] == "sampler"


def test_workflow_set_without_a_workflow_says_so(tmp_path):
    r = runner.invoke(
        cl.cli,
        ["--json", "--session", str(tmp_path / "s.json"), "workflow", "set", "1", "seed", "42"],
    )
    assert r.exit_code == 1 and "no workflow loaded" in r.output


def test_workflow_convert_with_no_server_fails_loudly(tmp_path, monkeypatch):
    class Down:
        def __init__(self, **kw):
            pass

        def object_info(self):
            raise be.ComfyUnavailable("http://127.0.0.1:8188")

    monkeypatch.setattr(cl, "ComfyUI", Down)
    canvas = tmp_path / "c.json"
    canvas.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    r = runner.invoke(
        cl.cli,
        ["--json", "--session", str(tmp_path / "s.json"), "workflow", "convert", str(canvas)],
    )
    assert r.exit_code == 1 and "No ComfyUI" in r.output


def test_run_queues_the_session_graph_and_records_the_prompt_id(tmp_path, monkeypatch):
    fake = FakeClient(outputs=OUT_ENTRY)
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    p = _seed_session(tmp_path / "s.json", {"1": {"class_type": "EmptyImage", "inputs": {}}})
    dest = tmp_path / "out"
    d = json.loads(
        runner.invoke(
            cl.cli, ["--json", "--session", str(p), "run", "--download", str(dest)]
        ).output
    )
    assert d["output_count"] == 1
    assert fake.submitted, "the graph was never submitted"
    assert sess.load(str(p))["last_prompt_id"] == d["prompt_id"]
    assert d["download"]["downloaded"] == 1


def test_windows_runs_every_graph_frees_between_and_downloads(tmp_path, monkeypatch):
    fake = FakeClient(outputs=OUT_ENTRY)
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g1 = str(_write_api_graph(tmp_path / "w1.json"))
    g2 = str(_write_api_graph(tmp_path / "w2.json"))
    dest = tmp_path / "out"
    d = json.loads(
        runner.invoke(
            cl.cli,
            [
                "--json",
                "--session",
                str(tmp_path / "s.json"),
                "windows",
                g1,
                g2,
                "--download",
                str(dest),
            ],
        ).output
    )
    assert d["succeeded"] == 2 and d["failed"] == 0
    assert fake.freed == 1, "one free between two windows"
    assert d["download"]["downloaded"] == 2
    assert dest.is_dir()


def test_windows_fails_with_an_exit_code_when_a_window_fails(tmp_path, monkeypatch):
    fake = FakeClient(outputs=OUT_ENTRY, fail_on={1})
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g1 = str(_write_api_graph(tmp_path / "w1.json"))
    g2 = str(_write_api_graph(tmp_path / "w2.json"))
    r = runner.invoke(cl.cli, ["--json", "--session", str(tmp_path / "s.json"), "windows", g1, g2])
    assert r.exit_code == 1
    d = json.loads(r.output)
    assert d["succeeded"] == 1 and d["failed"] == 1


def test_server_features_and_embeddings_reach_the_server(monkeypatch):
    class Up:
        def __init__(self, **kw):
            pass

        def features(self):
            return {"supports_preview_metadata": True}

        def embeddings(self):
            return ["clip_vision", "samdash"]

    monkeypatch.setattr(cl, "ComfyUI", Up)
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "features"]).output)
    assert d["supports_preview_metadata"] is True
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "embeddings"]).output)
    assert d["count"] == 2


def test_server_features_with_no_server_names_the_problem(monkeypatch):
    class Down:
        def __init__(self, **kw):
            pass

        def features(self):
            raise be.ComfyUnavailable("http://127.0.0.1:8188")

    monkeypatch.setattr(cl, "ComfyUI", Down)
    r = runner.invoke(cl.cli, ["--json", "server", "features"])
    assert r.exit_code == 1 and "No ComfyUI" in r.output


# ----------------------------------------------------------------- refine: backend


def _capture_requests(monkeypatch, payload=b"{}"):
    """Fake urlopen, keeping every Request so bodies and headers are checkable."""
    reqs = []

    def fake(req, timeout=None):
        reqs.append(req)
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return reqs


def test_submit_sends_front_and_extra_data(monkeypatch):
    reqs = _capture_requests(monkeypatch, payload=b'{"prompt_id":"p","number":1}')
    c = be.ComfyUI()
    c.submit({"1": {"class_type": "X", "inputs": {}}}, front=True, extra_data={"note": "hi"})
    body = json.loads(reqs[0].data)
    assert body["front"] is True
    assert body["extra_data"] == {"note": "hi"}
    assert body["client_id"] == c.client_id


def test_the_mutating_endpoints_hit_their_verbs_paths_and_bodies(monkeypatch):
    reqs = _capture_requests(monkeypatch)
    c = be.ComfyUI()
    c.free(unload_models=False)
    c.clear_queue()
    c.cancel("abc")
    c.interrupt()
    assert [(r.get_method(), r.full_url) for r in reqs] == [
        ("POST", "http://127.0.0.1:8188/free"),
        ("POST", "http://127.0.0.1:8188/queue"),
        ("POST", "http://127.0.0.1:8188/queue"),
        ("POST", "http://127.0.0.1:8188/interrupt"),
    ]
    assert json.loads(reqs[0].data) == {"unload_models": False, "free_memory": True}
    assert json.loads(reqs[1].data) == {"clear": True}
    assert json.loads(reqs[2].data) == {"delete": ["abc"]}


def test_history_models_and_object_info_build_their_query_paths(monkeypatch):
    reqs = _capture_requests(monkeypatch)
    c = be.ComfyUI()
    c.history("pid-9")
    c.history(max_items=5)
    c.models("checkpoints")
    c.models()
    c.object_info("KSampler")
    assert [r.full_url for r in reqs] == [
        "http://127.0.0.1:8188/history/pid-9",
        "http://127.0.0.1:8188/history?max_items=5",
        "http://127.0.0.1:8188/models/checkpoints",
        "http://127.0.0.1:8188/models",
        "http://127.0.0.1:8188/object_info/KSampler",
    ]


def test_upload_image_builds_a_multipart_body(monkeypatch, tmp_path):
    reqs = _capture_requests(monkeypatch, payload=b'{"name":"up.png","subfolder":""}')
    f = tmp_path / "up.png"
    f.write_bytes(b"\x89PNG fake")
    c = be.ComfyUI()
    res = c.upload_image(str(f), subfolder="shots")
    assert res["name"] == "up.png"
    req = reqs[0]
    assert req.get_method() == "POST" and req.full_url.endswith("/upload/image")
    boundary = req.headers["Content-type"].split("boundary=")[1].encode()
    body = req.data
    assert b'name="subfolder"\r\n\r\nshots\r\n' in body
    assert b'name="overwrite"\r\n\r\ntrue\r\n' in body
    assert b'filename="up.png"' in body and b"\x89PNG fake" in body
    assert body.startswith(b"--" + boundary) and body.endswith(b"--" + boundary + b"--\r\n")
    # With no subfolder the field is skipped entirely, not sent empty.
    c.upload_image(str(f))
    assert b'name="subfolder"' not in reqs[1].data


# ------------------------------------------------------------------ refine: run.py


def test_run_windows_reports_a_failed_free_instead_of_dying():
    class FreeBlowsUp(FakeClient):
        def free(self, **kw):
            self.freed += 1
            raise be.ComfyError("free exploded")

    res = run_core.run_windows(FreeBlowsUp(outputs=OUT_ENTRY), [{}, {}])
    assert res["succeeded"] == 2 and res["failed"] == 0
    assert "free exploded" in res["results"][0]["warnings"][0]


# ------------------------------------------------- refine: the rest of the CLI


class FakeServer:
    """A ComfyUI answering every CLI read/write from memory."""

    def __init__(self):
        self.url = "http://127.0.0.1:8188"
        self.free_calls = []
        self.cancelled = []
        self.cleared = 0
        self.uploads = []
        self.mask_uploads = []

    def object_info(self):
        return OI

    def system_stats(self):
        return {
            "system": {
                "comfyui_version": "0.34.5",
                "os": "nt",
                "python_version": "3.12.14 (main)",
                "ram_free": 8 * 2**30,
                "argv": ["main.py", "--listen"],
            },
            "devices": [
                {
                    "name": "RTX 3090",
                    "type": "cuda",
                    "vram_total": 24 * 2**30,
                    "vram_free": 6 * 2**30,
                }
            ],
        }

    def features(self):
        return {"supports_preview_metadata": True}

    def embeddings(self):
        return ["clip_vision"]

    def queue(self):
        return {
            "queue_running": [[1, "run-1"]],
            "queue_pending": [[2, "wait-1"], [3, "wait-2"]],
        }

    def cancel(self, pid):
        self.cancelled.append(pid)
        return {"cancelled": pid}

    def clear_queue(self):
        self.cleared += 1
        return {"cleared": True}

    def history(self, prompt_id=None, max_items=None):
        if prompt_id:
            return {"pid-9": OUT_ENTRY} if prompt_id == "pid-9" else {}
        return {
            "pid-9": OUT_ENTRY,
            "pid-8": {"outputs": {}, "status": {"status_str": "success"}},
        }

    def models(self, folder=None):
        return ["a.safetensors", "b.safetensors"] if folder else ["checkpoints", "loras", "vae"]

    def free(self, **kw):
        self.free_calls.append(kw)
        return {"unload_models": kw.get("unload_models", True), "free_memory": True}

    def interrupt(self):
        return {"interrupted": True}

    def upload_image(self, path, subfolder="", overwrite=True, kind="input"):
        self.uploads.append((path, subfolder))
        return {"name": os.path.basename(path), "subfolder": subfolder}

    def upload_mask(self, path, original_ref, kind="temp"):
        self.mask_uploads.append((path, original_ref, kind))
        return {"name": os.path.basename(path)}

    def logs(self, limit=None):
        return {
            "entries": [
                {"m": "got prompt", "t": 1},
                {"m": "model loaded", "t": 2},
            ][: limit or None]
        }

    def download(self, filename, dest, subfolder="", kind="output"):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(b"\x89PNG")
        return {"path": dest, "bytes": 4}


def test_queue_list_cancel_and_clear(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "queue", "list"]).output)
    assert d["running_count"] == 1 and d["pending_count"] == 2
    assert d["pending"][1]["position"] == 2
    human = runner.invoke(cl.cli, ["queue", "list"])
    assert human.exit_code == 0 and "run-1" in human.output
    d = json.loads(runner.invoke(cl.cli, ["--json", "queue", "cancel", "wait-1"]).output)
    assert d["cancelled"] == "wait-1" and fake.cancelled == ["wait-1"]
    d = json.loads(runner.invoke(cl.cli, ["--json", "queue", "clear"]).output)
    assert d["cleared"] is True and fake.cleared == 1


def test_history_list_and_outputs_with_download(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "history", "list"]).output)
    assert d["count"] == 2 and d["prompts"][0]["outputs"] == 1
    dest = tmp_path / "out"
    d = json.loads(
        runner.invoke(
            cl.cli, ["--json", "history", "outputs", "pid-9", "--download", str(dest)]
        ).output
    )
    assert d["output_count"] == 1 and d["download"]["downloaded"] == 1
    assert (dest / "a.png").read_bytes()[:4] == b"\x89PNG"


def test_history_outputs_falls_back_to_the_session_and_reports_unknowns(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    p = str(tmp_path / "s.json")
    st = sess.new(path=p)
    st["workflow"] = {"1": {"class_type": "SaveImage", "inputs": {}}}
    st["last_prompt_id"] = "pid-9"
    sess.save(st)
    d = json.loads(runner.invoke(cl.cli, ["--json", "--session", p, "history", "outputs"]).output)
    assert d["prompt_id"] == "pid-9"
    r = runner.invoke(
        cl.cli, ["--json", "--session", str(tmp_path / "none.json"), "history", "outputs"]
    )
    assert r.exit_code == 1 and "history list" in r.output
    r = runner.invoke(cl.cli, ["--json", "history", "outputs", "nope"])
    assert r.exit_code == 1 and "no history for prompt nope" in r.output


def test_assets_upload_and_download(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    d = json.loads(
        runner.invoke(cl.cli, ["--json", "assets", "upload", str(src), "--subfolder", "s"]).output
    )
    assert d["name"] == "in.png" and fake.uploads == [(str(src), "s")]
    dest = tmp_path / "o.png"
    d = json.loads(
        runner.invoke(cl.cli, ["--json", "assets", "download", "a.png", str(dest)]).output
    )
    assert d["bytes"] == 4 and dest.read_bytes() == b"\x89PNG"


def test_models_lists_folders_and_one_folder(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "models"]).output)
    assert d["count"] == 3 and d["folder"] is None
    d = json.loads(runner.invoke(cl.cli, ["--json", "models", "checkpoints"]).output)
    assert d["folder"] == "checkpoints" and d["count"] == 2


def test_nodes_list_search_schema_and_categories(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "nodes", "list"]).output)
    assert d["count"] == len(OI)
    d = json.loads(
        runner.invoke(cl.cli, ["--json", "nodes", "list", "--category", "sampling"]).output
    )
    assert d["count"] == 1
    d = json.loads(runner.invoke(cl.cli, ["--json", "nodes", "search", "sampler"]).output)
    assert any(h["name"] == "KSampler" for h in d["matches"])
    d = json.loads(runner.invoke(cl.cli, ["--json", "nodes", "categories"]).output)
    cats = {c["category"]: c["nodes"] for c in d["categories"]}
    assert cats == {"sampling": 1, "(none)": len(OI) - 1}
    d = json.loads(runner.invoke(cl.cli, ["--json", "nodes", "schema", "KSampler"]).output)
    seed = next(r for r in d["inputs"] if r["name"] == "seed")
    assert seed["widget"] is True and seed["control_after_generate"] is True
    model = next(r for r in d["inputs"] if r["name"] == "model")
    assert model["widget"] is False
    r = runner.invoke(cl.cli, ["--json", "nodes", "schema", "Nope"])
    assert r.exit_code == 1 and "nodes search Nope" in r.output


def test_workflow_validate_info_and_outputs_over_a_fake_object_info(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g = str(_write_api_graph(tmp_path / "g.json"))
    d = json.loads(runner.invoke(cl.cli, ["--json", "workflow", "validate", "--path", g]).output)
    assert d["ok"] is False and "EmptyImage" in json.dumps(d["problems"])
    d = json.loads(runner.invoke(cl.cli, ["--json", "workflow", "info", "--path", g]).output)
    assert d["nodes"] == 2 and len(d["output_nodes"]) == 1
    d = json.loads(runner.invoke(cl.cli, ["--json", "workflow", "outputs", "--path", g]).output)
    assert d["output_count"] == 1 and d["outputs"][0]["class_type"] == "SaveImage"


def test_workflow_outputs_without_a_workflow_says_so(tmp_path):
    r = runner.invoke(
        cl.cli, ["--json", "--session", str(tmp_path / "s.json"), "workflow", "outputs"]
    )
    assert r.exit_code == 1 and "no workflow loaded" in r.output


def test_workflow_deps_names_what_is_missing(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(ui([node(1, "FoobarNode", widgets=[1])])), encoding="utf-8")
    d = json.loads(runner.invoke(cl.cli, ["--json", "workflow", "deps", str(missing)]).output)
    assert d["satisfied"] is False and "FoobarNode" in d["missing_node_types"]
    have = tmp_path / "have.json"
    have.write_text(
        json.dumps(ui([node(1, "KSampler", widgets=[7, "fixed", 4, 1.0, "euler", "simple", 1.0])])),
        encoding="utf-8",
    )
    d = json.loads(runner.invoke(cl.cli, ["--json", "workflow", "deps", str(have)]).output)
    assert d["satisfied"] is True and d["subgraphs"] == []


def test_server_free_and_interrupt(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "free"]).output)
    assert d["unload_models"] is True
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "free", "--keep-models"]).output)
    assert d["unload_models"] is False
    assert fake.free_calls[-1] == {"unload_models": False, "free_memory": True}
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "interrupt"]).output)
    assert d["interrupted"] is True


# ------------------------------------------------ refine round 2: iterate & repro


def test_unset_input_removes_the_override():
    g = {"1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 4}}}
    wf.unset_input(g, "1", "seed")
    assert "seed" not in g["1"]["inputs"]
    assert g["1"]["inputs"]["steps"] == 4


def test_unset_input_errors_on_missing_node_or_input():
    g = {"1": {"class_type": "KSampler", "inputs": {}}}
    with pytest.raises(wf.WorkflowError, match="no node 2"):
        wf.unset_input(g, "2", "seed")
    with pytest.raises(wf.WorkflowError, match="no input 'seed'"):
        wf.unset_input(g, "1", "seed")


def test_diff_graphs_reports_added_removed_and_changed_inputs():
    before = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 4}},
        "2": {"class_type": "SaveImage", "inputs": {}},
    }
    after = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 2}},  # steps gone, seed changed
        "3": {"class_type": "SaveImage", "inputs": {}},
    }
    d = wf.diff_graphs(before, after)
    assert d["same"] is False
    assert d["added"] == ["3"] and d["removed"] == ["2"]
    changes = {(c["input"], c["from"], c["to"]) for c in d["changed"][0]["changes"]}
    assert changes == {("seed", 1, 2), ("steps", 4, "(absent)")}


def test_diff_graphs_sees_a_rewire_and_identical_graphs():
    before = {
        "1": {"class_type": "A", "inputs": {"x": ["2", 0]}},
        "2": {"class_type": "B", "inputs": {}},
    }
    after = {
        "1": {"class_type": "A", "inputs": {"x": ["3", 0]}},
        "2": {"class_type": "B", "inputs": {}},
    }
    d = wf.diff_graphs(before, after)
    assert d["changed"][0]["changes"] == [{"input": "x", "from": ["2", 0], "to": ["3", 0]}]
    assert wf.diff_graphs(before, dict(before))["same"] is True
    assert wf.diff_graphs({}, {})["same"] is True


def test_upload_mask_builds_a_multipart_body_with_the_original_ref(monkeypatch, tmp_path):
    reqs = _capture_requests(monkeypatch, payload=b'{"name":"mask.png"}')
    f = tmp_path / "mask.png"
    f.write_bytes(b"\x89PNG fake mask")
    c = be.ComfyUI()
    res = c.upload_mask(str(f), "up.png", kind="temp")
    assert res["name"] == "mask.png"
    req = reqs[0]
    assert req.get_method() == "POST" and req.full_url.endswith("/upload/mask")
    boundary = req.headers["Content-type"].split("boundary=")[1].encode()
    body = req.data
    assert b'name="original_ref"\r\n\r\nup.png\r\n' in body
    assert b'name="type"\r\n\r\ntemp\r\n' in body
    assert b'filename="mask.png"' in body and b"\x89PNG fake mask" in body
    assert body.startswith(b"--" + boundary) and body.endswith(b"--" + boundary + b"--\r\n")


def test_upload_mask_refuses_a_missing_file(tmp_path):
    with pytest.raises(be.ComfyError, match="no such file"):
        be.ComfyUI().upload_mask(str(tmp_path / "nope.png"), "up.png")


def test_logs_hits_internal_logs_and_honours_a_limit(monkeypatch):
    reqs = _capture_requests(monkeypatch, payload=b'{"entries":[{"m":"hi"}]}')
    c = be.ComfyUI()
    assert c.logs(limit=50) == {"entries": [{"m": "hi"}]}
    assert reqs[0].full_url == "http://127.0.0.1:8188/internal/logs?limit=50"
    c.logs()
    assert reqs[1].full_url == "http://127.0.0.1:8188/internal/logs"


def test_workflow_export_writes_the_patched_session_graph(tmp_path):
    p = _seed_session(
        tmp_path / "s.json",
        {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}},
    )
    out = str(tmp_path / "patched.json")
    r = runner.invoke(cl.cli, ["--json", "--session", str(p), "workflow", "export", "-o", out])
    assert r.exit_code == 0, r.output
    with open(out, encoding="utf-8") as fh:
        assert json.load(fh) == {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}


def test_workflow_export_without_a_workflow_says_so(tmp_path):
    r = runner.invoke(
        cl.cli,
        ["--json", "--session", str(tmp_path / "s.json"), "workflow", "export", "-o", "x.json"],
    )
    assert r.exit_code == 1 and "no workflow loaded" in r.output


def test_workflow_diff_compares_the_session_graph_against_a_file(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g = str(_write_api_graph(tmp_path / "g.json", prefix="old"))
    session = _seed_session(tmp_path / "s.json", json.load(open(g, encoding="utf-8")))
    # patch the session, then diff it against the file
    runner.invoke(
        cl.cli,
        ["--json", "--session", str(session), "workflow", "set", "2", "filename_prefix", "new"],
    )
    d = json.loads(
        runner.invoke(cl.cli, ["--json", "--session", str(session), "workflow", "diff", g]).output
    )
    assert d["same"] is False and d["baseline"] == g
    assert d["changed"][0]["changes"] == [{"input": "filename_prefix", "from": "old", "to": "new"}]


def test_workflow_diff_reports_identical_graphs_and_two_files(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g = str(_write_api_graph(tmp_path / "g.json"))
    session = _seed_session(tmp_path / "s.json", json.load(open(g, encoding="utf-8")))
    d = json.loads(
        runner.invoke(cl.cli, ["--json", "--session", str(session), "workflow", "diff", g]).output
    )
    assert d["same"] is True and d["changed"] == []
    g2 = str(_write_api_graph(tmp_path / "g2.json", prefix="other"))
    d = json.loads(runner.invoke(cl.cli, ["--json", "workflow", "diff", g, g2]).output)
    assert d["baseline"] == g and d["changed"][0]["changes"] == [
        {"input": "filename_prefix", "from": "cli", "to": "other"}
    ]


def test_workflow_diff_with_nothing_to_compare_says_so(tmp_path):
    r = runner.invoke(cl.cli, ["--json", "--session", str(tmp_path / "s.json"), "workflow", "diff"])
    assert r.exit_code == 1 and "give a file" in r.output


def test_workflow_diff_needs_a_loaded_graph_for_the_one_path_form(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g = str(_write_api_graph(tmp_path / "g.json"))
    r = runner.invoke(
        cl.cli, ["--json", "--session", str(tmp_path / "s.json"), "workflow", "diff", g]
    )
    assert r.exit_code == 1 and "no workflow loaded" in r.output


def test_workflow_unset_removes_the_override_from_the_session(tmp_path):
    p = _seed_session(
        tmp_path / "s.json", {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    )
    r = runner.invoke(cl.cli, ["--json", "--session", str(p), "workflow", "unset", "1", "seed"])
    assert r.exit_code == 0, r.output
    assert sess.load(str(p))["workflow"]["1"]["inputs"] == {}


def test_workflow_unset_errors_name_what_exists(tmp_path):
    p = _seed_session(
        tmp_path / "s.json", {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
    )
    r = runner.invoke(cl.cli, ["--json", "--session", str(p), "workflow", "unset", "9", "seed"])
    assert r.exit_code == 1 and "no node 9" in r.output
    r = runner.invoke(cl.cli, ["--json", "--session", str(p), "workflow", "unset", "1", "cfg"])
    assert r.exit_code == 1 and "no input 'cfg'" in r.output
    r = runner.invoke(
        cl.cli,
        ["--json", "--session", str(tmp_path / "empty.json"), "workflow", "unset", "1", "cfg"],
    )
    assert r.exit_code == 1 and "no workflow loaded" in r.output


def test_assets_mask_uploads_with_the_original_ref(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    f = tmp_path / "mask.png"
    f.write_bytes(b"\x89PNG fake mask")
    d = json.loads(runner.invoke(cl.cli, ["--json", "assets", "mask", str(f), "up.png"]).output)
    assert d["name"] == "mask.png"
    assert fake.mask_uploads == [(str(f), "up.png", "temp")]


def test_server_logs_reports_the_recent_lines(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "logs", "--limit", "1"]).output)
    assert d["count"] == 1 and d["entries"][0]["m"] == "got prompt"


# ------------------------------------- refine round 3: the converter's edge paths


def test_load_names_the_missing_file(tmp_path):
    with pytest.raises(wf.WorkflowError, match="no such workflow"):
        wf.load(str(tmp_path / "absent.json"))


def test_dict_style_link_rows_are_understood_too():
    """Newer canvas builds serialise `links` as objects, not positional rows.

    Reading only the list shape converts every such workflow with every input
    silently unwired — the graph then fails validation with a wall of
    "required input is missing" that looks like a schema problem.
    """
    g = {
        "nodes": [
            node(4, "CheckpointLoaderSimple", ["a.safetensors"]),
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 7}],
            ),
        ],
        "links": [
            {
                "id": 7,
                "origin_id": 4,
                "origin_slot": 0,
                "target_id": 1,
                "target_slot": 0,
                "type": "MODEL",
            }
        ],
    }
    assert wf.link_map(g) == {7: ("4", 0)}, "the dict row was not read"
    api, _ = wf.to_api(g, OI, strict=False)
    assert api["1"]["inputs"]["model"] == ["4", 0]


def test_a_link_whose_origin_node_does_not_exist_is_dropped_and_warned():
    """Link row present, origin node absent — the canvas of a half-deleted graph."""
    g = ui(
        [
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 7}],
            )
        ],
        links=[[7, 77, 0, 1, 0, "MODEL"]],
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert "model" not in api["1"]["inputs"]
    assert any("resolves only to" in w["why"] for w in rep["warnings"])


def test_an_input_with_a_link_id_the_map_never_heard_of_is_warned_about():
    g = ui(
        [
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 9}],
            )
        ],
        links=[[7, 4, 0, 1, 0, "MODEL"]],
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert "model" not in api["1"]["inputs"]
    assert any("link 9 has no origin" in w["why"] for w in rep["warnings"])


def test_an_input_with_no_link_at_all_is_skipped_quietly():
    """A placeholder input (link: null) is canvas scaffolding, not an error."""
    g = ui(
        [
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": None}],
            )
        ]
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert "model" not in api["1"]["inputs"]
    assert rep["warnings"] == [], "a null link is not worth a warning"


def test_a_bypassed_node_with_no_source_warns_instead_of_silently_rewiring():
    """Bypassed node with NO wired input: pass-through has nothing to pass."""
    g = ui(
        [
            node(9, "VAEDecode", mode=4),
            node(
                1,
                "KSampler",
                [1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "link": 2}],
            ),
        ],
        links=[[2, 9, 0, 1, 0, "MODEL"]],
    )
    api, rep = wf.to_api(g, OI, strict=False)
    assert "model" not in api["1"]["inputs"]
    assert any("resolves only to" in w["why"] for w in rep["warnings"])


def test_widgets_serialised_by_name_skip_the_positional_pass():
    """Some packs write `widgets_values` as {name: value} — nothing positional."""
    g = ui([node(1, "KSampler")])
    g["nodes"][0]["widgets_values"] = {"seed": 5, "steps": 3}
    api, _ = wf.to_api(g, OI, strict=False)
    assert api["1"]["inputs"]["seed"] == 5
    assert api["1"]["inputs"]["steps"] == 3


def test_extra_widget_values_are_ignored_but_reported():
    """A frontend-only widget (or pack drift) leaves values the schema never
    declared. Every NAMED input must still land, and the surplus is surfaced."""
    g = ui([node(1, "KSampler", [71177, "randomize", 4, 1.0, "euler", "simple", 1.0, "EXTRA"])])
    api, rep = wf.to_api(g, OI, strict=False)
    assert api["1"]["inputs"]["denoise"] == 1.0
    extra = [w for w in rep["warnings"] if "extra_values" in w]
    assert extra and extra[0]["extra_values"] == 1
    assert "frontend-only widget" in extra[0]["why"]


def test_a_node_title_travels_into_meta():
    """`workflow find --title` only works if conversion kept the title."""
    g = ui([node(1, "KSampler", [1, "fixed", 20, 8.0, "euler", "normal", 1.0], title="hero")])
    api, _ = wf.to_api(g, OI, strict=False)
    assert api["1"]["_meta"]["title"] == "hero"
    assert wf.find_nodes(api, title="hero")[0]["node"] == "1"


def test_find_nodes_sorts_digit_ids_numerically_and_survives_non_digit_ids():
    api = {
        "10": {"class_type": "SaveImage", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {}},
        "aux": {"class_type": "SaveImage", "inputs": {}},
    }
    hits = wf.find_nodes(api, class_type="SaveImage")
    assert [h["node"] for h in hits] == ["aux", "10"], (
        "10 and 2 must not sort as strings ('10' < '2'), and 'aux' anchors at 0"
    )


def test_validate_names_an_entry_with_no_class_type():
    res = wf.validate({"1": {"inputs": {}}}, OI)
    assert not res["ok"]
    assert res["problems"] == [{"node": "1", "problem": "no class_type"}]


def test_diff_graphs_reports_a_class_type_change():
    d = wf.diff_graphs(
        {"1": {"class_type": "KSampler", "inputs": {}}},
        {"1": {"class_type": "KSamplerAdvanced", "inputs": {}}},
    )
    assert d["changed"][0]["changes"] == [
        {"input": "(class_type)", "from": "KSampler", "to": "KSamplerAdvanced"}
    ]


# ------------------------------------- refine round 3: backend and session edges


def test_an_http_error_with_a_non_json_body_still_surfaces_the_text(monkeypatch):
    """A proxy's HTML 502 page is the most common 'why is nothing answering'."""
    exc = urllib.error.HTTPError(
        "http://x/models", 502, "Bad Gateway", {}, io.BytesIO(b"<html>gateway exploded</html>")
    )
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(exc))
    with pytest.raises(be.ComfyError, match="HTTP 502.*gateway exploded"):
        be.ComfyUI().models()


def test_an_empty_reply_is_an_empty_mapping_not_a_crash(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: io.BytesIO(b""))
    assert be.ComfyUI().models() == {}


def test_features_and_embeddings_hit_their_endpoints(monkeypatch):
    reqs = _capture_requests(monkeypatch, payload=b'{"supports_preview_metadata": true}')
    c = be.ComfyUI()
    assert c.features() == {"supports_preview_metadata": True}
    assert reqs[0].full_url == "http://127.0.0.1:8188/features"
    reqs2 = _capture_requests(monkeypatch, payload=b'["clip_vision"]')
    assert c.embeddings() == ["clip_vision"]
    assert reqs2[0].full_url == "http://127.0.0.1:8188/embeddings"


def test_wait_polls_and_sleeps_without_an_on_tick(monkeypatch):
    """The no-progress-callback path still polls, sleeps, and times out cleanly."""
    c = be.ComfyUI()
    monkeypatch.setattr(c, "history", lambda pid=None, max_items=None: {})
    with pytest.raises(be.ComfyError, match="still not finished"):
        c.wait("pid", timeout=0.2, poll=0)


def test_load_honours_an_explicit_url_over_the_stored_one(tmp_path):
    """`--url` must beat whatever the session file recorded last time."""
    p = tmp_path / "s.json"
    st = sess.new(url="http://stored:1", path=str(p))
    sess.save(st)
    assert sess.load(str(p), url="http://override:2")["url"] == "http://override:2"


# ------------------------------------- refine round 3: the CLI commands themselves


def test_server_status_reports_version_os_and_vram(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    d = json.loads(runner.invoke(cl.cli, ["--json", "server", "status"]).output)
    assert d["up"] is True and d["comfyui_version"] == "0.34.5"
    assert d["devices"][0]["vram_free_gb"] == 6.0
    assert d["devices"][0]["vram_total_gb"] == 24.0
    assert d["ram_free_gb"] == 8.0
    human = runner.invoke(cl.cli, ["server", "status"])
    assert human.exit_code == 0
    assert "ComfyUI 0.34.5 on nt" in human.output
    assert "RTX 3090" in human.output and "6.0/24.0 GB" in human.output


def test_a_command_without_json_falls_back_to_the_json_printer(monkeypatch):
    """`emit`'s third branch: no --json, no human rendering — JSON anyway."""
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: FakeServer())
    d = json.loads(runner.invoke(cl.cli, ["server", "features"]).output)
    assert d["supports_preview_metadata"] is True


def test_workflow_convert_takes_a_canvas_loads_the_session_and_writes_a_file(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    canvas = tmp_path / "canvas.json"
    canvas.write_text(
        json.dumps(
            ui(
                [
                    node(4, "CheckpointLoaderSimple", ["a.safetensors"]),
                    node(
                        1,
                        "KSampler",
                        [71177, "randomize", 4, 1.0, "euler", "simple", 1.0],
                        inputs=[{"name": "model", "link": 7}],
                        title="hero",
                    ),
                    node(
                        2,
                        "SaveImage",
                        ["comfy"],
                        inputs=[{"name": "images", "link": 8}],
                    ),
                    node(3, "VAEDecode"),
                ],
                links=[[7, 4, 0, 1, 0, "MODEL"], [8, 1, 0, 2, 0, "IMAGE"]],
            ),
        ),
        encoding="utf-8",
    )
    out = str(tmp_path / "api.json")
    session = str(tmp_path / "s.json")
    d = json.loads(
        runner.invoke(
            cl.cli,
            [
                "--json",
                "--session",
                session,
                "workflow",
                "convert",
                str(canvas),
                "-o",
                out,
            ],
        ).output
    )
    assert d["nodes"] == 4 and d["dropped"] == [], "every node has a schema here and is kept"
    assert sess.load(session)["workflow"]["1"]["_meta"]["title"] == "hero"
    with open(out, encoding="utf-8") as fh:
        assert json.load(fh)["2"]["class_type"] == "SaveImage"


def test_run_reports_a_rejected_graph_on_stderr_with_exit_1(tmp_path, monkeypatch):
    fake = FakeClient(fail_on={0})
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    p = _seed_session(tmp_path / "s.json", {"1": {"class_type": "EmptyImage", "inputs": {}}})
    r = runner.invoke(cl.cli, ["--json", "--session", str(p), "run"])
    assert r.exit_code == 1 and "window exploded" in r.output
    assert sess.load(str(p))["last_prompt_id"] == "", "a failed run records no prompt id"


def test_workflow_validate_exits_1_when_the_graph_is_rejectable(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(cl, "ComfyUI", lambda **kw: fake)
    g = str(_write_api_graph(tmp_path / "g.json"))
    r = runner.invoke(cl.cli, ["workflow", "validate", "--path", g])
    assert r.exit_code == 1, "an invalid graph must fail the shell, not just print"
    assert "problem(s)" in r.output and "not installed" in r.output
