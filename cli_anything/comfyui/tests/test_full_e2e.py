"""E2E — against a REAL ComfyUI. No graceful degradation.

Per HARNESS.md the software is a hard dependency. If ComfyUI is not reachable
these tests FAIL; they do not skip. A harness whose tests go green with no
backend is a harness that proves nothing.

Written against ComfyUI Desktop 0.34.5 on Windows, RTX 3090, reached from WSL at
127.0.0.1:8188. Override with COMFYUI_URL.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import pytest

from cli_anything.comfyui.core import run as run_core
from cli_anything.comfyui.core import workflow as wf
from cli_anything.comfyui.utils.comfyui_backend import ComfyUI, outputs_of

#: Jon's saved canvases, archived during the 2026-09-05 WSL->Windows migration.
CORPUS = "/mnt/e/ai-video-archive/workbench/workflows/comfyui-saved"


def _resolve_cli(name):
    """Resolve the installed CLI; fall back to `python -m` for development."""
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = "cli_anything.comfyui.comfyui_cli"
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]


@pytest.fixture(scope="session")
def client():
    c = ComfyUI()
    if not c.is_up():
        pytest.fail(
            f"No ComfyUI at {c.url}. These tests drive the REAL software and do not "
            f"skip — start ComfyUI (Desktop, or python main.py --listen 0.0.0.0) or "
            f"set COMFYUI_URL.")
    return c


@pytest.fixture(scope="session")
def object_info(client):
    return client.object_info()


# ----------------------------------------------------------------- the server

def test_the_server_reports_a_version_and_a_device(client):
    s = client.system_stats()
    assert s["system"]["comfyui_version"]
    assert s["devices"], "no compute device — ComfyUI cannot render"
    print(f"\n  ComfyUI {s['system']['comfyui_version']} on {s['system']['os']}, "
          f"{len(s['devices'])} device(s)")


def test_object_info_is_populated_and_ksampler_carries_the_flag(object_info):
    """The whole converter rests on control_after_generate being discoverable."""
    assert len(object_info) > 100, f"only {len(object_info)} node types — is this ComfyUI healthy?"
    assert "KSampler" in object_info
    seed = object_info["KSampler"]["input"]["required"]["seed"]
    assert wf._has_control_after_generate(seed), \
        "KSampler.seed no longer declares control_after_generate — the converter's core assumption"
    print(f"\n  {len(object_info)} node types installed")


def test_free_vram_returns_cleanly(client):
    assert client.free()["unload_models"] is True


# ------------------------------------------------- the real 48-canvas corpus

@pytest.mark.skipif(not os.path.isdir(CORPUS), reason="archived canvas corpus not mounted")
def test_every_archived_canvas_converts_or_explains_itself(object_info):
    """None of 48 real workflows may raise an UNEXPECTED exception.

    Missing packs and subgraphs are legitimate, reported outcomes. A traceback
    is not.
    """
    files = sorted(glob.glob(os.path.join(CORPUS, "*.json")))
    assert files, "corpus is mounted but empty"
    outcomes = {"clean": 0, "needs_packs": 0, "subgraph": 0, "already_api": 0}
    for f in files:
        ui = wf.load(f)
        api, rep = wf.to_api(ui, object_info, strict=False)   # must not raise
        if rep["already_api"]:
            outcomes["already_api"] += 1
        elif rep["missing_node_types"]:
            outcomes["needs_packs"] += 1
        elif rep["subgraphs"]:
            outcomes["subgraph"] += 1
        else:
            outcomes["clean"] += 1
    print(f"\n  {len(files)} canvases: {outcomes}")
    assert sum(outcomes.values()) == len(files)
    assert outcomes["clean"] > 0, "not one real canvas converted cleanly"


# ------------------------------------------------------------- a real render

def _minimal_graph(object_info):
    """The smallest graph that writes a file using nodes this box actually has."""
    for req in ("EmptyImage", "SaveImage"):
        if req not in object_info:
            pytest.fail(f"{req} is not installed — cannot build a minimal render")
    return {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
        "2": {"class_type": "SaveImage",
              "inputs": {"images": ["1", 0], "filename_prefix": "cli_anything_e2e"}},
    }


def test_a_real_render_produces_a_real_png(client, object_info, tmp_path):
    """Submit, wait, download, and check the MAGIC BYTES.

    "It exited 0" is not evidence that anything was produced.
    """
    graph = _minimal_graph(object_info)
    assert wf.validate(graph, object_info)["ok"]
    res = run_core.submit_and_wait(client, graph, timeout=300)
    assert res["output_count"] > 0, "the render produced no files"
    got = run_core.fetch(client, res["outputs"], str(tmp_path))
    assert got["empty"] == [], f"zero-byte outputs: {got['empty']}"
    first = got["files"][0]
    with open(first["path"], "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n", "not a PNG"
    print(f"\n  PNG: {first['path']} ({first['bytes']:,} bytes) in {res['elapsed_s']}s")


def test_the_queue_is_readable_and_history_carries_the_prompt(client, object_info):
    graph = _minimal_graph(object_info)
    res = run_core.submit_and_wait(client, graph, timeout=300)
    q = client.queue()
    assert "queue_running" in q and "queue_pending" in q
    entry = (client.history(res["prompt_id"]) or {}).get(res["prompt_id"])
    assert entry, "the finished prompt is not in history"
    assert outputs_of(entry), "history has the prompt but no outputs"


def test_a_graph_with_no_output_node_is_identified_before_it_wastes_a_render(object_info):
    """The quietest failure in ComfyUI: it runs, succeeds, and writes nothing."""
    graph = {"1": {"class_type": "EmptyImage",
                   "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}}}
    assert wf.outputs(graph, object_info) == []


def test_upload_then_download_round_trips(client, object_info, tmp_path):
    graph = _minimal_graph(object_info)
    res = run_core.submit_and_wait(client, graph, timeout=300)
    got = run_core.fetch(client, res["outputs"], str(tmp_path))
    src = got["files"][0]["path"]
    up = client.upload_image(src, subfolder="cli_anything_e2e")
    assert up.get("name"), f"upload returned no name: {up}"
    print(f"\n  uploaded as {up.get('subfolder', '')}/{up['name']}")


# --------------------------------------------------------- the installed CLI

class TestCLISubprocess:
    """The command as a user or agent actually invokes it. No cwd is set."""

    CLI_BASE = _resolve_cli("cli-anything-comfyui")

    def _run(self, args, check=True):
        proc = subprocess.run(self.CLI_BASE + args, capture_output=True, text=True,
                              timeout=600, check=False)
        if check:
            assert proc.returncode == 0, f"{args} -> {proc.returncode}\n{proc.stderr[-2000:]}"
        return proc

    def test_help(self):
        assert "ComfyUI" in self._run(["--help"]).stdout

    def test_version(self):
        assert "0.1.0" in self._run(["--version"]).stdout

    def test_server_status_json(self):
        d = json.loads(self._run(["--json", "server", "status"]).stdout)
        assert d["up"] is True and d["comfyui_version"]
        assert d["devices"], "no devices reported"

    def test_nodes_schema_marks_widgets_and_the_flag(self):
        d = json.loads(self._run(["--json", "nodes", "schema", "KSampler"]).stdout)
        by = {i["name"]: i for i in d["inputs"]}
        assert by["seed"]["widget"] is True and by["seed"]["control_after_generate"] is True
        assert by["model"]["widget"] is False

    def test_nodes_search_finds_something(self):
        d = json.loads(self._run(["--json", "nodes", "search", "save"]).stdout)
        assert d["count"] > 0

    def test_traps_are_listed(self):
        d = json.loads(self._run(["--json", "traps"]).stdout)
        assert d["count"] >= 5
        assert any(t["id"] == "control-after-generate" for t in d["traps"])

    def test_models_lists_folders(self):
        d = json.loads(self._run(["--json", "models"]).stdout)
        assert d["count"] > 0

    @pytest.mark.skipif(not os.path.isdir(CORPUS), reason="canvas corpus not mounted")
    def test_convert_then_info_on_a_real_canvas(self, tmp_path):
        canvas = os.path.join(CORPUS, "Minimax H3 - Text to image.json")
        if not os.path.isfile(canvas):
            pytest.skip("that canvas is not in the corpus")
        sess = str(tmp_path / "s.json")
        out = str(tmp_path / "api.json")
        d = json.loads(self._run(["--json", "--session", sess, "workflow", "convert",
                                  canvas, "-o", out]).stdout)
        assert d["nodes"] > 0 and os.path.exists(out)
        info = json.loads(self._run(["--json", "--session", sess, "workflow", "info"]).stdout)
        assert info["output_nodes"], "a real canvas with no output node?"

    def test_full_workflow_convert_set_validate_run(self, tmp_path):
        """Scenario B end to end, through the installed command only."""
        sess = str(tmp_path / "s.json")
        graph = str(tmp_path / "g.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump({"1": {"class_type": "EmptyImage",
                             "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
                       "2": {"class_type": "SaveImage",
                             "inputs": {"images": ["1", 0], "filename_prefix": "cli_e2e_sub"}}}, fh)
        self._run(["--json", "--session", sess, "workflow", "convert", graph])
        self._run(["--json", "--session", sess, "workflow", "set", "1", "width", "96"])
        v = json.loads(self._run(["--json", "--session", sess, "workflow", "validate"]).stdout)
        assert v["ok"], v["problems"]
        dl = str(tmp_path / "out")
        r = json.loads(self._run(["--json", "--session", sess, "run", "--download", dl]).stdout)
        assert r["output_count"] > 0
        files = [f for f in r["download"]["files"] if f["ok"]]
        assert files, "run reported outputs but nothing downloaded"
        with open(files[0]["path"], "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
        print(f"\n  subprocess render: {files[0]['path']} ({files[0]['bytes']:,} bytes)")

    def test_dry_run_does_not_write_the_session(self, tmp_path):
        sess = str(tmp_path / "s.json")
        graph = str(tmp_path / "g.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump({"1": {"class_type": "EmptyImage",
                             "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}}}, fh)
        self._run(["--json", "--dry-run", "--session", sess, "workflow", "convert", graph])
        assert not os.path.exists(sess), "--dry-run wrote the session anyway"

    def test_a_missing_node_type_fails_loudly(self, tmp_path):
        graph = str(tmp_path / "bad.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump({"nodes": [{"id": 1, "type": "DefinitelyNotInstalled",
                                  "widgets_values": [], "inputs": []}], "links": []}, fh)
        proc = self._run(["workflow", "convert", graph], check=False)
        assert proc.returncode != 0
        assert "DefinitelyNotInstalled" in (proc.stdout + proc.stderr)
