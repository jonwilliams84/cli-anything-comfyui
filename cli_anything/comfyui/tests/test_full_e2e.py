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
            f"set COMFYUI_URL."
        )
    return c


@pytest.fixture(scope="session")
def object_info(client):
    return client.object_info()


# ----------------------------------------------------------------- the server


def test_the_server_reports_a_version_and_a_device(client):
    s = client.system_stats()
    assert s["system"]["comfyui_version"]
    assert s["devices"], "no compute device — ComfyUI cannot render"
    print(
        f"\n  ComfyUI {s['system']['comfyui_version']} on {s['system']['os']}, "
        f"{len(s['devices'])} device(s)"
    )


def test_object_info_is_populated_and_ksampler_carries_the_flag(object_info):
    """The whole converter rests on control_after_generate being discoverable."""
    assert len(object_info) > 100, f"only {len(object_info)} node types — is this ComfyUI healthy?"
    assert "KSampler" in object_info
    seed = object_info["KSampler"]["input"]["required"]["seed"]
    assert wf._has_control_after_generate(seed), (
        "KSampler.seed no longer declares control_after_generate — the converter's core assumption"
    )
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
        api, rep = wf.to_api(ui, object_info, strict=False)  # must not raise
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
        "1": {
            "class_type": "EmptyImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], "filename_prefix": "cli_anything_e2e"},
        },
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
    graph = {
        "1": {
            "class_type": "EmptyImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
        }
    }
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
        proc = subprocess.run(
            self.CLI_BASE + args, capture_output=True, text=True, timeout=600, check=False
        )
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
        d = json.loads(
            self._run(
                ["--json", "--session", sess, "workflow", "convert", canvas, "-o", out]
            ).stdout
        )
        assert d["nodes"] > 0 and os.path.exists(out)
        info = json.loads(self._run(["--json", "--session", sess, "workflow", "info"]).stdout)
        assert info["output_nodes"], "a real canvas with no output node?"

    def test_full_workflow_convert_set_validate_run(self, tmp_path):
        """Scenario B end to end, through the installed command only."""
        sess = str(tmp_path / "s.json")
        graph = str(tmp_path / "g.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "1": {
                        "class_type": "EmptyImage",
                        "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
                    },
                    "2": {
                        "class_type": "SaveImage",
                        "inputs": {"images": ["1", 0], "filename_prefix": "cli_e2e_sub"},
                    },
                },
                fh,
            )
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

    def test_history_show_reports_a_real_render_and_writes_its_graph(self, tmp_path):
        """`history show` on a prompt this very run produced: the status is
        there, the outputs are there, and `--graph` hands back the exact graph
        the server ran."""
        sess = str(tmp_path / "s.json")
        graph = str(tmp_path / "g.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "1": {
                        "class_type": "EmptyImage",
                        "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0},
                    },
                    "2": {
                        "class_type": "SaveImage",
                        "inputs": {"images": ["1", 0], "filename_prefix": "cli_e2e_show"},
                    },
                },
                fh,
            )
        self._run(["--json", "--session", sess, "workflow", "convert", graph])
        r = json.loads(self._run(["--json", "--session", sess, "run"]).stdout)
        pid = r["prompt_id"]
        ran = str(tmp_path / "ran.json")
        d = json.loads(
            self._run(["--json", "--session", sess, "history", "show", pid, "--graph", ran]).stdout
        )
        assert d["prompt_id"] == pid and d["status"] == "success"
        assert d["completed"] is True
        assert d["output_count"] > 0 and d["outputs"][0]["filename"].endswith(".png")
        assert d["graph_nodes"] == 2 and d["graph"] == os.path.abspath(ran)
        with open(ran, encoding="utf-8") as fh:
            assert json.load(fh)["2"]["inputs"]["filename_prefix"] == "cli_e2e_show"

    def test_dry_run_does_not_write_the_session(self, tmp_path):
        sess = str(tmp_path / "s.json")
        graph = str(tmp_path / "g.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "1": {
                        "class_type": "EmptyImage",
                        "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
                    }
                },
                fh,
            )
        self._run(["--json", "--dry-run", "--session", sess, "workflow", "convert", graph])
        assert not os.path.exists(sess), "--dry-run wrote the session anyway"

    def test_a_missing_node_type_fails_loudly(self, tmp_path):
        graph = str(tmp_path / "bad.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "nodes": [
                        {
                            "id": 1,
                            "type": "DefinitelyNotInstalled",
                            "widgets_values": [],
                            "inputs": [],
                        }
                    ],
                    "links": [],
                },
                fh,
            )
        proc = self._run(["workflow", "convert", graph], check=False)
        assert proc.returncode != 0
        assert "DefinitelyNotInstalled" in (proc.stdout + proc.stderr)


# ---------------------------------------------- refine round 2: iterate & repro


def test_internal_logs_are_readable(client):
    """Newer ComfyUI builds expose the server's own log lines."""
    res = client.logs(limit=20)
    entries = res.get("entries") if isinstance(res, dict) else res
    assert entries is not None, f"unexpected /internal/logs reply: {str(res)[:200]}"


def test_a_mask_upload_is_accepted_for_a_rendered_image(client, object_info, tmp_path):
    """The inpainting path: render, fetch the png, upload it back as a mask."""
    res = run_core.submit_and_wait(client, _minimal_graph(object_info), timeout=300)
    got = run_core.fetch(client, res["outputs"], str(tmp_path))
    src = got["files"][0]
    up = client.upload_mask(src["path"], src["filename"])
    assert up.get("name"), f"mask upload returned no name: {up}"
    print(f"\n  mask filed as {up.get('subfolder', '')}/{up['name']} for {src['filename']}")


class TestCLIRefineSubprocess:
    """The round-2 commands, through the installed CLI only."""

    CLI_BASE = _resolve_cli("cli-anything-comfyui")

    def _run(self, args, check=True):
        proc = subprocess.run(
            self.CLI_BASE + args, capture_output=True, text=True, timeout=600, check=False
        )
        if check:
            assert proc.returncode == 0, f"{args} -> {proc.returncode}\n{proc.stderr[-2000:]}"
        return proc

    def _seed_graph(self, tmp_path):
        sess = str(tmp_path / "s.json")
        graph = str(tmp_path / "g.json")
        with open(graph, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "1": {
                        "class_type": "EmptyImage",
                        "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
                    },
                    "2": {
                        "class_type": "SaveImage",
                        "inputs": {"images": ["1", 0], "filename_prefix": "cli_e2e_orig"},
                    },
                },
                fh,
            )
        self._run(["--json", "--session", sess, "workflow", "convert", graph])
        return sess, graph

    def test_server_logs(self):
        d = json.loads(self._run(["--json", "server", "logs"]).stdout)
        assert "entries" in d

    def test_set_export_diff_unset_roundtrip(self, tmp_path):
        """set -> export -> diff -> unset.

        NOT "ending where it started": `unset` removes the override, it does
        not restore the value `set` overwrote. The original version of this
        test asserted the graph came back identical and failed against a real
        server on 2026-09-06 — the implementation and the test were written in
        the same pass and contradicted each other.
        """
        sess, graph = self._seed_graph(tmp_path)
        self._run(
            [
                "--json",
                "--session",
                sess,
                "workflow",
                "set",
                "2",
                "filename_prefix",
                "cli_e2e_patched",
            ]
        )
        exported = str(tmp_path / "patched.json")
        d = json.loads(
            self._run(["--json", "--session", sess, "workflow", "export", "-o", exported]).stdout
        )
        assert d["nodes"] == 2 and os.path.exists(exported)
        diff = json.loads(
            self._run(["--json", "--session", sess, "workflow", "diff", graph]).stdout
        )
        assert diff["changed"][0]["changes"] == [
            {"input": "filename_prefix", "from": "cli_e2e_orig", "to": "cli_e2e_patched"}
        ]
        self._run(["--json", "--session", sess, "workflow", "unset", "2", "filename_prefix"])
        diff2 = json.loads(
            self._run(["--json", "--session", sess, "workflow", "diff", graph]).stdout
        )
        # NOT a round trip. `unset` REMOVES the override; it does not put back the
        # value `set` overwrote, and workflow.unset_input says so. The graph
        # therefore still differs from the original — by the ABSENCE of the
        # input rather than by its new value.
        assert diff2["same"] is False, "unset must not silently restore a prior value"
        after = json.loads(
            self._run(
                [
                    "--json",
                    "--session",
                    sess,
                    "workflow",
                    "export",
                    "-o",
                    str(tmp_path / "unset.json"),
                ]
            ).stdout
        )
        assert after["nodes"] == 2

    def test_unset_on_an_unknown_input_fails_loudly(self, tmp_path):
        sess, _ = self._seed_graph(tmp_path)
        proc = self._run(
            ["--json", "--session", sess, "workflow", "unset", "1", "nope"], check=False
        )
        assert proc.returncode != 0 and "no input 'nope'" in (proc.stdout + proc.stderr)


def test_userdata_round_trip_against_the_live_server(client):
    """The /userdata API: save, read back, list, delete a converted graph."""
    p = f"cli-anything-e2e/userdata-roundtrip-{os.getpid()}.json"
    try:
        res = client.userdata_put(p, {"ok": True, "from": "e2e"})
        assert res is not None
        blob = client.userdata_get(p)
        assert json.loads(blob)["ok"] is True, "the bytes round-trip verbatim"
        listed = client.userdata_list("cli-anything-e2e")
        assert any("userdata-roundtrip" in str(x) for x in listed)
    finally:
        client.userdata_delete(p)
    with pytest.raises(Exception):
        client.userdata_get(p), "deleted means deleted"


def test_history_clear_prunes_the_finished_prompts(client, object_info):
    """POST /history: run once, prune that id, then wipe — and history empties."""
    graph = _minimal_graph(object_info)
    res = run_core.submit_and_wait(client, graph, timeout=300)
    assert (client.history(res["prompt_id"]) or {}).get(res["prompt_id"])
    client.history_delete([res["prompt_id"]])
    assert not (client.history(res["prompt_id"]) or {}).get(res["prompt_id"]), (
        "the pruned prompt is gone from history"
    )
    client.history_delete()
    assert client.history() in ({}, None), "a full clear leaves no entries"


def test_upload_without_overwrite_renames_rather_than_refusing(client, tmp_path):
    """`overwrite=false` does NOT error — ComfyUI de-duplicates the NAME.

    The server walks `name (1).png`, `name (2).png` until the path is free, and
    before each step compares the image HASH: identical bytes are treated as a
    duplicate, so it keeps the existing name and writes nothing at all
    (server.py, the `compare_image_hash` branch added for issue #3465).

    An earlier version of this test asserted HTTP 409. ComfyUI has no such
    response, and asserting it made the harness look like it had a behaviour the
    software does not — caught against the real server on 2026-09-06.
    """
    f = tmp_path / "cli-anything-e2e.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    first = client.upload_image(str(f), subfolder="cli-anything-e2e")
    assert first.get("name")
    again = client.upload_image(str(f), subfolder="cli-anything-e2e", overwrite=False)
    assert again.get("name"), "the second upload must still answer with a name"
    # Same bytes -> the hash check short-circuits and the original name stands.
    assert again["name"] == first["name"], (
        "identical bytes should hit the duplicate check, not create a copy"
    )

    # Different bytes under the same name DO get renamed rather than refused.
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"1" * 64)
    third = client.upload_image(str(f), subfolder="cli-anything-e2e", overwrite=False)
    assert third.get("name") != first["name"], "different bytes must not overwrite"
    print(
        f"\n  no-overwrite: {first['name']} -> same for identical bytes, "
        f"{third['name']} for different"
    )
