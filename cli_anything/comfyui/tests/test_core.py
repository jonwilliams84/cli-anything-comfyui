"""Unit tests — synthetic graphs, no server.

The centre of gravity is `to_api`. Everything else in this harness is a thin
wrapper over an HTTP call; the converter is the part that encodes knowledge, and
it is the part four hand-rolled `wf2api` scripts in `~/ai-video` got wrong.
"""

from __future__ import annotations

import json
import threading

import pytest

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
