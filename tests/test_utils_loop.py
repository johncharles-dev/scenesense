"""JSON helpers and the agent loop's tool dispatch (no LLM server needed)."""
from scenesense.agent import loop
from scenesense.utils import read_json, read_jsonl, write_json, write_jsonl


def test_jsonl_round_trip_skips_blank_lines(tmp_path):
    p = tmp_path / "sub" / "x.jsonl"
    rows = [{"a": 1}, {"b": [1, 2]}]
    write_jsonl(rows, p)
    assert read_jsonl(p) == rows
    p.write_text(p.read_text() + "\n\n")
    assert read_jsonl(p) == rows


def test_json_round_trip(tmp_path):
    p = tmp_path / "x.json"
    write_json({"k": "v"}, p, indent=2)
    assert read_json(p) == {"k": "v"}


def test_call_tool_unknown_tool():
    res, ok = loop.call_tool("delete_everything", "{}")
    assert not ok and "Unknown tool" in res["error"]


def test_call_tool_bad_json_and_non_object():
    res, ok = loop.call_tool("count", "{not json")
    assert not ok and "Could not parse" in res["error"]
    res, ok = loop.call_tool("count", "[1, 2]")
    assert not ok and "JSON object" in res["error"]


def test_call_tool_wrong_argument_names():
    res, ok = loop.call_tool("count", '{"klass": "bus"}')
    assert not ok and res["error"].startswith("Bad arguments for count")


def test_call_tool_validation_error_is_returned_not_raised(monkeypatch):
    def boom(**kw):
        raise ValueError("Unknown scene 'scene-9999'.")
    monkeypatch.setitem(loop.FUNCS, "count", boom)
    res, ok = loop.call_tool("count", '{"object_class": "bus", "scene": "scene-9999"}')
    assert not ok and res == {"error": "ValueError: Unknown scene 'scene-9999'."}


def test_call_tool_success(monkeypatch):
    monkeypatch.setitem(loop.FUNCS, "count", lambda **kw: {"count": 2, **kw})
    res, ok = loop.call_tool("count", '{"object_class": "bus"}')
    assert ok and res == {"count": 2, "object_class": "bus"}


def test_system_prompt_lists_scenes_and_classes():
    s = loop.system_prompt()
    assert "scene-0796" in s and "traffic_cone" in s
