"""Agent tool tests that run without the vLLM servers or the nuScenes images."""
import json
from pathlib import Path

import pytest

from scenesense.agent import tools
from scenesense.utils import read_jsonl


def _ev(track_id, cls, t, scene="scene-0001"):
    return {"scene": scene, "location": "x", "frame": int(t * 12), "t": t,
            "track_id": track_id, "class": cls, "score": 0.9, "box": [0, 0, 1, 1]}


@pytest.fixture
def fake_logs(tmp_path, monkeypatch):
    events = [_ev(1, "car", 0.0), _ev(1, "car", 0.5), _ev(1, "car", 3.0),     # one car, spans 0-3 s
              _ev(2, "car", 5.0), _ev(2, "car", 6.0),                        # second car, 5-6 s
              _ev(3, "pedestrian", 1.0), _ev(4, "pedestrian", 1.5), _ev(5, "bus", 9.0)]
    with open(tmp_path / "scene-0001.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    (tmp_path / "scene-0002.jsonl").write_text(json.dumps(_ev(1, "bus", 0.0, "scene-0002")) + "\n")
    monkeypatch.setattr(tools, "TRACKS_DIR", tmp_path)
    tools._events.cache_clear()
    yield tmp_path
    tools._events.cache_clear()


def test_count_distinct_track_ids(fake_logs):
    assert tools.count("car", scene="scene-0001")["count"] == 2          # 5 car events, 2 IDs
    assert tools.count("pedestrian", scene="scene-0001")["count"] == 2
    assert tools.count("bicycle", scene="scene-0001")["count"] == 0
    assert "ID switches" in tools.count("car", scene="scene-0001")["note"]


def test_count_time_range_is_inclusive_and_by_presence(fake_logs):
    assert tools.count("car", scene="scene-0001", t_start=0, t_end=2)["count"] == 1    # only track 1
    assert tools.count("car", scene="scene-0001", t_start=3, t_end=5)["count"] == 2    # both endpoints included
    assert tools.count("car", scene="scene-0001", t_start=7, t_end=20)["count"] == 0


def test_count_normalises_names(fake_logs):
    assert tools.count("cars", scene="1")["count"] == 2                  # "1" -> scene-0001, cars -> car
    assert tools.count("people", scene="SCENE-0001")["count"] == 2
    assert tools.count("Buses", scene="scene-0001")["count"] == 1


def test_count_rejects_bad_arguments(fake_logs):
    with pytest.raises(ValueError, match="Unknown scene .*scene-0001, scene-0002"):
        tools.count("car", scene="queenstown")
    with pytest.raises(ValueError, match="Unknown object class 'tree'"):
        tools.count("tree", scene="scene-0001")
    with pytest.raises(ValueError, match="before t_start"):
        tools.count("car", scene="scene-0001", t_start=5, t_end=1)
    with pytest.raises(ValueError, match="outside scene-0001"):
        tools.count("car", scene="scene-0001", t_start=45, t_end=50)
    with pytest.raises(ValueError, match="not both"):
        tools.count("car", scene="scene-0001", location="loc-a")


def test_count_over_scenes_gives_breakdown_total_and_argmax(fake_logs, monkeypatch):
    meta = {"scene-0001": {"location": "loc-a", "night": False}, "scene-0002": {"location": "loc-b", "night": True}}
    monkeypatch.setattr(tools, "_meta", lambda: meta)
    r = tools.count("bus")                                          # all scenes
    assert [c["count"] for c in r["per_scene"]] == [1, 1] and r["total"] == 2
    assert r["most"] == ["scene-0001", "scene-0002"]                # ties are all reported
    r = tools.count("car", location="loc-a")
    assert r["total"] == 2 and r["most"] == ["scene-0001"]
    assert tools.count("bus", night=True)["per_scene"][0]["scene"] == "scene-0002"
    with pytest.raises(ValueError, match="Unknown location"):
        tools.count("car", location="queenstown")


@pytest.mark.parametrize("q,fam", [
    ("How many pedestrians are in the image?", "count"),
    ("Is there a bus in the image?", "exists"),
    ("Is there an  traffic cone in the image?", "exists"),
    ("Is the truck to the left or right of the car?", "left_right"),
    ("Where is the bus?", "ground"),
    ("Where is the leftmost car? Give its bounding box.", "ground"),
])
def test_question_family_accepts_trained_forms(q, fam):
    assert tools.question_family(q)[0] == fam


@pytest.mark.parametrize("q", ["Describe the scene.", "Is the pedestrian crossing the road?",
                               "What colour is the bus?", "Are there pedestrians?"])
def test_question_family_rejects_other_forms(q):
    with pytest.raises(ValueError, match="Unsupported VLM question"):
        tools.question_family(q)


def test_question_family_rejects_untrained_classes():
    with pytest.raises(ValueError, match="only trained on"):
        tools.question_family("How many trees are in the image?")
    with pytest.raises(ValueError, match="only trained on"):
        tools.question_family("Is the dog to the left or right of the car?")


@pytest.mark.skipif(not Path("data/vqa/val.jsonl").exists(), reason="VQA set not present")
def test_question_family_accepts_every_training_template():
    from scenesense.vqa.format import family
    rows = read_jsonl("data/vqa/val.jsonl")
    for r in rows:
        assert tools.question_family(r["question"])[0] == family(r["type"]), r["question"]


def test_tool_schemas_match_functions():
    assert {t["function"]["name"] for t in tools.TOOLS} == set(tools.FUNCS)
    assert "ALWAYS" in next(t for t in tools.TOOLS if t["function"]["name"] == "count")["function"]["description"]


@pytest.mark.skipif(not Path("data/tracks/scene-0796.jsonl").exists(), reason="tracking logs not present")
def test_count_matches_raw_log():
    events = read_jsonl("data/tracks/scene-0796.jsonl")
    truth = len({e["track_id"] for e in events if e["class"] == "bus"})
    assert tools.count("bus", scene="scene-0796")["count"] == truth
