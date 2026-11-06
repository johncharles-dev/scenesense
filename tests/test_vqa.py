"""VQA format, parsing, scoring and template generation; no model or data needed."""
import random

import pytest

from scenesense.vqa.build_vqa import candidates
from scenesense.vqa.eval_vlm import accuracy_ci, report
from scenesense.vqa.format import (
    family,
    iou,
    parse_box,
    prompt,
    score,
    target,
    to_qwen_box,
)

W, H = 1600, 900


@pytest.mark.parametrize(("qtype", "fam"), [("count", "count"), ("exists_yes", "exists"), ("exists_no", "exists"),
                                            ("ground", "ground"), ("ground_ref", "ground"),
                                            ("left_right", "left_right")])
def test_family(qtype, fam):
    assert family(qtype) == fam


def test_prompt_appends_answer_format():
    p = prompt({"type": "exists_no", "question": "Is there a bus in the image?"})
    assert p == "Is there a bus in the image? Answer yes or no."


def test_box_round_trip_within_grid_resolution():
    box = [400, 225, 800, 675]
    q = to_qwen_box(box, W, H)
    assert q == "(250,250),(500,750)"
    back = parse_box(q, W, H)
    assert all(abs(a - b) <= W / 1000 for a, b in zip(back, box, strict=True))


def test_parse_box_needs_four_numbers():
    assert parse_box("(250,250)", W, H) is None
    assert parse_box("no box here", W, H) is None


def test_target_for_grounding_has_ref_and_box_tokens():
    row = {"type": "ground_ref", "question": "Where is the leftmost car? Give its bounding box.",
           "box": [0, 0, 160, 90], "width": W, "height": H, "answer": "[0, 0, 160, 90]"}
    t = target(row)
    assert t == ("<|object_ref_start|>the leftmost car<|object_ref_end|>"
                 "<|box_start|>(0,0),(100,100)<|box_end|>")
    assert target({"type": "count", "answer": "3"}) == "3"


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0                 # degenerate: no division by zero


@pytest.mark.parametrize(("qtype", "answer", "text", "ok"), [
    ("count", "3", "3", True),
    ("count", "3", "There are three buses.", True),               # number words
    ("count", "3", "13", False),                                  # no partial digit match
    ("count", "2", "none", False),
    ("exists_yes", "yes", "Yes.", True),
    ("exists_no", "no", "No, there isn't.", True),
    ("exists_no", "no", "yes", False),
    ("left_right", "left", "It is to the left.", True),
    ("left_right", "left", "right", False),
])
def test_score_text_answers(qtype, answer, text, ok):
    assert score({"type": qtype, "answer": answer}, text) is ok


def test_score_grounding_uses_iou_half():
    row = {"type": "ground", "box": [400, 225, 800, 675], "width": W, "height": H}
    assert score(row, "(250,250),(500,750)")
    assert not score(row, "(500,250),(750,750)")                 # shifted by the box width: IoU 0
    assert not score(row, "I can't see it")


def test_accuracy_ci_brackets_mean_and_is_deterministic():
    flags = [1, 0, 1, 1, 0, 1, 1, 1]
    acc, lo, hi = accuracy_ci(flags)
    assert acc == 0.75 and lo <= acc <= hi
    assert accuracy_ci(flags) == (acc, lo, hi)


def test_report_groups_by_family_and_verified_subset():
    preds = [{"type": "exists_yes", "answer": "yes", "correct": True, "image_id": 1, "question": "a"},
             {"type": "exists_no", "answer": "no", "correct": False, "image_id": 2, "question": "b"},
             {"type": "ground", "answer": "[0]", "correct": True, "image_id": 3, "question": "c"}]
    s = report(preds, {(1, "a")}, "test")
    assert s["exists"]["n"] == 2 and s["exists"]["acc"] == 0.5
    assert s["ground"]["majority"] == 0.0
    assert s["all"]["n"] == 3
    assert s["verified"]["n"] == 1 and s["verified"]["acc"] == 1.0
    assert "count" not in s


def _ann(cat, x, y, w, h):
    return {"category_id": cat, "bbox": [x, y, w, h], "area": w * h}


def test_candidates_are_correct_by_construction():
    cats = {1: "car", 2: "bus", 3: "pedestrian"}
    anns = [_ann(1, 100, 400, 200, 100), _ann(1, 700, 400, 100, 80),   # two cars, both big
            _ann(2, 1200, 300, 300, 200),                               # one bus, on the right
            _ann(3, 50, 400, 10, 20)]                                   # one tiny pedestrian
    im = {"width": W, "height": H}
    out = candidates(im, anns, cats, random.Random(0), random.Random(0))
    by_type = {}
    for t, q, a, box in out:
        by_type.setdefault(t, []).append((q, a, box))

    counts = {q: a for q, a, _ in by_type["count"]}
    assert counts == {"How many cars are in the image?": "2", "How many buses are in the image?": "1"}
    assert ("Where is the bus? Give its bounding box.", "[1200, 300, 1500, 500]", [1200, 300, 1500, 500]) \
        in by_type["ground"]
    assert all("pedestrian" not in q for q, _, _ in by_type["ground"])  # too small to ground
    assert by_type["exists_yes"][0][1] == "yes"
    absent_q = by_type["exists_no"][0][0]
    assert not any(n in absent_q for n in ("car", "bus", "pedestrian"))  # truly absent class
    [(q, _, box)] = by_type["ground_ref"]                               # one referring question per class
    expected = {"leftmost": [100, 400, 300, 500], "rightmost": [700, 400, 800, 480], "largest": [100, 400, 300, 500]}
    assert box == expected[q.split()[3]]
    assert "left_right" not in by_type                                   # needs two unique classes
