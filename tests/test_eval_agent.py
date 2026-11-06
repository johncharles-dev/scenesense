"""Rule-based agent scorer: parsing and per-type correctness on hand-written answers."""
from scenesense.agent.eval_agent import (
    bootstrap,
    citation_ok,
    citations,
    claims,
    correct,
    log_supported,
    numbers,
    question_class,
    yes_no,
)

COUNT = {"type": "count", "answer": 78, "scenes": ["scene-1077"]}
ZERO = {"type": "count", "answer": 0, "scenes": ["scene-0103"]}
TOTAL_T = {"type": "count", "answer": 3, "scenes": ["scene-1100"]}
COMPARE = {"type": "compare", "answer": "scene-1077", "scenes": ["scene-0796", "scene-1077"]}
MOMENT = {"type": "moment", "answer": [["scene-1094", 16.0, 18.0]], "scenes": ["scene-1094"]}
VQA = {"type": "vqa", "answer": "no", "scenes": ["scene-0103"], "t": 10}
GROUND = {"type": "ground", "answer": [653.1, 361.0, 1231.2, 537.6], "scenes": ["scene-0757"], "t": 5}
UNA = {"type": "unanswerable", "answer": None, "scenes": []}


def test_numbers_ignore_scene_ids_and_times():
    assert numbers("There are 78 cars tracked in scene-1077 (scene-1077, 0-19.95 s).") == [78]
    assert numbers("3 pedestrians appear between 5 and 10 seconds (scene-1100, 5-10 s).") == [3]
    assert numbers("Box [1, 2, 3, 4] in a 1600x900 image") == []


def test_count_and_zero():
    assert correct(COUNT, "78 cars are tracked (scene-1077, 0-19.95 s).")
    assert not correct(COUNT, "About 80 cars (scene-1077, 0-19.95 s).")
    assert correct(ZERO, "No buses are tracked in scene-0103 (scene-0103, 0-19.4 s).")
    assert correct(TOTAL_T, "There are 3 pedestrians between 5 and 10 seconds (scene-1100, 5-10 s).")


def test_compare_uses_first_candidate_named():
    assert correct(COMPARE, "More barriers in scene-1077 (10) than scene-0796 (0).")
    assert not correct(COMPARE, "scene-0796 has more barriers than scene-1077.")
    assert correct(COMPARE, "In scene-0796, no barriers are tracked, whereas scene-1077 has 10. "
                            "Therefore, more barriers are tracked in scene-1077.")


def test_moment_needs_scene_and_overlapping_time():
    assert correct(MOMENT, "A truck is visible at night after rain (scene-1094, 16-18 s).")
    assert correct(MOMENT, "See scene-1094 at 17 s.")
    assert not correct(MOMENT, "See scene-1094.")                       # no time
    assert not correct(MOMENT, "See scene-1094 at 4 s.")                # wrong window


def test_vqa_and_ground():
    assert yes_no("No, there is no bus in the image (scene-0103, 10 s).") == "no"
    assert yes_no("Yes, there is a bus.") == "yes"
    assert correct(VQA, "There is no bus visible (scene-0103, 10 s).")
    assert correct(GROUND, "The bus is at [650, 360, 1230, 540] (scene-0757, 5 s).")
    assert not correct(GROUND, "The bus is at [100, 100, 200, 200].")


def test_refusal():
    assert correct(UNA, "There is no data for Paris; the scenes are in Singapore and Boston.")
    assert correct(UNA, "Speed cannot be measured by the tools.")
    assert correct(UNA, "The tools do not measure weather conditions like rain.")
    assert not correct(UNA, "There are 143 cars.")


def test_citations():
    assert citations("At 12 s in scene-1077 there is a bus.") == [("scene-1077", 12.0, 12.0)]
    assert citation_ok(VQA, "No bus (scene-0103, 10 s).") == (True, True)
    assert citation_ok(VQA, "No bus (scene-0103, 15 s).") == (True, False)
    assert citation_ok(COUNT, "78 cars (scene-0061, 0-19 s).") == (True, False)
    assert citation_ok(COUNT, "78 cars.") == (False, False)


def test_bootstrap_mean_and_ratio():
    r = bootstrap([1, 0, 1, 1])
    assert r["mean"] == 0.75 and r["lo"] <= 0.75 <= r["hi"]
    assert bootstrap([2, 3], weights=[2, 4])["mean"] == 5 / 6


def test_claims_split_polarity_per_scene():
    c = claims("In scene-0796, no barriers are tracked (scene-0796, 0-19.6 s), whereas in scene-1077, "
               "10 barriers are tracked (scene-1077, 0-19.95 s).")
    timed = [(sc, present) for sc, a, _, present in c if a is not None]
    assert timed == [("scene-0796", False), ("scene-1077", True)]
    assert question_class("Are more traffic cones tracked in scene-0103?") == "traffic_cone"
    assert question_class("How many construction vehicles are tracked?") == "construction_vehicle"


def test_log_supported():
    events = {"scene-0001": [{"class": "bus", "t": 5.0}], "scene-0002": []}
    q = {"type": "vqa", "question": "Is there a bus in scene-0001 at 5 seconds?"}
    assert log_supported(q, "Yes, there is a bus (scene-0001, 5 s).", events)
    assert log_supported(q, "There is a bus at 5.8 s in scene-0001.", events)        # within +-1 s
    assert not log_supported(q, "There is a bus (scene-0001, 9 s).", events)         # no bus near 9 s
    assert not log_supported(q, "There is no bus (scene-0001, 5 s).", events)        # contradicts log
    assert log_supported(q, "There is no bus (scene-0002, 5 s).", events)
    assert not log_supported(q, "There is a bus in scene-0001.", events)              # no time: unverifiable


def test_claims_repeated_and_bare_citations_keep_polarity():
    c = claims("There are 0 buses tracked in scene-0103 (scene-0103, 0-19.4 s).")
    assert [p for _, a, _, p in c if a is not None] == [False, False]
    c = claims("No construction vehicles appear in scene-0757 between 5 and 10 seconds. (scene-0757, 5-10 s)")
    assert [p for _, a, _, p in c if a is not None] == [False, False]      # the bare citation inherits "no"
