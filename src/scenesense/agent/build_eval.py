"""Build the held-out agent eval set: data/agent_eval/questions.jsonl.

Ground truth for count / total / comparison / moment questions is computed here from the tracking logs and
scene metadata, never written by hand. VQA items use the tracker as reference at frames that were also
checked by eye (see VQA_ITEMS). The 7 dev questions used while building the agent are excluded.
"""
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from scenesense.agent.index import WINDOW, scene_meta
from scenesense.utils import read_jsonl

OUT = Path("data/agent_eval/questions.jsonl")
UPSTREAM_OUT = Path("data/agent_eval/upstream.jsonl")
PLURAL = {"car": "cars", "truck": "trucks", "bus": "buses", "trailer": "trailers",
          "construction_vehicle": "construction vehicles", "motorcycle": "motorcycles", "bicycle": "bicycles",
          "pedestrian": "pedestrians", "traffic_cone": "traffic cones", "barrier": "barriers"}
PLACE = {"boston-seaport": "Boston", "singapore-onenorth": "One North",
         "singapore-queenstown": "Queenstown", "singapore-hollandvillage": "Holland Village"}

# Dev questions (used while building the agent); none of these may appear in the eval set.
DEV = {"How many buses appear in the Queenstown scenes?",
       "Find a night moment at a bus stop and tell me if there is a pedestrian in that frame.",
       "Which Boston scene has the most pedestrians crossing?",
       "How many trees are in scene-0061?", "How many cars are in the Tokyo scenes?",
       "Is the bus to the left of the car in scene-0796 at 5 seconds, and what colour is it?",
       "Where is the bus in scene-0796 at 14 seconds?"}

# (scene, t, class, expected) - tracker reference, each frame checked by eye on 2026-11-05.
# Dropped after the visual check: scene-0061 t=3 truck (not clearly visible), scene-0553 t=8 truck (a shuttle bus).
VQA_ITEMS = [("scene-0061", 0, "truck", "yes"), ("scene-0103", 6, "pedestrian", "yes"),
             ("scene-0655", 12, "bus", "yes"), ("scene-0553", 8, "pedestrian", "yes"),
             ("scene-0757", 5, "bus", "yes"), ("scene-0103", 10, "bus", "no"),
             ("scene-1100", 8, "truck", "no"), ("scene-0757", 4, "construction_vehicle", "no")]
GROUND_ITEMS = [("scene-0757", 5, "bus")]

# Moment questions: text + a predicate over windows (description keyword, location, night, class present).
MOMENTS = [
    ("Find a moment in a parking lot where a motorcycle is visible.", "parking lot", None, None, "motorcycle"),
    ("Find a night moment where a bicycle is visible.", "", None, True, "bicycle"),
    ("Find a moment at a Boston intersection with a bus in view.", "intersection", "boston-seaport", None, "bus"),
    ("Find a moment at night after rain where a truck is visible.", "after rain", None, True, "truck"),
    ("Find a moment where a trailer appears.", "", None, None, "trailer"),
    ("Find a daytime moment in Queenstown with a motorcycle.", "", "singapore-queenstown", False, "motorcycle"),
    ("Find a moment in a Boston parking lot with a bus.", "parking lot", "boston-seaport", None, "bus"),
    ("Find a moment near construction in One North where a traffic cone is visible.", "construction",
     "singapore-onenorth", None, "traffic_cone"),
]

# Known upstream failures: the scene description says the object is there, but the detector/tracker never
# found it, so no tool can confirm it. Kept as a separate set (not dropped) so the main set is not made easier.
# (question, scene whose description mentions it, class, description phrase)
UPSTREAM = [
    ("Find a moment in a parking lot where a bicycle is visible.", "scene-0916", "bicycle", "parked bicycles"),
    ("Find a night moment where a construction vehicle is present.", "scene-1077", "construction_vehicle",
     "construction vehicle"),
]

UNANSWERABLE = [
    ("How many cars are in the Paris scenes?", "place"),
    ("How many dogs are in scene-0916?", "class"),
    ("How many traffic lights are in scene-0553?", "class"),
    ("What is the average speed of the cars in scene-1077?", "unmeasurable"),
    ("How many pedestrians crossed the street in scene-1100?", "unmeasurable"),
    ("What colour is the bus in scene-0655?", "unmeasurable"),
    ("Was it raining in scene-0061?", "unmeasurable"),
    ("How many buses are in scene-0796 at 45 seconds?", "time"),
    ("How many pedestrians are in scene-0420?", "scene"),
]


def load_events():
    ev = defaultdict(list)
    for p in sorted(glob.glob("data/tracks/scene-*.jsonl")):
        for e in read_jsonl(p):
            ev[e["scene"]].append(e)
    return ev


def n_tracks(events, cls, t0=0.0, t1=1e9):
    return len({e["track_id"] for e in events if e["class"] == cls and t0 <= e["t"] <= t1})


def windows(ev, meta):
    """(scene, t_start, t_end, classes present) for every 2-s window, as in the Chroma index."""
    out = defaultdict(set)
    for s, events in ev.items():
        for e in events:
            out[(s, int(e["t"] // WINDOW))].add(e["class"])
    return [{"scene": s, "t_start": w * WINDOW, "t_end": (w + 1) * WINDOW, "classes": sorted(c),
             "location": meta[s]["location"], "night": meta[s]["night"],
             "description": meta[s]["description"].lower()} for (s, w), c in sorted(out.items())]


def build(seed=0):
    rng = random.Random(seed)
    ev, meta = load_events(), scene_meta()
    scenes = sorted(ev)
    qs = []

    # 1. Single-scene counts (10): random (scene, class) with a nonzero count, plus 2 zeros and 3 time ranges.
    pairs = [(s, c) for s in scenes for c in PLURAL if n_tracks(ev[s], c) > 0]
    zeros = [(s, c) for s in scenes for c in ("bus", "trailer", "construction_vehicle") if n_tracks(ev[s], c) == 0]
    for s, c in rng.sample(pairs, 5) + rng.sample(zeros, 2):
        qs.append({"type": "count", "question": f"How many {PLURAL[c]} are tracked in {s}?",
                   "answer": n_tracks(ev[s], c), "scenes": [s], "t_range": None})
    for s, c in rng.sample(pairs, 3):
        t0 = rng.choice([0, 5, 10]); t1 = t0 + 5
        qs.append({"type": "count", "question": f"How many {PLURAL[c]} appear in {s} between {t0} and {t1} seconds?",
                   "answer": n_tracks(ev[s], c, t0, t1), "scenes": [s], "t_range": [t0, t1]})

    # 2. Totals over a set of scenes (5).
    groups = [("location", loc, f"the {PLACE[loc]} scenes") for loc in PLACE] + \
             [("night", True, "the night scenes"), ("night", False, "the daytime scenes")]
    for key, val, phrase in rng.sample(groups, 5):
        c = rng.choice(["car", "pedestrian", "truck", "traffic_cone", "barrier"])
        ss = [s for s in scenes if meta[s][key] == val]
        q = f"How many {PLURAL[c]} are tracked in total across {phrase}?"
        if q in DEV:
            continue
        qs.append({"type": "total", "question": q, "answer": sum(n_tracks(ev[s], c) for s in ss), "scenes": ss})

    # 3. Comparisons (10): argmax/argmin over a group with a unique winner, and pairwise with margin >= 3.
    comp = []
    for key, val, phrase in groups:
        ss = [s for s in scenes if meta[s][key] == val]
        if len(ss) < 2:
            continue
        for c in ("car", "pedestrian", "truck", "traffic_cone", "barrier", "bus"):
            counts = {s: n_tracks(ev[s], c) for s in ss}
            for word, fn in (("most", max), ("fewest", min)):
                best = fn(counts.values())
                winners = [s for s, v in counts.items() if v == best]
                if len(winners) == 1:
                    comp.append({"type": "compare",
                                 "question": f"Which of {phrase} has the {word} {PLURAL[c]}?",
                                 "answer": winners[0], "scenes": ss, "counts": counts})
    comp = [q for q in comp if q["question"] not in DEV]
    qs += rng.sample(comp, 6)
    pair_qs = []
    for _ in range(200):
        a, b = rng.sample(scenes, 2)
        c = rng.choice(["car", "pedestrian", "truck", "traffic_cone", "barrier"])
        na, nb = n_tracks(ev[a], c), n_tracks(ev[b], c)
        if abs(na - nb) >= 3:
            pair_qs.append({"type": "compare", "question": f"Are more {PLURAL[c]} tracked in {a} or in {b}?",
                            "answer": a if na > nb else b, "scenes": [a, b], "counts": {a: na, b: nb}})
        if len(pair_qs) == 4:
            break
    qs += pair_qs

    # 4. Find a moment (8): acceptable windows computed from metadata.
    wins = windows(ev, meta)
    for q, kw, loc, night, cls in MOMENTS:
        ok = [w for w in wins if kw in w["description"] and (loc is None or w["location"] == loc)
              and (night is None or w["night"] == night) and cls in w["classes"]]
        assert ok, q
        qs.append({"type": "moment", "question": q,
                   "answer": [[w["scene"], w["t_start"], w["t_end"]] for w in ok],
                   "scenes": sorted({w["scene"] for w in ok})})

    # 5. VQA on a given frame (9).
    for s, t, c, ans in VQA_ITEMS:
        name = PLURAL[c][:-2] if c == "bus" else PLURAL[c][:-1]
        qs.append({"type": "vqa", "question": f"In {s} at {t} seconds, is there a {name} in the front camera image?",
                   "answer": ans, "scenes": [s], "t": t, "reference": "tracker, checked by eye"})
    for s, t, c in GROUND_ITEMS:
        near = [e for e in ev[s] if e["class"] == c and abs(e["t"] - t) < 0.05]
        assert len(near) == 1, (s, t, c, len(near))
        qs.append({"type": "ground", "question": f"Where is the bus in {s} at {t} seconds? Give its bounding box.",
                   "answer": near[0]["box"], "scenes": [s], "t": t, "reference": "tracker box, checked by eye"})

    # 6. Unanswerable (9).
    for q, why in UNANSWERABLE:
        qs.append({"type": "unanswerable", "question": q, "answer": None, "reason": why, "scenes": []})

    assert not {q["question"] for q in qs} & DEV
    assert len({q["question"] for q in qs}) == len(qs), "duplicate questions"
    for i, q in enumerate(qs):
        q["id"] = i
    return qs


def build_upstream():
    """Truth from the scene description (any window of that scene); the tracker has zero tracks of the class."""
    ev, meta = load_events(), scene_meta()
    out = []
    for i, (q, scene, cls, phrase) in enumerate(UPSTREAM):
        assert phrase in meta[scene]["description"].lower(), (scene, phrase)
        assert not any(e["class"] == cls for e in ev[scene]), (scene, cls)
        n_win = int(max(e["t"] for e in ev[scene]) // WINDOW) + 1
        out.append({"id": 100 + i, "type": "upstream", "question": q, "scenes": [scene],
                    "answer": [[scene, w * WINDOW, (w + 1) * WINDOW] for w in range(n_win)],
                    "reason": f"description of {scene} says '{phrase}' but the tracker has 0 {cls} tracks"})
    return out


def main():
    up = build_upstream()
    UPSTREAM_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(UPSTREAM_OUT, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in up)
    print(f"{len(up)} upstream-failure questions -> {UPSTREAM_OUT}")
    qs = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in qs)
    by = defaultdict(int)
    for q in qs:
        by[q["type"]] += 1
    print(f"{len(qs)} questions -> {OUT}: {dict(by)}")


if __name__ == "__main__":
    main()
