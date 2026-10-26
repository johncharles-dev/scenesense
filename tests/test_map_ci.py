import json
import numpy as np
from scenesense.eval.map_ci import coco_map, bootstrap_map, _evaluate_once, _map_on_subset, _quiet
from pycocotools.coco import COCO

GT, PRED = "data/coco/mini.json", "data/coco/pred_shifted.json"


def _materialise(ids, tmp_path):
    """Write a GT/pred pair where each drawn image is a fresh copy with a new id (repeats kept)."""
    gt, pred = json.load(open(GT)), json.load(open(PRED))
    imgs = {im["id"]: im for im in gt["images"]}
    anns, dets = {}, {}
    for a in gt["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    for d in pred:
        dets.setdefault(d["image_id"], []).append(d)
    new_imgs, new_anns, new_dets, aid = [], [], [], 1
    for new_id, old in enumerate(ids, start=1):
        new_imgs.append({**imgs[old], "id": new_id})
        for a in anns.get(old, []):
            new_anns.append({**a, "id": aid, "image_id": new_id}); aid += 1
        for d in dets.get(old, []):
            new_dets.append({**d, "image_id": new_id})
    g, p = tmp_path / "gt.json", tmp_path / "pred.json"
    g.write_text(json.dumps({"images": new_imgs, "annotations": new_anns, "categories": gt["categories"]}))
    p.write_text(json.dumps(new_dets))
    return str(g), str(p)


def test_fast_resample_matches_materialised(tmp_path):
    gt = _quiet(lambda: COCO(GT))
    dt = _quiet(lambda: gt.loadRes(PRED))
    ids = sorted(gt.getImgIds())
    ev = _evaluate_once(gt, dt, ids)
    rng = np.random.default_rng(0)
    for b in range(5):
        draw = [int(i) for i in rng.choice(ids, size=len(ids), replace=True)]
        g, p = _materialise(draw, tmp_path)
        assert abs(_map_on_subset(ev, draw) - coco_map(g, p)) < 1e-9, f"replicate {b}"


def test_perfect_is_one():
    point, lo, hi, _ = bootstrap_map(GT, "data/coco/pred_perfect.json", n_boot=20)["pred_perfect"]
    assert point == 1.0 and lo == 1.0 and hi == 1.0


def test_seed_is_deterministic():
    a = bootstrap_map(GT, PRED, n_boot=10, seed=1)["pred_shifted"][3]
    b = bootstrap_map(GT, PRED, n_boot=10, seed=1)["pred_shifted"][3]
    assert np.array_equal(a, b)

def test_paired_difference_excludes_zero():
    res = bootstrap_map(GT, ["data/coco/pred_perfect.json", PRED], n_boot=50)
    point, lo, hi, _ = res["pred_perfect-pred_shifted"]
    assert point > 0 and lo > 0