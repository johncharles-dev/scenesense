"""COCO mAP with bootstrap 95% confidence intervals."""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
import numpy as np
import copy
import json
from pathlib import Path
from tqdm import tqdm

def _quiet(fn):
    """Run fn() with COCOeval's printing suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _evaluate_once(gt, dt, img_ids):
    ev = COCOeval(gt, dt, iouType="bbox")
    ev.params.imgIds = img_ids
    _quiet(ev.evaluate)
    return ev


def _map_on_subset(ev, img_ids):
    """mAP@[.5:.95] (area=all, maxDets=100) over a (resampled) list of image ids."""
    p = ev.params
    pos = {iid: i for i, iid in enumerate(p.imgIds)}
    idx = [pos[i] for i in img_ids]
    n_img, n_area = len(p.imgIds), len(p.areaRng)
    a_all = p.areaRngLbl.index("all")
    sub = COCOeval(ev.cocoGt, ev.cocoDt, iouType="bbox")
    sub.params = copy.deepcopy(p)
    sub.params.imgIds = list(img_ids)
    sub.params.areaRng = [p.areaRng[a_all]]
    sub.params.areaRngLbl = ["all"]
    sub.params.maxDets = [p.maxDets[-1]]
    sub.evalImgs = []
    for c in range(len(p.catIds)):
        base = c * n_area * n_img + a_all * n_img
        sub.evalImgs.extend(ev.evalImgs[base + i] for i in idx)
    sub._paramsEval = sub.params
    _quiet(sub.accumulate)
    prec = sub.eval["precision"][..., 0, 0]      # [IoU, recall, class] for area=all, maxDets=100
    valid = prec[prec > -1]
    return float(valid.mean()) if valid.size else -1.0


def bootstrap_map(gt_path, pred_paths, n_boot=500, seed=0, img_ids=None):
    """Paired bootstrap: returns {name: (point, lo, hi, samples)} plus 'A-B' difference entries."""
    if isinstance(pred_paths, str):
        pred_paths = [pred_paths]
    gt = _quiet(lambda: COCO(gt_path))
    ids = sorted(img_ids if img_ids is not None else gt.getImgIds())
    evs = {}
    for p in tqdm(pred_paths, desc="evaluate"):
        name = Path(p).stem
        dt = _quiet(lambda: gt.loadRes(p))
        evs[name] = _evaluate_once(gt, dt, ids)
    rng = np.random.default_rng(seed)
    samples = {k: np.empty(n_boot) for k in evs}
    for b in tqdm(range(n_boot), desc="bootstrap"):
        draw = list(rng.choice(ids, size=len(ids), replace=True))   # same draw for every model
        for k, ev in evs.items():
            samples[k][b] = _map_on_subset(ev, draw)
    out = {}
    for k, ev in evs.items():
        lo, hi = np.percentile(samples[k], [2.5, 97.5])
        out[k] = (_map_on_subset(ev, ids), float(lo), float(hi), samples[k])
    names = list(evs)
    for other in names[1:]:
        d = samples[names[0]] - samples[other]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[f"{names[0]}-{other}"] = (out[names[0]][0] - out[other][0], float(lo), float(hi), d)
    return out


def per_class_ap(gt_path, pred_path, img_ids=None):
    """AP@[.5:.95] per class plus AP by object size (area=all/small/medium/large, maxDets=100)."""
    gt = _quiet(lambda: COCO(gt_path))
    dt = _quiet(lambda: gt.loadRes(pred_path))
    ev = COCOeval(gt, dt, iouType="bbox")
    ev.params.imgIds = sorted(img_ids if img_ids is not None else gt.getImgIds())
    _quiet(lambda: (ev.evaluate(), ev.accumulate(), ev.summarize()))
    prec = ev.eval["precision"][:, :, :, 0, -1]          # [IoU, recall, class] at area=all, maxDets=100
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        v = prec[:, :, k]
        v = v[v > -1]
        per_class[gt.cats[int(cid)]["name"]] = float(v.mean()) if v.size else float("nan")
    sizes = {"small": float(ev.stats[3]), "medium": float(ev.stats[4]), "large": float(ev.stats[5])}
    return per_class, sizes

def select_images(gt_path, slice_expr):
    """'location=singapore-onenorth' -> list of image ids with that field value."""
    if not slice_expr:
        return None
    key, val = slice_expr.split("=", 1)
    data = json.loads(Path(gt_path).read_text())
    return [im["id"] for im in data["images"] if str(im.get(key)) == val]

def coco_map(gt_path: str, pred_path: str) -> float:
    gt = COCO(gt_path)
    dt = gt.loadRes(pred_path)
    ev = COCOeval(gt, dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])      # stats[0] = mAP@[.5:.95]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="COCO mAP with paired bootstrap 95% CIs.")
    ap.add_argument("--per-class", action="store_true", help="print per-class and per-size AP, no bootstrap")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", action="append", required=True, help="repeat for several models")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--slice", default=None, help="e.g. location=singapore-onenorth")
    ap.add_argument("--out", default=None, help="write results JSON")
    a = ap.parse_args()
    if a.per_class:
        ids = select_images(a.gt, a.slice)
        rows = {}
        for p in a.pred:
            pc, sz = per_class_ap(a.gt, p, ids)
            rows[Path(p).stem] = {"per_class": pc, "size": sz}
        names = list(rows)
        print(f"{'class':22s}" + "".join(f"{n[:18]:>20s}" for n in names))
        for c in rows[names[0]]["per_class"]:
            print(f"{c:22s}" + "".join(f"{rows[n]['per_class'][c]:20.3f}" for n in names))
        for s in ("small", "medium", "large"):
            print(f"{'AP ' + s:22s}" + "".join(f"{rows[n]['size'][s]:20.3f}" for n in names))
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=2))
        raise SystemExit
    res = bootstrap_map(a.gt, a.pred, a.n_boot, img_ids=select_images(a.gt, a.slice))
    rows = {}
    for k, (point, lo, hi, _) in res.items():
        print(f"{k:28s} {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        rows[k] = {"map": point, "ci95": [lo, hi]}
    if a.out:
        Path(a.out).write_text(json.dumps({"slice": a.slice, "n_boot": a.n_boot, "results": rows}, indent=2))