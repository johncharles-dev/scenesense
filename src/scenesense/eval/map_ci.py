"""COCO mAP with bootstrap 95% confidence intervals."""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
import numpy as np
import copy


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
    """Re-aggregate an evaluated COCOeval over a (resampled) list of image ids."""
    p = ev.params
    pos = {iid: i for i, iid in enumerate(p.imgIds)}
    idx = [pos[i] for i in img_ids]
    n_img, n_area = len(p.imgIds), len(p.areaRng)
    sub = COCOeval(ev.cocoGt, ev.cocoDt, iouType="bbox")
    sub.params = copy.deepcopy(p)
    sub.params.imgIds = list(img_ids)
    sub.evalImgs = []
    for c in range(len(p.catIds)):
        for a in range(n_area):
            base = c * n_area * n_img + a * n_img
            sub.evalImgs.extend(ev.evalImgs[base + i] for i in idx)
    sub._paramsEval = sub.params
    _quiet(lambda: (sub.accumulate(), sub.summarize()))
    return float(sub.stats[0])


def bootstrap_map(gt_path, pred_path, n_boot=500, seed=0):
    gt = _quiet(lambda: COCO(gt_path))
    dt = _quiet(lambda: gt.loadRes(pred_path))
    img_ids = sorted(gt.getImgIds())
    ev = _evaluate_once(gt, dt, img_ids)
    point = _map_on_subset(ev, img_ids)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        ids = list(rng.choice(img_ids, size=len(img_ids), replace=True))
        samples[b] = _map_on_subset(ev, ids)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi), samples

def coco_map(gt_path: str, pred_path: str) -> float:
    gt = COCO(gt_path)
    dt = gt.loadRes(pred_path)
    ev = COCOeval(gt, dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])      # stats[0] = mAP@[.5:.95]


if __name__ == "__main__":
    import sys
    point, lo, hi, _ = bootstrap_map(sys.argv[1], sys.argv[2], n_boot=100)
    print(f"mAP = {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")