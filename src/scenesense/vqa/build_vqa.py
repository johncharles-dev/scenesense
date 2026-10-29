"""Templated VQA from nuImages ground-truth boxes: answers are correct by construction."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MIN_AREA = 32 * 32                       # COCO 'medium' and above: big enough to see
NAMES = {"car": "car", "truck": "truck", "bus": "bus", "trailer": "trailer",
         "construction_vehicle": "construction vehicle", "motorcycle": "motorcycle",
         "bicycle": "bicycle", "pedestrian": "pedestrian", "traffic_cone": "traffic cone",
         "barrier": "barrier"}
PLURAL = {k: v + "s" for k, v in NAMES.items()} | {"bus": "buses"}


def candidates(im, anns, cats, rng, ref_rng):
    """All unambiguous questions for one image: (type, question, answer, box)."""
    by_cls = defaultdict(list)
    for a in anns:
        by_cls[cats[a["category_id"]]].append(a)
    big = {c: [a for a in v if a["area"] >= MIN_AREA] for c, v in by_cls.items()}
    out = []

    for c, v in by_cls.items():                                   # count: every instance clearly visible
        if len(v) == len(big[c]) and 1 <= len(v) <= 8:
            out.append(("count", f"How many {PLURAL[c]} are in the image?", str(len(v)), None))

    present = [c for c in big if big[c]]
    absent = [c for c in NAMES if c not in by_cls]                # absent entirely, not just small
    if present:
        c = rng.choice(present)
        out.append(("exists_yes", f"Is there a {NAMES[c]} in the image?", "yes", None))
    if absent:
        c = rng.choice(absent)
        out.append(("exists_no", f"Is there a {NAMES[c]} in the image?", "no", None))

    uniq = [c for c, v in by_cls.items() if len(v) == 1 and big[c]]
    for c in uniq:                                                # ground: exactly one, and big
        x, y, w, h = by_cls[c][0]["bbox"]
        box = [round(x), round(y), round(x + w), round(y + h)]
        out.append(("ground", f"Where is the {NAMES[c]}? Give its bounding box.", str(box), box))

    if len(uniq) >= 2:                                            # left/right between two unique objects
        a, b = rng.sample(uniq, 2)
        ax = by_cls[a][0]["bbox"][0] + by_cls[a][0]["bbox"][2] / 2
        bx = by_cls[b][0]["bbox"][0] + by_cls[b][0]["bbox"][2] / 2
        if abs(ax - bx) > 0.1 * im["width"]:
            out.append(("left_right", f"Is the {NAMES[a]} to the left or right of the {NAMES[b]}?",
                        "left" if ax < bx else "right", None))

    for c, v in by_cls.items():                                   # referring expressions: one of several
        if 2 <= len(v) <= 6 and len(big[c]) == len(v):            # every instance clearly visible
            cx = lambda a: a["bbox"][0] + a["bbox"][2] / 2
            by_x = sorted(v, key=cx)
            by_area = sorted(v, key=lambda a: a["area"])
            opts = []
            if cx(by_x[1]) - cx(by_x[0]) > 0.05 * im["width"]:
                opts.append(("leftmost", by_x[0]))
            if cx(by_x[-1]) - cx(by_x[-2]) > 0.05 * im["width"]:
                opts.append(("rightmost", by_x[-1]))
            if by_area[-1]["area"] > 1.5 * by_area[-2]["area"]:
                opts.append(("largest", by_area[-1]))
            if opts:
                word, a = ref_rng.choice(opts)
                x, y, w, h = a["bbox"]
                box = [round(x), round(y), round(x + w), round(y + h)]
                out.append(("ground_ref", f"Where is the {word} {NAMES[c]}? Give its bounding box.", str(box), box))
    return out




def build(gt_path, out_path, per_type, seed):
    rng = random.Random(seed)
    ref_rng = random.Random(seed + 1)
    d = json.load(open(gt_path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)

    pool = defaultdict(list)
    for im in d["images"]:
        for qtype, q, ans, box in candidates(im, anns[im["id"]], cats, rng, ref_rng):
            pool[qtype].append({"image_id": im["id"], "image": im["file_name"],
                                "width": im["width"], "height": im["height"],
                                "location": im.get("location"), "type": qtype,
                                "question": q, "answer": ans, "box": box})

    take = {"count": per_type, "exists_yes": per_type // 2, "exists_no": per_type // 2,
            "ground": per_type, "left_right": per_type, "ground_ref": per_type}
    rows = []
    for qtype, n in take.items():
        items = pool[qtype]
        rng.shuffle(items)
        if qtype == "count":                          # equal share per answer 1-8: no 'always say 1' shortcut
            by_ans = defaultdict(list)
            for it in items:
                by_ans[it["answer"]].append(it)
            per_ans = n // len(by_ans)
            chosen = [it for ans in sorted(by_ans) for it in by_ans[ans][:per_ans]]
        else:
            chosen = items[:n]
        rows += chosen
        print(f"  {qtype:11s} available {len(items):6d}  taken {len(chosen):5d}")
    for i, r in enumerate(rows):
        r["id"] = i

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = Counter(r["answer"] for r in rows if r["type"] == "count")
    lr = Counter(r["answer"] for r in rows if r["type"] == "left_right")
    print(f"{len(rows)} questions -> {out_path}")
    print("  count answers:", dict(sorted(counts.items())))
    print("  left/right:   ", dict(lr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-type", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(f"data/coco/{a.split}.json", f"data/vqa/{a.split}.jsonl", a.per_type, a.seed)