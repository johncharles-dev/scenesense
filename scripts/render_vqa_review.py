"""Render VQA questions with their ground-truth evidence drawn, for a hand check."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

from scenesense.vqa.build_vqa import NAMES, PLURAL

FONT = cv2.FONT_HERSHEY_SIMPLEX


def classes_in(question):
    """Class names mentioned in the question, in order of appearance."""
    hits = []
    for c in NAMES:
        m = re.search(rf"\b({re.escape(PLURAL[c])}|{re.escape(NAMES[c])})\b", question)
        if m:
            hits.append((m.start(), c))
    return [c for _, c in sorted(hits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="data/vqa/val.jsonl")
    ap.add_argument("--gt", default="data/coco/val.json")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/vqa/review")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.vqa)]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"].split("_")[0]].append(r)            # count, exists, ground, left
    sample = [r for v in by_type.values() for r in v[: a.n // len(by_type)]]

    gt = json.load(open(a.gt))
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    anns = defaultdict(list)
    for x in gt["annotations"]:
        anns[x["image_id"]].append(x)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    colours = [(0, 165, 255), (255, 120, 0)]                  # orange = first class named, blue = second
    for r in sample:
        img = cv2.imread(str(Path(a.images_root) / r["image"]))
        for k, c in enumerate(classes_in(r["question"])):
            for x in anns[r["image_id"]]:
                if cats[x["category_id"]] == c:
                    X, Y, W, H = map(int, x["bbox"])
                    cv2.rectangle(img, (X, Y), (X + W, Y + H), colours[k % 2], 3)
        if r["box"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(img, r["question"], (12, 34), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"answer: {r['answer']}    [{r['type']}  id {r['id']}]", (12, 70), FONT, 1.0,
                    (0, 255, 0), 2)
        cv2.imwrite(str(out / f"{r['id']:04d}_{r['type']}.jpg"), cv2.resize(img, (1280, 720)))

    with open(out / "review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "answer", "verdict", "note"])
        for r in sample:
            w.writerow([r["id"], r["type"], r["question"], r["answer"], "", ""])
    print(f"{len(sample)} images + review.csv -> {out}")


if __name__ == "__main__":
    main()