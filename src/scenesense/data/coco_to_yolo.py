"""Write an Ultralytics YOLO dataset (symlinked images + txt labels) from our COCO JSON."""
import argparse
import json
import os
from pathlib import Path

from scenesense.data.nuimages_to_coco import CLASSES


def convert(gt_path, images_root, out_root, split):
    d = json.load(open(gt_path))
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    img_dir = Path(out_root) / "images" / split
    lbl_dir = Path(out_root) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_box = 0
    for im in d["images"]:
        src = (Path(images_root) / im["file_name"]).resolve()
        dst = img_dir / f"{im['id']}.jpg"          # numeric name -> YOLO reports image_id = this number
        if not dst.exists():
            os.symlink(src, dst)
        W, H = im["width"], im["height"]
        lines = []
        for a in anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)       # clip to the image; YOLO drops out-of-range labels
            x2, y2 = min(W, x + w), min(H, y + h)
            w, h = x2 - x1, y2 - y1
            if w < 1 or h < 1:
                continue
            cls = a["category_id"] - 1              # YOLO classes start at 0, COCO ids at 1
            lines.append(f"{cls} {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (lbl_dir / f"{im['id']}.txt").write_text("\n".join(lines))
        n_box += len(lines)
    print(f"{split}: {len(d['images'])} images, {n_box} boxes")


def write_yaml(out_root):
    root = Path(out_root).resolve()
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (root / "nuimages.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print("wrote", root / "nuimages.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default="data/coco")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--out", default="data/yolo")
    a = ap.parse_args()
    for split in ("train", "val"):
        convert(f"{a.coco_dir}/{split}.json", a.images_root, a.out, split)
    write_yaml(a.out)