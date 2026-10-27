"""Run an RT-DETR checkpoint over a COCO image list and write COCO-format predictions."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from scenesense.data.nuimages_to_coco import CLASS_ID

# Pretrained COCO label -> our class, for the zero-shot baseline (only 6 of our 10 classes exist in COCO)
COCO_TO_OURS = {"car": "car", "truck": "truck", "bus": "bus", "motorbike": "motorcycle",
                "bicycle": "bicycle", "person": "pedestrian"}


@torch.no_grad()
def predict(gt_path, images_root, model_name, out_path, zero_shot, batch=16, limit=None, threshold=0.01):
    gt = json.load(open(gt_path))
    images = gt["images"][:limit] if limit else gt["images"]
    device = "cuda"
    proc = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name).to(device).eval()
    id2label = model.config.id2label
    if zero_shot:
        label_to_cat = {i: CLASS_ID[COCO_TO_OURS[n]] for i, n in id2label.items() if n in COCO_TO_OURS}
    else:
        label_to_cat = {i: CLASS_ID[n] for i, n in id2label.items()}

    results = []
    for start in tqdm(range(0, len(images), batch), desc="predict"):
        chunk = images[start:start + batch]
        pil = [Image.open(Path(images_root) / im["file_name"]).convert("RGB") for im in chunk]
        inputs = proc(images=pil, return_tensors="pt").to(device)
        out = model(**inputs)
        sizes = torch.tensor([[im["height"], im["width"]] for im in chunk], device=device)
        dets = proc.post_process_object_detection(out, target_sizes=sizes, threshold=threshold)
        for im, d in zip(chunk, dets):
            for score, label, box in zip(d["scores"].tolist(), d["labels"].tolist(), d["boxes"].tolist()):
                if label not in label_to_cat:
                    continue
                x1, y1, x2, y2 = box
                results.append({"image_id": im["id"], "category_id": label_to_cat[label],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"))
    print(f"{len(images)} images, {len(results)} detections -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/coco/val.json")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--model", default="PekingU/rtdetr_r50vd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-shot", action="store_true", help="model is COCO-pretrained; map its labels to ours")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    predict(a.gt, a.images_root, a.model, a.out, a.zero_shot, a.batch, a.limit)