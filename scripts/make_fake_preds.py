"""Build fake COCO result files from GT so the eval harness can be tested before any model exists."""
import json, sys

gt = json.load(open(sys.argv[1]))
perfect, shifted = [], []
for i, a in enumerate(gt["annotations"]):
    x, y, w, h = a["bbox"]
    perfect.append({"image_id": a["image_id"], "category_id": a["category_id"], "bbox": [x, y, w, h], "score": 0.9})
    dx = 0.0 if i % 2 else 0.8 * w          # shift every other box far enough to miss IoU 0.5
    shifted.append({"image_id": a["image_id"], "category_id": a["category_id"], "bbox": [x + dx, y, w, h], "score": 0.9})
json.dump(perfect, open("data/coco/pred_perfect.json", "w"))
json.dump(shifted, open("data/coco/pred_shifted.json", "w"))
print(len(perfect), "boxes written")