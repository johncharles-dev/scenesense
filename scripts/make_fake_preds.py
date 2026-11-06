"""Build fake COCO result files from GT so the eval harness can be tested before any model exists."""
import sys

from scenesense.utils import read_json, write_json

gt = read_json(sys.argv[1])
perfect, shifted = [], []
for i, a in enumerate(gt["annotations"]):
    x, y, w, h = a["bbox"]
    perfect.append({"image_id": a["image_id"], "category_id": a["category_id"], "bbox": [x, y, w, h], "score": 0.9})
    dx = 0.0 if i % 2 else 0.8 * w          # shift every other box far enough to miss IoU 0.5
    shifted.append({"image_id": a["image_id"], "category_id": a["category_id"], "bbox": [x + dx, y, w, h], "score": 0.9})
write_json(perfect, "data/coco/pred_perfect.json")
write_json(shifted, "data/coco/pred_shifted.json")
print(len(perfect), "boxes written")