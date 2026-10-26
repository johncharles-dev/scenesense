from nuimages import NuImages
from collections import Counter, defaultdict
import json
from pathlib import Path

# nuImages fine class -> nuScenes detection class (10). Anything missing is dropped.
CLASS_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}
CLASSES = ["car", "truck", "bus", "trailer", "construction_vehicle",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone", "barrier"]
CLASS_ID = {name: i + 1 for i, name in enumerate(CLASSES)}   # COCO ids start at 1


def boxes_by_frame(n):
    by_frame = defaultdict(list)
    for a in n.object_ann:
        by_frame[a["sample_data_token"]].append(a)
    return by_frame

def convert(dataroot: str, version: str, out_path: str):
    n = NuImages(dataroot=dataroot, version=version, lazy=True, verbose=False)
    by_frame = boxes_by_frame(n)
    images, annotations = [], []
    total = Counter()
    ann_id = 1
    for img_id, s in enumerate(n.sample, start=1):
        sd = n.get("sample_data", s["key_camera_token"])
        log = n.get("log", s["log_token"])
        images.append({
            "id": img_id,
            "file_name": sd["filename"],
            "width": sd["width"],
            "height": sd["height"],
            "location": log["location"],
            "timestamp": sd["timestamp"],
        })
        for a in by_frame[sd["token"]]:
            name = n.get("category", a["category_token"])["name"]
            if name not in CLASS_MAP:
                continue
            cls = CLASS_MAP[name]
            x1, y1, x2, y2 = a["bbox"]
            w, h = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_ID[cls],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
            total[cls] += 1

    categories = [{"id": CLASS_ID[c], "name": c} for c in CLASSES]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    print(f"{version}: {len(images)} images, {len(annotations)} boxes -> {out}")
    for c in CLASSES:
        print(f"  {c:22s} {total[c]}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert nuImages to COCO detection JSON.")
    ap.add_argument("--dataroot", default="data/nuimages")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default=None, help="defaults to data/coco/<split>.json")
    a = ap.parse_args()
    out = a.out or f"data/coco/{a.version.replace('v1.0-', '')}.json"
    convert(a.dataroot, a.version, out)

