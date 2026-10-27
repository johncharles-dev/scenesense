"""Detect and track objects through nuScenes scenes (CAM_FRONT, 12 Hz); write an event log and a demo video."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from torchvision.ops import nms
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from scenesense.track.bytetrack import ByteTrack


def scene_frames(nusc, scene, channel="CAM_FRONT"):
    """Every camera frame of a scene in time order: keyframes and sweeps, via sample_data.next."""
    first = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", first["data"][channel])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    frames = []
    while True:
        frames.append(sd)
        if not sd["next"]:
            break
        sd = nusc.get("sample_data", sd["next"])
    return frames


def render_video(nusc, frames, events, id2label, path):
    """Draw the final event log: boxes, '#ID class' labels and trails, coloured by track ID."""
    label2id = {v: k for k, v in id2label.items()}
    by_frame = {}
    for e in events:
        by_frame.setdefault(e["frame"], []).append(e)
    boxes_ann = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    labels_ann = sv.LabelAnnotator(text_scale=0.5, color_lookup=sv.ColorLookup.TRACK)
    trace_ann = sv.TraceAnnotator(trace_length=24, color_lookup=sv.ColorLookup.TRACK)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12,
                             (frames[0]["width"], frames[0]["height"]))
    for i, sd in enumerate(frames):
        img = cv2.imread(str(Path(nusc.dataroot) / sd["filename"]))
        ev = by_frame.get(i, [])
        if ev:
            dets = sv.Detections(xyxy=np.array([e["box"] for e in ev], dtype=float),
                                 confidence=np.array([e["score"] for e in ev]),
                                 class_id=np.array([label2id[e["class"]] for e in ev]),
                                 tracker_id=np.array([e["track_id"] for e in ev]))
            img = trace_ann.annotate(img, dets)
            img = boxes_ann.annotate(img, dets)
            img = labels_ann.annotate(img, dets, labels=[f"#{e['track_id']} {e['class']}" for e in ev])
        writer.write(img)
    writer.release()


@torch.no_grad()
def track_scene(nusc, scene, model, proc, out_dir, threshold=0.1, video=False,
                new_thresh=0.4, min_len=6, nms_iou=0.7):
    log = nusc.get("log", scene["log_token"])
    frames = scene_frames(nusc, scene)
    id2label = model.config.id2label
    tracker = ByteTrack(frame_rate=12, new=new_thresh)
    t0 = frames[0]["timestamp"]
    events = []
    start = time.time()
    for i, sd in enumerate(frames):
        bgr = cv2.imread(str(Path(nusc.dataroot) / sd["filename"]))
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = proc(images=rgb, return_tensors="pt").to("cuda")
        out = model(**inputs)
        det = proc.post_process_object_detection(
            out, target_sizes=torch.tensor([[sd["height"], sd["width"]]], device="cuda"),
            threshold=threshold)[0]
        keep = nms(det["boxes"], det["scores"], nms_iou)        # class-agnostic: one box per object
        xyxy, tids, scores, clss = tracker.update(det["boxes"][keep].cpu().numpy(),
                                                 det["scores"][keep].cpu().numpy(),
                                                 det["labels"][keep].cpu().numpy())
        for box, score, cls, tid in zip(xyxy, scores, clss, tids):
            events.append({
                "scene": scene["name"], "location": log["location"], "frame": i,
                "timestamp": sd["timestamp"], "t": round((sd["timestamp"] - t0) / 1e6, 3),
                "track_id": int(tid), "class": id2label[int(cls)], "score": round(float(score), 3),
                "box": [round(float(v), 1) for v in box],
            })
    secs = time.time() - start

    lengths = Counter(e["track_id"] for e in events)
    kept = [e for e in events if lengths[e["track_id"]] >= min_len]
    votes = {}
    for e in kept:
        votes.setdefault(e["track_id"], Counter())[e["class"]] += 1
    for e in kept:                                          # one stable class per track
        e["det_class"] = e["class"]
        e["class"] = votes[e["track_id"]].most_common(1)[0][0]

    events_path = Path(out_dir) / f"{scene['name']}.jsonl"
    with open(events_path, "w") as f:
        for e in kept:
            f.write(json.dumps(e) + "\n")
    if video:
        render_video(nusc, frames, kept, id2label, Path(out_dir) / f"{scene['name']}.mp4")

    flips = sum(1 for e in kept if e["det_class"] != e["class"])
    print(f"{scene['name']:11s} {log['location']:24s} {len(frames):4d} frames  "
          f"{len(lengths):4d} tracks -> {len(votes):4d} kept  {len(kept):6d} events  "
          f"class flips {flips / max(len(kept), 1):4.1%}  {len(frames) / secs:5.1f} fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--model", default="runs/rtdetr_r50_ft/final")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene names; default all")
    ap.add_argument("--video", nargs="*", default=[], help="scene names to render as MP4")
    ap.add_argument("--threshold", type=float, default=0.1, help="detector score threshold")
    ap.add_argument("--new-thresh", type=float, default=0.4, help="min score to start a new track")
    ap.add_argument("--min-len", type=int, default=6, help="drop tracks shorter than this many frames")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="class-agnostic NMS IoU before tracking")
    ap.add_argument("--out", default="data/tracks")
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    proc = RTDetrImageProcessor.from_pretrained(a.model)
    model = RTDetrForObjectDetection.from_pretrained(a.model).to("cuda").eval()
    for scene in nusc.scene:
        if a.scenes and scene["name"] not in a.scenes:
            continue
        track_scene(nusc, scene, model, proc, a.out, a.threshold, video=scene["name"] in a.video,
                    new_thresh=a.new_thresh, min_len=a.min_len, nms_iou=a.nms_iou)


if __name__ == "__main__":
    main()