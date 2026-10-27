"""Model-only latency of the fine-tuned detectors: same input, batch, precision and GPU."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import RTDetrForObjectDetection
from ultralytics import YOLO


def load(name, dtype):
    if name == "rtdetr_r50":
        m = RTDetrForObjectDetection.from_pretrained("runs/rtdetr_r50_ft/final")
        m = m.to("cuda", dtype).eval()
        return lambda x: m(pixel_values=x)
    m = YOLO("runs/yolo11s_ft/weights/best.pt").model.float().fuse()   # fuse conv+BN, as in deployment
    m = m.to("cuda", dtype).eval()
    return lambda x: m(x)


@torch.no_grad()
def bench(fwd, batch, dtype, warmup=30, iters=200):
    x = torch.rand(batch, 3, 640, 640, device="cuda", dtype=dtype)
    for _ in range(warmup):                       # let cuDNN pick kernels, fill caches
        fwd(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fwd(x)
        end.record()
        torch.cuda.synchronize()                  # wait for the GPU before reading the timer
        times.append(start.elapsed_time(end))     # milliseconds
    t = np.array(times)
    med = float(np.median(t))
    return {"median_ms": round(med, 2), "p95_ms": round(float(np.percentile(t, 95)), 2),
            "img_per_s": round(batch * 1000 / med, 1)}


def main():
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")
    print(f"{'model':12s} {'precision':9s} {'batch':>5s} {'median ms':>10s} {'p95 ms':>8s} {'img/s':>8s}")
    for name in ("rtdetr_r50", "yolo11s"):
        for dtype, dname in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            fwd = load(name, dtype)
            for batch in (1, 16):
                r = bench(fwd, batch, dtype)
                rows.append({"model": name, "runtime": "pytorch", "precision": dname, "batch": batch, **r})
                print(f"{name:12s} {dname:9s} {batch:5d} {r['median_ms']:10.2f} {r['p95_ms']:8.2f} {r['img_per_s']:8.1f}")
            del fwd
            torch.cuda.empty_cache()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latency_pytorch.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()