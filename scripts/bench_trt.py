"""Model-only latency of TensorRT engines, timed exactly like bench_latency.py."""
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

TRT2TORCH = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
             trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64, trt.DataType.BOOL: torch.bool}


def load_engine(path):
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    data = Path(path).read_bytes()
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:                      # Ultralytics prepends a 4-byte length + JSON metadata header
        n = int.from_bytes(data[:4], "little")
        engine = runtime.deserialize_cuda_engine(data[4 + n:])
    return engine


@torch.no_grad()
def bench_engine(path, warmup=30, iters=200):
    engine = load_engine(path)
    ctx = engine.create_execution_context()
    buffers, batch = [], None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        t = torch.empty(shape, dtype=TRT2TORCH[engine.get_tensor_dtype(name)], device="cuda")
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t.uniform_()
            batch = shape[0]
        ctx.set_tensor_address(name, t.data_ptr())   # TensorRT reads/writes these GPU buffers directly
        buffers.append(t)
    stream = torch.cuda.current_stream()
    run = lambda: ctx.execute_async_v3(stream.cuda_stream)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = np.array(times)
    med = float(np.median(t))
    return batch, {"median_ms": round(med, 2), "p95_ms": round(float(np.percentile(t, 95)), 2),
                   "img_per_s": round(batch * 1000 / med, 1)}


def main():
    # usage: bench_trt.py name=path.engine [name=path.engine ...]
    rows = []
    print(f"TensorRT {trt.__version__} | {torch.cuda.get_device_name(0)}\n")
    print(f"{'model':12s} {'batch':>5s} {'median ms':>10s} {'p95 ms':>8s} {'img/s':>8s}")
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        batch, r = bench_engine(path)
        rows.append({"model": name, "runtime": "tensorrt", "precision": "fp16", "batch": batch, **r})
        print(f"{name:12s} {batch:5d} {r['median_ms']:10.2f} {r['p95_ms']:8.2f} {r['img_per_s']:8.1f}")
    Path("reports/latency_trt.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()