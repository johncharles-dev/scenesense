"""Evaluate the vLLM-served VLM through its OpenAI-compatible API; measure latency and throughput."""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from scenesense.vqa.eval_vlm import report
from scenesense.vqa.format import prompt, score


async def ask(client, model, r, root, sem):
    b64 = base64.b64encode((Path(root) / r["image"]).read_bytes()).decode()
    async with sem:                                   # at most N requests in flight
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=32,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt(r)}]}])
        dt = time.perf_counter() - t0
    ans = resp.choices[0].message.content or ""
    return {**r, "pred": ans, "correct": bool(score(r, ans)), "latency_s": dt}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="scenesense-vlm")
    ap.add_argument("--vqa", default="data/vqa/val.jsonl")
    ap.add_argument("--verified", default="data/vqa/val_verified.jsonl")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default="vllm_bf16")
    a = ap.parse_args()

    client = AsyncOpenAI(base_url=a.url, api_key="none")
    rows = [json.loads(l) for l in open(a.vqa)][: a.limit]
    verified = {(r["image_id"], r["question"]) for r in map(json.loads, open(a.verified))}
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.perf_counter()
    preds = await asyncio.gather(*(ask(client, a.model, r, a.images_root, sem) for r in rows))
    wall = time.perf_counter() - t0
    lat = np.array([p["latency_s"] for p in preds])
    print(f"{len(preds)} requests in {wall:.1f} s -> {len(preds) / wall:.1f} req/s | "
          f"latency p50 {np.median(lat) * 1000:.0f} ms, p95 {np.percentile(lat, 95) * 1000:.0f} ms "
          f"(concurrency {a.concurrency})")
    summary = report(preds, verified, a.name)
    with open(f"data/preds/vqa_{a.name}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    summary["speed"] = {"req_per_s": len(preds) / wall, "p50_ms": float(np.median(lat) * 1000),
                        "p95_ms": float(np.percentile(lat, 95) * 1000), "concurrency": a.concurrency}
    Path(f"reports/vqa_{a.name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())