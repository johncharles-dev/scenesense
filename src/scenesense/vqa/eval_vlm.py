"""Evaluate Qwen2-VL (optionally with a LoRA adapter) on the scenesense VQA set."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from scenesense.vqa.format import family, prompt, score

MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1024 * 28 * 28     # ~1,000 visual tokens per 1600x900 image


def accuracy_ci(flags, n_boot=1000, seed=0):
    x = np.array(flags, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def report(preds, verified_keys, title):
    print(f"\n{title}")
    print(f"{'type':11s} {'n':>5s} {'acc':>7s}   95% CI           majority baseline")
    by_f = defaultdict(list)
    for p in preds:
        by_f[family(p["type"])].append(p)
    summary = {}
    for f in ("count", "exists", "left_right", "ground"):
        ps = by_f.get(f, [])
        if not ps:
            continue
        acc, lo, hi = accuracy_ci([p["correct"] for p in ps])
        answers = [p["answer"] for p in ps]
        base = 0.0 if f == "ground" else max(answers.count(a) for a in set(answers)) / len(answers)
        print(f"{f:11s} {len(ps):5d} {acc:7.1%}   [{lo:.1%}, {hi:.1%}]   {base:.1%}")
        summary[f] = {"n": len(ps), "acc": acc, "ci95": [lo, hi], "majority": base}
    acc, lo, hi = accuracy_ci([p["correct"] for p in preds])
    print(f"{'all':11s} {len(preds):5d} {acc:7.1%}   [{lo:.1%}, {hi:.1%}]")
    summary["all"] = {"n": len(preds), "acc": acc, "ci95": [lo, hi]}
    ver = [p for p in preds if (p["image_id"], p["question"]) in verified_keys]
    if ver:
        acc, lo, hi = accuracy_ci([p["correct"] for p in ver])
        print(f"{'verified':11s} {len(ver):5d} {acc:7.1%}   [{lo:.1%}, {hi:.1%}]")
        summary["verified"] = {"n": len(ver), "acc": acc, "ci95": [lo, hi]}
    return summary


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory")
    ap.add_argument("--vqa", default="data/vqa/val.jsonl")
    ap.add_argument("--verified", default="data/vqa/val_verified.jsonl")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default="qwen2vl2b_zeroshot")
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "left"                      # batched generation needs left padding
    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
                                                            device_map="cuda", attn_implementation="sdpa")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(a.vqa)][: a.limit]
    verified = {(r["image_id"], r["question"]) for r in map(json.loads, open(a.verified))} \
        if Path(a.verified).exists() else set()
    preds = []
    for i in tqdm(range(0, len(rows), a.batch), desc="vqa"):
        chunk = rows[i:i + a.batch]
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(r)}]}]
                for r in chunk]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        imgs = [Image.open(Path(a.images_root) / r["image"]).convert("RGB") for r in chunk]
        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        answers = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for r, ans in zip(chunk, answers):
            ans = ans.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            preds.append({**r, "pred": ans, "correct": bool(score(r, ans))})

    Path("data/preds").mkdir(parents=True, exist_ok=True)
    with open(f"data/preds/vqa_{a.name}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    summary = report(preds, verified, a.name)
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/vqa_{a.name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()