"""AWQ 4-bit (W4A16) quantisation of the merged Qwen2-VL with llm-compressor, calibrated on our VQA train set."""
import json
import sys

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from scenesense.vqa.format import prompt, target

SRC, OUT, N = sys.argv[1], sys.argv[2], 256
proc = AutoProcessor.from_pretrained(SRC, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
model = Qwen2VLForConditionalGeneration.from_pretrained(SRC, dtype=torch.bfloat16)
rows = [json.loads(l) for l in open("data/vqa/train.jsonl")][:N]


def encode(r):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(r)}]},
            {"role": "assistant", "content": [{"type": "text", "text": target(r)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False)
    img = Image.open(f"data/nuimages/{r['image']}").convert("RGB")
    return {k: v.tolist() for k, v in proc(text=[text], images=[img], return_tensors="pt").items()}


ds = Dataset.from_list([encode(r) for r in rows])
recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*visual.*"])
oneshot(model=model, processor=proc, dataset=ds, recipe=recipe, max_seq_length=2048,
        num_calibration_samples=N, data_collator=lambda b: {k: torch.tensor(v) for k, v in b[0].items()})
model.save_pretrained(OUT, save_compressed=True)
proc.save_pretrained(OUT)
print("saved", OUT)