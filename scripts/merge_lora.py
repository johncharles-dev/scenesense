"""Merge a LoRA adapter into Qwen2-VL-2B and save a standalone model for serving."""
import sys

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

base, adapter, out = "Qwen/Qwen2-VL-2B-Instruct", sys.argv[1], sys.argv[2]
model = Qwen2VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()   # W + A·B baked into W
model.save_pretrained(out)
AutoProcessor.from_pretrained(base).save_pretrained(out)
print("saved", out)