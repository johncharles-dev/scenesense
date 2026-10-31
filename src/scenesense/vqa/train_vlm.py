"""LoRA fine-tuning of Qwen2-VL-2B on the scenesense VQA set."""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments

from scenesense.vqa.eval_vlm import MAX_PIXELS, MIN_PIXELS
from scenesense.vqa.format import prompt, target


class VQA(Dataset):
    def __init__(self, path, limit=None):
        self.rows = [json.loads(l) for l in open(path)][:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Builds chat-formatted batches; the loss covers only the answer tokens."""

    def __init__(self, proc, images_root):
        self.proc, self.root = proc, Path(images_root)
        self.marker = proc.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, rows):
        texts, imgs = [], []
        for r in rows:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(r)}]},
                    {"role": "assistant", "content": [{"type": "text", "text": target(r)}]}]
            texts.append(self.proc.apply_chat_template(msgs, tokenize=False))
            imgs.append(Image.open(self.root / r["image"]).convert("RGB"))
        batch = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100                      # ignore padding
        m = len(self.marker)
        for i, ids in enumerate(batch["input_ids"].tolist()):
            start = max(j + m for j in range(len(ids) - m + 1) if ids[j:j + m] == self.marker)
            labels[i, :start] = -100                                     # ignore system, image, question
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--train", default="data/vqa/train.jsonl")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--out", default="runs/qwen2vl2b_lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    a = ap.parse_args()

    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "vlm-vqa")
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "right"
    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=50, bf16=True,
        logging_steps=20, save_strategy="epoch", save_total_limit=2, report_to=["mlflow"],
        run_name=Path(a.out).name, remove_unused_columns=False, dataloader_num_workers=8, seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=VQA(a.train, a.limit),
                      data_collator=Collator(proc, a.images_root))
    trainer.train()
    model.save_pretrained(Path(a.out) / "adapter")
    proc.save_pretrained(Path(a.out) / "adapter")
    print("saved", Path(a.out) / "adapter")


if __name__ == "__main__":
    main()