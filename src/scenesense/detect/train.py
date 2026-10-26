"""Fine-tune RT-DETR on nuImages (COCO format) with the Hugging Face Trainer."""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, Trainer, TrainingArguments

from scenesense.data.nuimages_to_coco import CLASSES

ID2LABEL = {i: c for i, c in enumerate(CLASSES)}   # model label i  <->  COCO category_id i+1
LABEL2ID = {c: i for i, c in ID2LABEL.items()}


class CocoDetection(Dataset):
    def __init__(self, gt_path, images_root, processor, limit=None, train=True):
        d = json.load(open(gt_path))
        self.images = d["images"][:limit] if limit else d["images"]
        self.anns = {}
        for a in d["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        self.root, self.proc, self.train = Path(images_root), processor, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = Image.open(self.root / im["file_name"]).convert("RGB")
        anns = []
        for a in self.anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:                      # drop degenerate boxes
                continue
            anns.append({"bbox": [x, y, w, h], "category_id": a["category_id"] - 1,
                         "area": w * h, "iscrowd": 0})
        if self.train and random.random() < 0.5:    # horizontal flip augmentation
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for a in anns:
                x, y, w, h = a["bbox"]
                a["bbox"] = [im["width"] - x - w, y, w, h]
        enc = self.proc(images=img, annotations={"image_id": im["id"], "annotations": anns},
                        return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": [b["labels"] for b in batch]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/coco/train.json")
    ap.add_argument("--images-root", default="data/nuimages")
    ap.add_argument("--model", default="PekingU/rtdetr_r50vd")
    ap.add_argument("--out", default="runs/rtdetr_r50_ft")
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--limit", type=int, default=None, help="use first N images (smoke test)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "rtdetr-nuimages")
    proc = RTDetrImageProcessor.from_pretrained(a.model)
    model = RTDetrForObjectDetection.from_pretrained(
        a.model, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)

    train_ds = CocoDetection(a.train, a.images_root, proc, limit=a.limit, train=True)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": a.backbone_lr},
                                   {"params": rest, "lr": a.lr}], weight_decay=1e-4)

    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        max_grad_norm=0.1,
        bf16=True,
        dataloader_num_workers=a.workers,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        report_to=["mlflow"],
        run_name=Path(a.out).name,
        seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=collate, optimizers=(optimizer, None))
    trainer.train()
    final = Path(a.out) / "final"
    trainer.save_model(final)
    proc.save_pretrained(final)
    print("saved", final)


if __name__ == "__main__":
    main()