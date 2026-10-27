# scenesense

Urban scene understanding on Singapore driving video — a fine-tuned detector with tracking, a LoRA-tuned vision-language model for box-grounded scene QA, and an LLM agent that answers questions over the detection logs with frame citations.

Built on [nuImages](https://www.nuscenes.org/nuimages) (Singapore and Boston scenes; CC BY-NC-SA 4.0). Every headline number is reported with a bootstrap 95% confidence interval.

Work in progress — results table filled in week by week.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (developed on an RTX 5090, CUDA 13).

```bash
git clone https://github.com/johncharles-dev/scenesense && cd scenesense
uv sync
```

## Data

[nuImages v1.0](https://www.nuscenes.org/nuimages) — 67,279 train / 16,445 val keyframes (1600×900) from Singapore (one-north) and Boston (seaport). Licence CC BY-NC-SA 4.0; download requires registration and is not included here.

The 25 nuImages categories are merged to the 10 nuScenes detection classes (car, truck, bus, trailer, construction_vehicle, motorcycle, bicycle, pedestrian, traffic_cone, barrier); other categories are dropped. Each COCO image record carries `location`, `timestamp`, `local_hour` and `daypart` so results can be sliced by city and time of day.

```bash
uv run python -m scenesense.data.nuimages_to_coco --version v1.0-train
uv run python -m scenesense.data.nuimages_to_coco --version v1.0-val
```

## Results — detection (nuImages val, 16,445 frames)

| Model | Params | Train (12 ep, 640 px, RTX 5090) | mAP@[.5:.95] | 95% CI |
|---|---|---|---|---|
| RT-DETR r50vd, COCO weights (zero-shot) | 42M | — | 0.173 | [0.171, 0.177] |
| YOLO11s, fine-tuned | 9.4M | 26 min | 0.404 | [0.400, 0.410] |
| **RT-DETR r50vd, fine-tuned** | 42M | 3 h 11 min | **0.477** | [0.470, 0.483] |

Paired bootstrap (200 resamples of val images): fine-tuning adds **+0.304** [0.297, 0.309] over zero-shot; RT-DETR beats YOLO11s by **+0.072** [0.069, 0.076] and leads on all 10 classes.

| | car | truck | bus | trailer | constr. | moto | bicycle | ped | cone | barrier | AP small | AP large |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR ft | 0.635 | 0.497 | 0.535 | 0.258 | 0.310 | 0.580 | 0.519 | 0.430 | 0.505 | 0.499 | 0.205 | 0.683 |
| YOLO11s ft | 0.574 | 0.437 | 0.437 | 0.179 | 0.250 | 0.504 | 0.448 | 0.348 | 0.434 | 0.434 | 0.134 | 0.623 |

Small objects are the main weakness (AP small 0.205 vs 0.683 large); higher input resolution is the next experiment. Models were trained with their standard recipes at equal epochs and resolution, so this is a recipe-level comparison.

### Reproduce

```bash
uv run python -m scenesense.detect.train --out runs/rtdetr_r50_ft
uv run python -m scenesense.detect.predict --model runs/rtdetr_r50_ft/final --out data/preds/rtdetr_ft_val.json
uv run python -m scenesense.data.coco_to_yolo
uv run yolo detect train data=data/yolo/nuimages.yaml model=yolo11s.pt imgsz=640 epochs=12 batch=32 project=$PWD/runs name=yolo11s_ft
uv run python -m scenesense.eval.map_ci --gt data/coco/val.json \
    --pred data/preds/rtdetr_ft_val.json --pred data/preds/yolo11s_ft_val.json --n-boot 200
```