# scenesense

Urban scene understanding on Singapore driving video — a fine-tuned detector with tracking, a LoRA-tuned vision-language model for box-grounded scene QA, and an LLM agent that answers questions over the detection logs with frame citations.

Built on [nuImages](https://www.nuscenes.org/nuimages) (Singapore and Boston scenes; CC BY-NC-SA 4.0). Every headline number is reported with a bootstrap 95% confidence interval.

Work in progress — results table filled in week by week.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (developed on an RTX 5090, CUDA 12.8).

```bash
git clone https://github.com/johncharles-dev/scenesense && cd scenesense
uv sync
```

## Data

[nuImages v1.0](https://www.nuscenes.org/nuimages) — 67,279 train / 16,445 val keyframes (1600×900) from Singapore (one-north) and Boston (seaport). Licence CC BY-NC-SA 4.0; download requires registration and is not included here.

The 25 nuImages categories are merged to the 10 nuScenes detection classes (car, truck, bus, trailer, construction_vehicle, motorcycle, bicycle, pedestrian, traffic_cone, barrier); other categories are dropped. Each COCO image record carries `location` and `timestamp` so results can be sliced by city and time of day.

```bash
uv run python -m scenesense.data.nuimages_to_coco --version v1.0-train
uv run python -m scenesense.data.nuimages_to_coco --version v1.0-val
```


