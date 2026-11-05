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

[nuImages v1.0](https://www.nuscenes.org/nuimages) — 67,279 train / 16,445 val keyframes (1600×900) from Singapore (one-north, Queenstown) and Boston (seaport); train is 75% Singapore / 25% Boston. Licence CC BY-NC-SA 4.0; download requires registration and is not included here.

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

### Slices

| Slice | Frames | RT-DETR ft | YOLO11s ft | Zero-shot | RT-DETR − YOLO (paired) |
|---|---|---|---|---|---|
| All val | 16,445 | 0.477 [0.470, 0.483] | 0.404 [0.400, 0.410] | 0.173 | +0.072 [0.069, 0.076] |
| Boston | 3,051 | 0.427 [0.417, 0.439] | 0.344 [0.333, 0.356] | 0.165 | +0.083 [0.076, 0.090] |
| Night (one drive) | 192 | 0.489 [0.418, 0.594] | 0.332 [0.254, 0.417] | 0.201 | +0.158 [0.097, 0.236] |

RT-DETR's lead holds in every slice and roughly doubles at night. The night slice is a single 192-frame drive, so its intervals understate the true uncertainty; treat it as indicative. Cross-slice differences for one model mix city and class-mix effects. Fine-tuned models drop 0.05–0.06 mAP in Boston while the zero-shot model drops only 0.008, suggesting fine-tuning specialised toward the majority city (train is 75% Singapore).

## Results — grounded scene QA (Qwen2-VL-2B)

1,248 templated test questions from nuImages val images (answers from ground-truth boxes; 200 hand-checked, 185 kept as a verified subset).

| Type | Zero-shot | LoRA (r=16, 0.8% of weights) | Paired gain, 95% CI |
|---|---|---|---|
| Count (1–8, balanced) | 21.4% | 39.1% | +17.7 [+11.3, +25.0] |
| Exists (yes/no) | 82.8% | 91.6% | +8.8 [+4.8, +12.8] |
| Left / right | 78.8% | 95.2% | +16.4 [+11.6, +21.6] |
| Box grounding (IoU ≥ 0.5) | 55.8% | 74.6% | +18.8 |
| **All** | **59.0%** | **75.1%** | **+16.1 [+13.9, +18.6]** |
| Verified subset (185) | 70.8% | 84.9% | +14.1 |

A second epoch added nothing measurable (+0.6 [−0.9, +2.2]). The hand check found ~5% label noise in nuImages (reflections in shop windows labelled as vehicles), concentrated in single-object grounding questions.

## Results — scene-query agent

An LLM agent answers questions about the 10 nuScenes-mini scenes with a hand-written tool-calling loop (OpenAI-compatible API on vLLM, no framework). Facts come from tools: `count` computes distinct track IDs from the tracking log (per scene, or per-scene breakdown + total + argmax over a location / day-night set); `search` retrieves 2-second windows from Chroma to find moments; `ask_vlm` sends a frame to the fine-tuned VLM using only its four trained question forms and ten classes. Every claim cites `(scene, time)`.

**Eval set.** 51 held-out questions, written before either model ran on them and excluding the 7 dev questions used while building the agent: count 10, total 5, compare 10, find-a-moment 8, VQA on a frame 8, grounding 1, unanswerable 9. Truth for counts, comparisons and moments is computed from the tracking logs by `scenesense.agent.build_eval`; VQA items use the tracker as reference at frames also checked by eye. Scoring is rule-based (unit-tested regexes) with a full hand audit. One run per model at temperature 0, same frozen prompt and tools; 95% bootstrap CIs over questions, paired for the difference.

| Metric | Qwen2.5-7B bf16 | Qwen2.5-14B-AWQ | Paired diff (14B − 7B) |
|---|---|---|---|
| Answer accuracy, answerable (n=42) | 97.6% [92.9, 100] | 92.9% [85.7, 100] | −4.8 [−11.9, 0.0] |
| Correct refusal, unanswerable (n=9) | 55.6% [22.2, 88.9] | 100% | +44.4 [+11.1, +77.8] |
| Citation includes a time | 45.2% [31.0, 59.5] | 95.2% [88.1, 100] | +50.0 [+35.7, +64.3] |
| Citation supported by the log (class present/absent in that scene within ±1 s, as claimed) | 40.5% [26.2, 54.8] | 88.1% [78.6, 97.6] | +47.6 [+28.6, +64.3] |
| Tool calls that parse and run | 93.8% (61/65) | 95.2% (59/62) | — |
| Known upstream failures (n=2): names the described scene / claim supported by any tool | 2/2 / 0/2 | 2/2 / 0/2 | — |
| Latency p50 / mean | 0.68 s / 0.79 s | 0.64 s / 0.94 s | — |

The 7B model is slightly better on answerable questions, but the difference is not significant (n=42; the CI's upper bound is exactly 0; it is two find-a-moment questions where the 14B checked the VLM, got "no" on a night frame and hedged). The 14B model is clearly better on refusals and on citations that the log actually supports. **We use Qwen2.5-14B-Instruct-AWQ**: for an agent, a confident wrong answer costs more than a refusal. The 7B answered "How many cars are in the Paris scenes?" with Queenstown's count (85) and turned "dogs" (not a tracked class) into "there are no dogs"; the 14B refused all 9 unanswerable questions. Both fit the same 0.60 GPU-memory budget. All failed tool calls were intended validation rejections (unknown class or scene, time outside the scene); none were malformed. Answerable questions are mostly easy *by design*: one `count` call returns the breakdown and argmax, so the discriminating part is refusal and citation discipline. The hand audit disagreed with the rule-based citation-support check once (a 14B answer that names a scene where the log has no bicycle, then hedges "not consistently visible"); audited, the 14B figure is 36/42.

**Moved out of the main set: known upstream failures.** Two questions were moved out of the main set because the detector/tracker never found the object, so no tool can confirm it: "parked bicycles" in scene-0916 and a "construction vehicle" in scene-1077 appear in the scene descriptions but have zero tracks. They are reported in their own row rather than dropped, because dropping hard cases would make the main set look easier than it is. Both models name the right scene (from the description) in 2/2, but for the wrong reasons: neither claim is supported by tool output, and the 14B's only "evidence" for the construction vehicle was a VLM count of 1 on a frame where none is visible (checked by eye).

**Failure analysis.**
- *An error passed on from one stage to the next:* in scene-1100 at 8 s the VLM answers "yes" to "Is there a truck in the image?". The tracker has no truck anywhere in that scene, and none is visible in the frame (checked by eye). Both agents faithfully report "there is a truck" with a correct-looking citation. The agent did its job and the claim is still false: an agent is only as reliable as its weakest tool, and nothing in the loop cross-checks the VLM against the tracker.
- Place substitution (7B): unknown places ("Paris") answered with a real location's data.
- Unknown class turned into zero (7B): "Unknown object class 'dog'" became "there are no dogs".
- Behaviour stated as fact (7B): "9 pedestrians crossed" with no proxy label.
- Tracker vs VLM disagreement on moments (both): the tracker has a motorcycle in a window, the VLM says "no" on the sampled frames (small or distant objects, night).
- Retrieval: the window text embeds the per-scene description plus the classes present, so retrieval works at scene level and only refines by class within a scene.

```bash
uv run python -m scenesense.agent.index                 # Chroma window index
uv run python -m scenesense.agent.build_eval            # 51 questions + 2 upstream-failure questions
uv run python -m scenesense.agent.loop "Which of the Boston scenes has the fewest buses?"
uv run python -m scenesense.agent.eval_agent run --name qwen14b_awq
uv run python -m scenesense.agent.eval_agent run --name upstream_qwen14b_awq --questions data/agent_eval/upstream.jsonl
uv run python -m scenesense.agent.eval_agent compare qwen7b qwen14b_awq
```

Serving (VLM on :8001, agent LLM on :8002):

```bash
vllm serve runs/qwen2vl2b_merged --served-model-name scenesense-vlm --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 1}' --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 802816}' \
    --gpu-memory-utilization 0.25 --port 8001
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name agent-llm --max-model-len 8192 \
    --gpu-memory-utilization 0.60 --enable-auto-tool-choice --tool-call-parser hermes --port 8002
```

### Reproduce

```bash
uv run python -m scenesense.detect.train --out runs/rtdetr_r50_ft
uv run python -m scenesense.detect.predict --model runs/rtdetr_r50_ft/final --out data/preds/rtdetr_ft_val.json
uv run python -m scenesense.data.coco_to_yolo
uv run yolo detect train data=data/yolo/nuimages.yaml model=yolo11s.pt imgsz=640 epochs=12 batch=32 project=$PWD/runs name=yolo11s_ft
uv run yolo detect val data=data/yolo/nuimages.yaml model=runs/yolo11s_ft/weights/best.pt imgsz=640 conf=0.001 save_json=True project=$PWD/runs name=yolo11s_val
cp runs/yolo11s_val/predictions.json data/preds/yolo11s_ft_val.json
uv run python -m scenesense.eval.map_ci --gt data/coco/val.json \
    --pred data/preds/rtdetr_ft_val.json --pred data/preds/yolo11s_ft_val.json --n-boot 200
```
