"""Demo GIF for the README: a tracking clip, then one recorded agent trace and the frame it cites.

Everything shown comes from saved artifacts (tracking video, tracking log, agent eval predictions); nothing is re-run.

    uv run python scripts/make_demo_gif.py --out assets/demo.gif
"""
import argparse
import json
import textwrap
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from scenesense.agent import tools

W, H = 640, 360
BG, FG, DIM, ACC, OK = (24, 26, 31), (230, 230, 230), (140, 145, 155), (255, 196, 0), (110, 210, 120)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def banner(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 26, W, H], fill=BG)
    d.text((8, H - 21), text, font=font(14), fill=FG)
    return img


def clip_frames(video, start, end, step):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for i in range(start, end):
        ok, f = cap.read()
        if not ok:
            break
        if (i - start) % step == 0:
            out.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA),
                                                    cv2.COLOR_BGR2RGB)))
    return out


def card(lines):
    """Terminal-style text card. lines: (text, colour, bold) tuples; long lines wrap."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 14
    for text, colour, bold in lines:
        for part in textwrap.wrap(text, 62) or [""]:
            d.text((14, y), part, font=font(15, bold), fill=colour)
            y += 21
    return img


def trace_cards(pred):
    q = [("Q: " + pred["question"], ACC, True), ("", FG, False)]
    cards = [card(q)]
    shown = list(q)
    for step in pred["trace"]:
        args = json.loads(step["args"])
        shown.append((f"→ {step['tool']}({', '.join(f'{k}={v!r}' for k, v in args.items())})", FG, False))
        res = step["result"]
        if step["tool"] == "search":
            top = res[0]
            summary = f"rank 1: {top['scene']} {top['t_start']:g}-{top['t_end']:g} s, objects: {', '.join(top['objects_present'])}"
        else:
            summary = f"frame {res['frame']} (t={res['t']:g} s): answer {res['answer']!r}"
        shown.append(("  " + summary, DIM, False))
        cards.append(card(shown))
    shown += [("", FG, False), ("A: " + pred["answer"], OK, True)]
    cards.append(card(shown))
    return cards


def cited_frame(pred, cls):
    """The frame the agent asked the VLM about, with the tracker's boxes for the queried class drawn on it."""
    vlm = [s for s in pred["trace"] if s["tool"] == "ask_vlm"][-1]["result"]
    fr = tools.get_frame(vlm["scene"], vlm["t"])
    img = cv2.imread(fr["image"])
    for e in tools._events(vlm["scene"]):
        if e["frame"] == fr["frame"] and e["class"] == cls:
            x1, y1, x2, y2 = map(int, e["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 196, 255), 4)
            cv2.putText(img, f"#{e['track_id']} {cls} (tracker)", (x1, max(y1 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 196, 255), 3)
    pil = Image.fromarray(cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return banner(pil, f"{vlm['scene']}, {vlm['t']:g} s - VLM: \"{vlm['question']}\" -> {vlm['answer']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/tracks/scene-0796.mp4")
    ap.add_argument("--start", type=int, default=36)
    ap.add_argument("--end", type=int, default=108)           # 6 s at 12 Hz
    ap.add_argument("--preds", default="data/preds/agent_eval_qwen14b_awq.jsonl")
    ap.add_argument("--id", type=int, default=27)
    ap.add_argument("--cls", default="bus")
    ap.add_argument("--out", default="assets/demo.gif")
    a = ap.parse_args()

    scene = Path(a.video).stem
    frames = [banner(f, f"{scene}: RT-DETR + ByteTrack, colour = track ID") for f in clip_frames(a.video, a.start, a.end, 2)]
    durations = [167] * len(frames)                            # 6 fps, real time
    with open(a.preds) as f:
        pred = next(p for p in map(json.loads, f) if p["id"] == a.id)
    cards = trace_cards(pred)
    frames += cards
    durations += [1600] * (len(cards) - 1) + [3500]
    frames.append(cited_frame(pred, a.cls))
    durations.append(3500)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"{a.out}: {len(frames)} frames, {Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
