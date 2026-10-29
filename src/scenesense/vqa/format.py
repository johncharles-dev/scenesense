"""Prompts, training targets and answer scoring for Qwen2-VL on the scenesense VQA set."""
import re

INSTRUCT = {
    "count": "Answer with a number only.",
    "exists": "Answer yes or no.",
    "left_right": "Answer left or right.",
    "ground": "Answer with the bounding box as (x1,y1),(x2,y2) in 0-1000 coordinates.",
}
WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8"}


def family(qtype):
    """exists_yes/exists_no -> exists; ground/ground_ref -> ground."""
    if qtype.startswith("exists"):
        return "exists"
    if qtype.startswith("ground"):
        return "ground"
    return qtype


def prompt(row):
    return f"{row['question']} {INSTRUCT[family(row['type'])]}"


def to_qwen_box(box, w, h):
    x1, y1, x2, y2 = box
    return f"({round(x1 / w * 1000)},{round(y1 / h * 1000)}),({round(x2 / w * 1000)},{round(y2 / h * 1000)})"


def target(row):
    """The answer text the model is trained to produce."""
    if family(row["type"]) == "ground":
        return "<|box_start|>" + to_qwen_box(row["box"], row["width"], row["height"]) + "<|box_end|>"
    return row["answer"]


def parse_box(text, w, h):
    """First four numbers, read as Qwen 0-1000 coordinates, converted to pixels."""
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    return [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(row, text):
    t = text.strip().lower()
    f = family(row["type"])
    if f == "count":
        for word, digit in WORDS.items():
            t = re.sub(rf"\b{word}\b", digit, t)
        m = re.search(r"\d+", t)
        return m is not None and m.group() == row["answer"]
    if f in ("exists", "left_right"):
        m = re.search(r"\b(yes|no)\b" if f == "exists" else r"\b(left|right)\b", t)
        return m is not None and m.group(1) == row["answer"]
    box = parse_box(text, row["width"], row["height"])
    return box is not None and iou(box, row["box"]) >= 0.5