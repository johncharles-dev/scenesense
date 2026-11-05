"""Evaluate the agent on data/agent_eval/questions.jsonl, and compare two runs with paired bootstrap CIs.

uv run python -m scenesense.agent.eval_agent run --name qwen7b
uv run python -m scenesense.agent.eval_agent compare qwen7b qwen14b_awq

Scoring is rule-based (regexes over the final answer) so it is transparent and testable; every rule is
spelled out below and was audited by hand on the actual answers (see the step memo).
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

QUESTIONS = Path("data/agent_eval/questions.jsonl")
SCENE = re.compile(r"scene-\d{4}")
NUM = r"\d+(?:\.\d+)?"
TIME_RANGE = re.compile(rf"({NUM})\s*(?:-|–|to|and)\s*({NUM})\s*(?:s\b|sec|seconds)")
TIME_POINT = re.compile(rf"({NUM})\s*(?:s\b|sec|seconds)")
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10}
REFUSAL = re.compile(
    r"cannot|can't|can not|unable|not able|not possible|not available|no data|no information|not measurable|"
    r"not (?:in|part of) the (?:logs|data|tools|available)|does not exist|doesn't exist|not one of|"
    r"not tracked|isn't tracked|are not tracked|outside|only (?:runs|lasts|covers)|not included|no such|"
    r"not a valid|invalid|not supported|not listed|no scenes?\b|"
    r"(?:do|does) not (?:measure|provide|track|include|contain)", re.I)
COMPARATIVE = re.compile(r"\b(more|most|fewer|fewest|less|least|higher|highest|lower|lowest|greater)\b", re.I)


# ---------- parsing ----------

def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _times(text):
    ranges = [(float(a), float(b)) for a, b in TIME_RANGE.findall(text)]
    return ranges or [(float(a), float(a)) for a in TIME_POINT.findall(TIME_RANGE.sub(" ", text))]


def citations(text):
    """(scene, t0, t1) for each scene mention; times are taken from the same sentence (t0=t1=None if none)."""
    out = []
    for s in sentences(text):
        scenes = SCENE.findall(s)
        if not scenes:
            continue
        ranges = _times(SCENE.sub(" ", s))
        for sc in scenes:
            out += [(sc, a, b) for a, b in ranges] or [(sc, None, None)]
    return out


# ---------- citation support against the tracking log ----------

CLASS_PHRASES = {"traffic cone": "traffic_cone", "construction vehicle": "construction_vehicle",
                 "pedestrian": "pedestrian", "motorcycle": "motorcycle", "bicycle": "bicycle", "trailer": "trailer",
                 "barrier": "barrier", "truck": "truck", "bus": "bus", "car": "car"}
NEGATION = re.compile(r"\b(no|not|none|zero|0|isn't|aren't|absent)\b", re.I)


def question_class(question):
    """The object class a question is about (longest phrase first, so 'traffic cone' wins over 'car')."""
    q = question.lower()
    for phrase in sorted(CLASS_PHRASES, key=len, reverse=True):
        if re.search(rf"\b{phrase}(e?s)?\b", q):
            return CLASS_PHRASES[phrase]
    return None


def claims(text):
    """(scene, t0, t1, present) per scene mention.

    Polarity: the context of a mention is the text between the nearest mentions of a DIFFERENT scene in the
    same sentence, so 'scene A has no X, whereas scene B has 10' gives A absent, B present, and a repeated
    '(scene A, 0-19 s)' shares A's context. A bare citation such as '(scene-0757, 5-10 s)' with no words around
    it inherits the last claim made about that scene. Times come from the text after the mention (up to the
    next one), else from the sentence if it has exactly one time; t0=None if the mention has no time."""
    out, last = [], {}
    for s in sentences(text):
        ms = list(SCENE.finditer(s))
        for i, m in enumerate(ms):
            j = i
            while j > 0 and ms[j - 1].group() == m.group():
                j -= 1
            k = i
            while k + 1 < len(ms) and ms[k + 1].group() == m.group():
                k += 1
            ctx = s[ms[j - 1].end() if j else 0: ms[k + 1].start() if k + 1 < len(ms) else len(s)]
            stripped = TIME_POINT.sub(" ", TIME_RANGE.sub(" ", SCENE.sub(" ", ctx)))
            if re.search(r"[a-z]{2,}", stripped, re.I):
                present = not NEGATION.search(stripped)
            else:
                present = last.get(m.group(), True)
            last[m.group()] = present
            after = s[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(s)]
            sent_times = _times(SCENE.sub(" ", s))
            times = _times(after) or (sent_times if len(sent_times) == 1 else [])
            out += [(m.group(), a, b, present) for a, b in times] or [(m.group(), None, None, present)]
    return out


def log_supported(q, answer, events):
    """True if the answer has >=1 time-stamped citation and, for every one, the log agrees with the claim:
    the question's class has (or, for a negative claim, has no) track in that scene within +-1 s of the time."""
    cls = question_class(q["question"])
    timed = [c for c in claims(answer) if c[1] is not None]
    if cls is None or not timed:
        return False
    for sc, a, b, present in timed:
        if sc not in events:
            return False
        seen = any(e["class"] == cls and a - 1 <= e["t"] <= b + 1 for e in events[sc])
        if seen != present:
            return False
    return True


_EVENTS = {}


def tracking_events():
    if not _EVENTS:
        for p in sorted(Path("data/tracks").glob("scene-*.jsonl")):
            _EVENTS[p.stem] = [json.loads(line) for line in open(p)]
    return _EVENTS


def numbers(text):
    """Integers stated as answers: scene IDs, time expressions, image sizes and box coordinates removed."""
    t = SCENE.sub(" ", text)
    t = TIME_RANGE.sub(" ", t)
    t = TIME_POINT.sub(" ", t)
    t = re.sub(r"\d+\s*x\s*\d+|\[[^\]]*\]|~?\d+\s*%", " ", t)
    nums = [int(float(n)) for n in re.findall(NUM, t)]
    nums += [v for w, v in WORDS.items() if re.search(rf"\b{w}\b", t, re.I)]
    return nums


def yes_no(text):
    t = text.lower()
    if re.search(r"\b(no|not|isn't|is no|are no|absent|none)\b", t):
        return "no"
    if re.search(r"\b(yes|there is an?|is present|is visible|can be seen|detected)\b", t):
        return "yes"
    return None


def box(text):
    m = re.search(rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]", text)
    return [float(v) for v in m.groups()] if m else None


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps(c, windows):
    sc, a, b = c
    return a is not None and any(sc == s and a <= t1 and b >= t0 for s, t0, t1 in windows)


# ---------- scoring ----------

def correct(q, answer):
    """Answer accuracy for one question (bool)."""
    typ, truth = q["type"], q["answer"]
    if typ == "unanswerable":
        return bool(REFUSAL.search(answer))
    if typ in ("count", "total"):
        if truth == 0 and re.search(r"\b(no|none|zero|not any)\b", answer, re.I):
            return True
        return truth in numbers(answer)
    if typ == "compare":
        # The winner is the first candidate scene in the LAST sentence that states the comparison
        # ("A has 0, whereas B has 10. Therefore more in B."); fall back to the first scene named.
        comp = [s for s in sentences(answer) if COMPARATIVE.search(s) and
                any(sc in q["scenes"] for sc in SCENE.findall(s))]
        named = [sc for sc in SCENE.findall(comp[-1] if comp else answer) if sc in q["scenes"]]
        return bool(named) and named[0] == truth
    if typ in ("moment", "upstream"):
        return any(overlaps(c, truth) for c in citations(answer))
    if typ == "vqa":
        return yes_no(answer) == truth
    if typ == "ground":
        b = box(answer)
        return b is not None and iou(b, truth) >= 0.5
    raise ValueError(typ)


def citation_ok(q, answer):
    """(present, correct): a scene is cited, and every cited scene/time supports the claim."""
    cits = citations(answer)
    if not cits:
        return False, False
    if any(sc not in q["scenes"] for sc, _, _ in cits):
        return True, False
    if q["type"] in ("moment", "upstream"):
        return True, all(overlaps(c, q["answer"]) for c in cits)
    if q["type"] in ("vqa", "ground"):
        return True, any(a is not None and a - 1 <= q["t"] <= b + 1 for _, a, b in cits)
    return True, True


def score(q, pred):
    ans = pred.get("answer") or ""
    calls = pred.get("trace", [])
    present, cit = citation_ok(q, ans) if q["type"] != "unanswerable" else (None, None)
    answerable = q["type"] != "unanswerable"
    timed = any(a is not None for _, a, _ in citations(ans)) if answerable else None
    supported = log_supported(q, ans, tracking_events()) if answerable else None
    return {"correct": correct(q, ans), "refused": bool(REFUSAL.search(ans)),
            "citation_present": present, "citation_with_time": timed, "citation_correct": cit,
            "citation_log_supported": supported,
            "n_calls": len(calls), "n_ok": sum(c["ok"] for c in calls),
            "n_parse_err": sum(not c["ok"] and not str(c["result"].get("error", "")).startswith("ValueError")
                               for c in calls)}


# ---------- statistics ----------

def bootstrap(values, n=10_000, seed=0, weights=None):
    """Mean (or ratio of sums with weights) and 95% percentile CI, resampling questions."""
    if len(values) == 0:
        return None
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    stats = (v[idx]).sum(1) / np.maximum(w[idx].sum(1), 1e-9)
    return {"mean": float(v.sum() / w.sum()), "lo": float(np.percentile(stats, 2.5)),
            "hi": float(np.percentile(stats, 97.5)), "n": int(len(v))}


def summarize(qs, rows):
    by_id = {q["id"]: q for q in qs}
    ans = [r for r in rows if by_id[r["id"]]["type"] != "unanswerable"]
    una = [r for r in rows if by_id[r["id"]]["type"] == "unanswerable"]
    out = {
        "answer_accuracy": bootstrap([r["s"]["correct"] for r in ans]),
        "refusal_on_unanswerable": bootstrap([r["s"]["correct"] for r in una]),
        "false_refusal_on_answerable": bootstrap([r["s"]["refused"] and not r["s"]["correct"] for r in ans]),
        "citation_present": bootstrap([r["s"]["citation_present"] for r in ans]),
        "citation_with_time": bootstrap([r["s"]["citation_with_time"] for r in ans]),
        "citation_correct": bootstrap([r["s"]["citation_correct"] for r in ans]),
        "citation_log_supported": bootstrap([r["s"]["citation_log_supported"] for r in ans]),
        "tool_call_validity": bootstrap([r["s"]["n_ok"] for r in rows], weights=[r["s"]["n_calls"] for r in rows]),
        "tool_calls_total": int(sum(r["s"]["n_calls"] for r in rows)),
        "tool_calls_schema_errors": int(sum(r["s"]["n_parse_err"] for r in rows)),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_latency_s": float(np.mean([r["latency_s"] for r in rows])),
        "p50_latency_s": float(np.median([r["latency_s"] for r in rows])),
        "by_type": {},
    }
    for t in sorted({q["type"] for q in qs}):
        v = [r["s"]["correct"] for r in rows if by_id[r["id"]]["type"] == t]
        out["by_type"][t] = {"correct": int(sum(v)), "n": len(v)}
    return out


def paired(qs, a_rows, b_rows, metric, subset):
    """b - a on the same questions, paired bootstrap over questions."""
    by_id = {q["id"]: q for q in qs}
    ids = [r["id"] for r in a_rows if subset(by_id[r["id"]])]
    a = {r["id"]: r for r in a_rows}
    b = {r["id"]: r for r in b_rows}
    d = np.array([float(b[i]["s"][metric]) - float(a[i]["s"][metric]) for i in ids])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10_000, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": len(d),
            "b_only": int((d > 0).sum()), "a_only": int((d < 0).sum())}


# ---------- CLI ----------

def run_all(name, max_steps=8, questions=QUESTIONS):
    import openai

    from scenesense.agent.loop import LLM_URL, run

    qs = [json.loads(line) for line in open(questions)]
    served = openai.OpenAI(base_url=LLM_URL, api_key="none").models.list().data[0].root
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        try:
            pred = run(q["question"], max_steps=max_steps, verbose=False)
        except Exception as e:                       # server error: count as a wrong, empty answer
            pred = {"answer": "", "trace": [], "steps": 0, "latency_s": time.perf_counter() - t0,
                    "error": f"{type(e).__name__}: {e}"}
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "truth": q["answer"],
               **{k: pred.get(k) for k in ("answer", "trace", "steps", "latency_s", "error")}}
        row["s"] = score(q, pred)
        rows.append(row)
        print(f"{q['id']:3d} {q['type']:12s} {'OK ' if row['s']['correct'] else 'ERR'} "
              f"{row['steps']} steps {row['latency_s']:.1f}s | {(row['answer'] or '')[:110]}")
    Path("data/preds").mkdir(exist_ok=True)
    with open(f"data/preds/agent_eval_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    summary = {"name": name, "served_model": served, **summarize(qs, rows)}
    Path(f"reports/agent_eval_{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_type"}, indent=1))
    print(summary["by_type"])


def compare(a, b):
    qs = [json.loads(line) for line in open(QUESTIONS)]
    load = lambda n: [json.loads(line) for line in open(f"data/preds/agent_eval_{n}.jsonl")]  # noqa: E731
    ra, rb = load(a), load(b)
    for rows in (ra, rb):                            # rescore with the current rules
        by_id = {q["id"]: q for q in qs}
        for r in rows:
            r["s"] = score(by_id[r["id"]], r)
    answerable = lambda q: q["type"] != "unanswerable"  # noqa: E731
    out = {"a": a, "b": b,
           "summary": {a: summarize(qs, ra), b: summarize(qs, rb)},
           f"{b}_minus_{a}": {
               "answer_accuracy": paired(qs, ra, rb, "correct", answerable),
               "refusal_on_unanswerable": paired(qs, ra, rb, "correct", lambda q: not answerable(q)),
               "all_questions": paired(qs, ra, rb, "correct", lambda q: True),
               "citation_correct": paired(qs, ra, rb, "citation_correct", answerable),
               "citation_with_time": paired(qs, ra, rb, "citation_with_time", answerable),
               "citation_log_supported": paired(qs, ra, rb, "citation_log_supported", answerable)}}
    Path(f"reports/agent_eval_{b}_vs_{a}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out[f"{b}_minus_{a}"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--max-steps", type=int, default=8)
    r.add_argument("--questions", default=str(QUESTIONS))
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run_all(a.name, a.max_steps, Path(a.questions))
    else:
        compare(a.a, a.b)


if __name__ == "__main__":
    main()
