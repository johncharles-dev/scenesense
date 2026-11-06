"""Hand-written tool-calling loop over the OpenAI-compatible API served by vLLM.

uv run python -m scenesense.agent.loop "How many buses appear in the Queenstown scenes?"
"""
import argparse
import json
import os
import time

from openai import OpenAI

from scenesense.agent.tools import CLASSES, FUNCS, TOOLS, list_scenes

LLM_URL = os.environ.get("SCENESENSE_LLM_URL", "http://localhost:8002/v1")
LLM_MODEL = "agent-llm"

SYSTEM = """You answer questions about 10 driving scenes (nuScenes mini, ~20 s each) using tools over \
object-tracking logs and a fine-tuned vision-language model (VLM). These are ALL the scenes that exist:
{scenes}
Object classes (the only ones tracked): {classes}.

First decide which kind of question it is, then follow that procedure exactly.

A. COUNT or COMPARE ("how many", "total", "which scene has the most/fewest", "more X or Y"): call count. \
For one scene pass scene; for several scenes (a location, night/day, all) make ONE call with location \
and/or night and read per_scene, total and most. Count whole scenes unless a time range is given. Do NOT \
call search for these, even if the question describes a behaviour.
B. FIND A MOMENT ("find a moment/time where ..."): call search. Take the rank-1 result (it is the best \
match). If the question also asks what is in that frame, call ask_vlm on the rank-1 scene with a time \
inside its window. Report that scene and window.
C. A SPECIFIC FRAME ("in scene X at t s, is there / where is / left or right"): call ask_vlm. Before \
"Where is the X?", first ask "Is there a X in the image?"; if no, say the object is not visible.

Always:
- Never state a number, scene or time that did not come from a tool result. Cite each fact as \
(scene-0796, 4-6 s); for whole-scene counts use the scene's full range from the count result.
- The tools measure only object class, presence, count, position in the image and time. Anything else \
(behaviour such as crossing or turning, speed, direction of motion, colour, weather) is NOT measurable. \
When a question asks for it: (1) say in one sentence that the tools cannot measure it, (2) give the \
closest measurable proxy from a tool, (3) label it explicitly as a proxy, e.g. "Proxy: pedestrians \
tracked, not pedestrians crossing".
- If the question names a place, scene or object class not listed above, say there is no data for it. \
Never substitute another place or class.
- Keep the final answer to one to three sentences."""


def system_prompt():
    scenes = "\n".join(f"- {s['scene']}: {s['location']}, {'night' if s['night'] else 'day'}, {s['description']}"
                       for s in list_scenes())
    return SYSTEM.format(scenes=scenes, classes=", ".join(CLASSES))


def call_tool(name, raw_args):
    """Run one tool call; any problem comes back as {"error": ...} so the model can recover."""
    if name not in FUNCS:
        return {"error": f"Unknown tool {name!r}. Available: {', '.join(FUNCS)}."}, False
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise TypeError("arguments must be a JSON object")
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Could not parse arguments {raw_args!r}: {e}"}, False
    try:
        return FUNCS[name](**args), True
    except TypeError as e:                                # missing / unexpected argument names
        return {"error": f"Bad arguments for {name}: {e}"}, False
    except Exception as e:  # noqa: BLE001  ValueError from validation, server errors, ...
        return {"error": f"{type(e).__name__}: {e}"}, False


def _short(obj, n=300):
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ..."


def run(question, max_steps=8, verbose=True, client=None):
    """Answer one question. Returns answer, per-call trace, number of LLM steps and latency."""
    client = client or OpenAI(base_url=LLM_URL, api_key="none", timeout=120)
    messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": question}]
    trace, t0 = [], time.perf_counter()
    if verbose:
        print(f"Q: {question}")
    answer, steps = None, 0
    for step in range(1, max_steps + 1):
        steps = step
        last = step == max_steps                          # final step: force a text answer
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0, max_tokens=1024,
            tools=TOOLS, tool_choice="none" if last else "auto")
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            answer = (msg.content or "").strip()
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            result, ok = call_tool(c.function.name, c.function.arguments)
            trace.append({"step": step, "tool": c.function.name, "args": c.function.arguments,
                          "ok": ok, "result": result})
            if verbose:
                print(f"  [{step}] {c.function.name}({c.function.arguments})")
                print(f"      -> {_short(result)}")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
    latency = time.perf_counter() - t0
    if answer is None:
        answer = "I could not reach an answer within the step limit."
    if verbose:
        print(f"A: {answer}\n   ({steps} LLM steps, {len(trace)} tool calls, {latency:.1f} s)")
    return {"question": question, "answer": answer, "trace": trace, "steps": steps,
            "n_tool_calls": len(trace), "latency_s": round(latency, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--max-steps", type=int, default=8)
    a = ap.parse_args()
    run(a.question, a.max_steps)


if __name__ == "__main__":
    main()
