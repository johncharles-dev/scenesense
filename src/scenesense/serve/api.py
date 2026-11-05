"""HTTP API over the agent, the VLM and the tracking-log counts.

uv run uvicorn scenesense.serve.api:app --port 8080

The VLM and the agent LLM are separate vLLM servers (SCENESENSE_VLM_URL, SCENESENSE_LLM_URL);
this process only orchestrates, so it needs no GPU.
"""
import io
import time

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from scenesense.agent import loop, tools

app = FastAPI(title="scenesense", version="0.1.0",
              description="Scene-query agent, grounded VQA and exact counts over nuScenes-mini tracking logs.")

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    max_steps: int = Field(8, ge=1, le=12)
    include_trace: bool = True


def _probe(url):
    try:
        r = httpx.get(f"{url}/models", timeout=2)
        r.raise_for_status()
        return {"ok": True, "models": [m["id"] for m in r.json()["data"]]}
    except Exception as e:                                # report, don't raise: health shows what is down
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/health")
def health():
    """Liveness of this API plus reachability of both model servers and the local data."""
    checks = {"vlm": _probe(tools.VLM_URL), "llm": _probe(loop.LLM_URL),
              "tracks": {"ok": bool(tools._scene_names()), "scenes": len(tools._scene_names())}}
    status = "ok" if all(c["ok"] for c in checks.values()) else "degraded"
    return {"status": status, **checks}


@app.post("/ask")
async def ask(req: AskRequest):
    """Run the tool-calling agent on one question; returns the answer, the tool trace and latency."""
    try:
        res = await run_in_threadpool(loop.run, req.question, req.max_steps, False)
    except Exception as e:                                # model server down or failing
        raise HTTPException(503, f"agent failed: {type(e).__name__}: {e}") from e
    if not req.include_trace:
        res.pop("trace")
    return res


@app.post("/vqa")
async def vqa(image: UploadFile = File(...), question: str = Form(...)):
    """Ask the fine-tuned VLM one question in a trained form about an uploaded image."""
    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        width, height, fmt = img.width, img.height, (img.format or "JPEG").lower()
    except UnidentifiedImageError as e:
        raise HTTPException(400, "not a readable image") from e
    try:
        tools.question_family(question)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    t0 = time.perf_counter()
    try:
        res = await run_in_threadpool(tools.vlm_on_image, data, question, width, height, f"image/{fmt}")
    except Exception as e:
        raise HTTPException(503, f"VLM failed: {type(e).__name__}: {e}") from e
    return {**res, "width": width, "height": height, "latency_s": round(time.perf_counter() - t0, 3)}


@app.get("/count")
def count(object_class: str, scene: str | None = None, location: str | None = None,
          night: bool | None = None, t_start: float = 0.0, t_end: float = 1e9):
    """Exact distinct-track count from the tracking log (no model involved)."""
    try:
        return tools.count(object_class, scene=scene, location=location, night=night,
                           t_start=t_start, t_end=t_end)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
