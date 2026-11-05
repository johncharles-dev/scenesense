"""Agent tools over the tracking logs, the Chroma window index and the served VLM.

Facts (counts, comparisons) come from the event log; retrieval only finds moments.
Every tool raises ValueError with a readable message on bad input; the loop sends it back to the model.
"""
import base64
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from scenesense.agent.index import EMBEDDER, scene_meta
from scenesense.vqa.format import INSTRUCT, parse_box

TRACKS_DIR = Path("data/tracks")
CHROMA_DIR = "data/chroma"
NUSC_ROOT = "data/nuscenes"
VLM_URL = os.environ.get("SCENESENSE_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL = "scenesense-vlm"
CLASSES = ["car", "truck", "bus", "trailer", "construction_vehicle",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone", "barrier"]
SYNONYMS = {"person": "pedestrian", "people": "pedestrian", "persons": "pedestrian",
            "buses": "bus", "cone": "traffic_cone", "cones": "traffic_cone",
            "motorbike": "motorcycle", "motorbikes": "motorcycle", "bike": "bicycle", "bikes": "bicycle",
            "vehicle_construction": "construction_vehicle"}
ID_SWITCH_NOTE = ("Counts are distinct track IDs; ID switches (~7% of tracks) can split one object "
                  "into two IDs, so counts may be slightly inflated.")


# ---------- cached resources ----------

@lru_cache(maxsize=None)
def _events(scene):
    path = TRACKS_DIR / f"{scene}.jsonl"
    return [json.loads(line) for line in open(path)]


def _scene_names():
    return sorted(p.stem for p in TRACKS_DIR.glob("scene-*.jsonl"))


@lru_cache(maxsize=1)
def _nusc():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version="v1.0-mini", dataroot=NUSC_ROOT, verbose=False)


@lru_cache(maxsize=1)
def _meta():
    return scene_meta(_nusc())


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_DIR).get_collection("windows")


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDER, device="cpu")   # GPU belongs to the vLLM servers


@lru_cache(maxsize=None)
def _frames(scene):
    from scenesense.track.run_tracking import scene_frames
    nusc = _nusc()
    rec = next(s for s in nusc.scene if s["name"] == scene)
    return scene_frames(nusc, rec)


@lru_cache(maxsize=1)
def _vlm_client():
    from openai import OpenAI
    return OpenAI(base_url=VLM_URL, api_key="none", timeout=60)


# ---------- validation ----------

def check_scene(scene):
    names = _scene_names()
    s = str(scene).strip().lower()
    if re.fullmatch(r"\d{1,4}", s):                       # "796" -> "scene-0796"
        s = f"scene-{int(s):04d}"
    if s not in names:
        raise ValueError(f"Unknown scene {scene!r}. Valid scenes: {', '.join(names)}. "
                         "Call list_scenes to see locations.")
    return s


def check_class(name):
    c = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    c = SYNONYMS.get(c, c)
    if c not in CLASSES and c.endswith("s") and c[:-1] in CLASSES:
        c = c[:-1]                                        # cars -> car
    if c not in CLASSES:
        raise ValueError(f"Unknown object class {name!r}. Valid classes: {', '.join(CLASSES)}.")
    return c


def check_location(location):
    locs = sorted({m["location"] for m in _meta().values()})
    if location not in locs:
        raise ValueError(f"Unknown location {location!r}. Valid locations: {', '.join(locs)}.")
    return location


# ---------- tools ----------

def list_scenes():
    return [{"scene": s, "location": m["location"], "night": m["night"], "description": m["description"]}
            for s, m in sorted(_meta().items()) if s in _scene_names()]


def search(query, location=None, night=None, k=5):
    filters = []
    if location is not None:
        filters.append({"location": check_location(location)})
    if night is not None:
        filters.append({"night": bool(night)})
    where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}
    k = max(1, min(int(k), 20))
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    r = _collection().query(query_embeddings=emb, n_results=k, where=where)
    # Return where/when and which classes are present, but not the per-window counts embedded in the indexed
    # text: retrieval finds moments, counts come from `count` (the LLM otherwise quotes window counts as facts).
    meta = _meta()
    return [{"rank": i + 1, "scene": m["scene"], "t_start": m["t_start"], "t_end": m["t_end"],
             "location": m["location"], "night": m["night"], "objects_present": m["classes"].split(","),
             "scene_description": meta[m["scene"]]["description"], "score": round(1 - d, 3)}
            for i, (m, d) in enumerate(zip(r["metadatas"][0], r["distances"][0]))]


def _count_one(scene, cls, t_start, t_end):
    ids = {e["track_id"] for e in _events(scene) if e["class"] == cls and t_start <= e["t"] <= t_end}
    duration = max(e["t"] for e in _events(scene))
    return {"scene": scene, "t_start": t_start, "t_end": min(t_end, duration), "count": len(ids)}


def count(object_class, scene=None, location=None, night=None, t_start=0, t_end=1e9):
    """Distinct track IDs of a class. One scene, or every scene matching location/night with a total."""
    cls = check_class(object_class)
    t_start, t_end = float(t_start), float(t_end)
    if t_end < t_start:
        raise ValueError(f"t_end ({t_end}) is before t_start ({t_start}).")
    if scene is not None:
        if location is not None or night is not None:
            raise ValueError("Give either scene or location/night filters, not both.")
        return {"object_class": cls, **_count_one(check_scene(scene), cls, t_start, t_end),
                "note": ID_SWITCH_NOTE}
    if location is not None:
        location = check_location(location)
    scenes = [s for s in _scene_names()
              if (location is None or _meta()[s]["location"] == location)
              and (night is None or _meta()[s]["night"] == bool(night))]
    if not scenes:
        raise ValueError("No scenes match these filters. Call list_scenes.")
    per = [_count_one(s, cls, t_start, t_end) for s in scenes]
    top = max(c["count"] for c in per)
    return {"object_class": cls, "location": location, "night": night, "per_scene": per,
            "total": sum(c["count"] for c in per),
            "most": [c["scene"] for c in per if c["count"] == top], "note": ID_SWITCH_NOTE}


def get_frame(scene, t):
    scene, t = check_scene(scene), float(t)
    frames = _frames(scene)
    t0 = frames[0]["timestamp"]
    i = min(range(len(frames)), key=lambda j: abs((frames[j]["timestamp"] - t0) / 1e6 - t))
    return {"scene": scene, "frame": i, "t": round((frames[i]["timestamp"] - t0) / 1e6, 3),
            "image": str(Path(NUSC_ROOT) / frames[i]["filename"]),
            "width": frames[i]["width"], "height": frames[i]["height"]}


# The question forms the VLM was fine-tuned on (scenesense.vqa.build_vqa) -> instruction family.
# Named groups capture the object(s), which must be one of the 10 trained classes.
QUESTION_FORMS = [
    ("count", re.compile(r"^how many (?P<a>[\w\s-]+?) (are )?in the image\?$", re.I)),
    ("exists", re.compile(r"^is there an? (?P<a>[\w\s-]+) in the image\?$", re.I)),
    ("left_right", re.compile(r"^is the (?P<a>[\w\s-]+) to the left or right of the (?P<b>[\w\s-]+)\?$", re.I)),
    ("ground", re.compile(r"^where is the (?:(?:leftmost|rightmost|largest|smallest|nearest) )?"
                          r"(?P<a>[\w\s-]+?)\?( give its bounding box\.)?$", re.I)),
]


def question_family(question):
    q = " ".join(str(question).split())
    for fam, pattern in QUESTION_FORMS:
        m = pattern.match(q)
        if m:
            for obj in filter(None, (m.groupdict().get("a"), m.groupdict().get("b"))):
                try:
                    check_class(obj)
                except ValueError:
                    raise ValueError(f"The VLM was only trained on these object classes: {', '.join(CLASSES)}; "
                                     f"{obj!r} is not one of them, so it cannot answer reliably.") from None
            return fam, q
    raise ValueError(
        f"Unsupported VLM question {question!r}. Use exactly one of: 'How many <objects> are in the image?', "
        "'Is there a <object> in the image?', 'Is the <A> to the left or right of the <B>?', "
        "'Where is the <object>? Give its bounding box.'")


def ask_vlm(scene, t, question):
    fam, q = question_family(question)
    if fam == "ground" and not q.lower().endswith("give its bounding box."):
        q += " Give its bounding box."
    fr = get_frame(scene, t)
    b64 = base64.b64encode(Path(fr["image"]).read_bytes()).decode()
    resp = _vlm_client().chat.completions.create(
        model=VLM_MODEL, temperature=0, max_tokens=32,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": f"{q} {INSTRUCT[fam]}"}]}])
    answer = (resp.choices[0].message.content or "").strip()
    out = {"scene": fr["scene"], "frame": fr["frame"], "t": fr["t"], "question": q, "answer": answer}
    if fam == "ground":                                   # only expose pixels, not Qwen's 0-1000 grid
        box = parse_box(answer, fr["width"], fr["height"])
        out["answer"] = [round(v) for v in box] if box else "no box returned"
        out["answer_format"] = f"[x1, y1, x2, y2] in pixels of a {fr['width']}x{fr['height']} image"
    return out


FUNCS = {"list_scenes": list_scenes, "search": search, "count": count,
         "get_frame": get_frame, "ask_vlm": ask_vlm}

_SCENE = {"type": "string", "description": "Scene name exactly as returned by list_scenes, e.g. 'scene-0796'."}
_CLASS = {"type": "string", "enum": CLASSES, "description": "Object class."}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_scenes",
        "description": "List all 10 scenes with location, night flag and a short description. "
                       "Call this first to resolve place names (e.g. Queenstown, Boston) to scene names.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Semantic search over 2-second window summaries of the tracking logs. Use it ONLY to find "
                       "moments ('find a moment where ...'); never use its text to answer counting questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "location": {"type": "string", "enum": ["boston-seaport", "singapore-hollandvillage",
                                                    "singapore-onenorth", "singapore-queenstown"],
                         "description": "Optional location filter."},
            "night": {"type": "boolean", "description": "Optional: true for night scenes only, false for day."},
            "k": {"type": "integer", "description": "Number of results (default 5, max 20)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "count",
        "description": "Exact number of distinct tracked objects of one class, computed from the tracking log. "
                       "ALWAYS use this for any how-many, total or which-has-most question; never estimate counts "
                       "yourself or read them off search results. Give `scene` for one scene, or give "
                       "`location` and/or `night` (or neither, for all 10 scenes) to get a per-scene breakdown "
                       "with the total and the scene(s) with the most. Omit t_start/t_end for whole scenes.",
        "parameters": {"type": "object", "properties": {
            "object_class": _CLASS,
            "scene": {**_SCENE, "description": _SCENE["description"] + " Omit to count over several scenes."},
            "location": {"type": "string", "enum": ["boston-seaport", "singapore-hollandvillage",
                                                    "singapore-onenorth", "singapore-queenstown"],
                         "description": "Count every scene at this location (instead of scene)."},
            "night": {"type": "boolean", "description": "Only night (true) or day (false) scenes."},
            "t_start": {"type": "number", "description": "Start time in seconds (default 0)."},
            "t_end": {"type": "number", "description": "End time in seconds (default: end of scene)."}},
            "required": ["object_class"]}}},
    {"type": "function", "function": {
        "name": "get_frame",
        "description": "Nearest camera frame (front camera) to time t in a scene: frame index, exact time, image path.",
        "parameters": {"type": "object", "properties": {
            "scene": _SCENE, "t": {"type": "number", "description": "Time in seconds from scene start."}},
            "required": ["scene", "t"]}}},
    {"type": "function", "function": {
        "name": "ask_vlm",
        "description": "Ask the fine-tuned vision-language model about the camera frame at time t. Only these "
                       "question forms are allowed: 'How many <objects> are in the image?', "
                       "'Is there a <object> in the image?', 'Is the <A> to the left or right of the <B>?', "
                       "'Where is the <object>? Give its bounding box.' (box in pixels).",
        "parameters": {"type": "object", "properties": {
            "scene": _SCENE, "t": {"type": "number", "description": "Time in seconds from scene start."},
            "question": {"type": "string", "description": "One question in an allowed form."}},
            "required": ["scene", "t", "question"]}}},
]
