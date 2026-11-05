"""Turn tracking event logs into 2-second window summaries and index them in Chroma."""
import glob
import json
from collections import defaultdict

import chromadb
from nuscenes.nuscenes import NuScenes
from sentence_transformers import SentenceTransformer

WINDOW = 2.0
EMBEDDER = "BAAI/bge-small-en-v1.5"


def scene_meta(nusc=None):
    nusc = nusc or NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=False)
    meta = {}
    for s in nusc.scene:
        log = nusc.get("log", s["log_token"])
        meta[s["name"]] = {"location": log["location"], "description": s["description"],
                           "night": "night" in s["description"].lower()}
    return meta


def window_docs(events, meta):
    tracks, frames = defaultdict(lambda: defaultdict(set)), defaultdict(list)
    for e in events:
        key = (e["scene"], int(e["t"] // WINDOW))
        tracks[key][e["class"]].add(e["track_id"])
        frames[key].append(e["frame"])
    docs = []
    for (scene, w), by_cls in sorted(tracks.items()):
        m, t0 = meta[scene], w * WINDOW
        # Embedded text: description, place, day/night and classes present. No IDs or counts: they are
        # noise for semantic matching (and counts are answered exactly by the count tool, not retrieval).
        present = sorted(by_cls, key=lambda c: -len(by_cls[c]))
        text = (f"{m['description']}. {m['location'].replace('-', ' ')}, {'night' if m['night'] else 'day'}. "
                f"Objects: {', '.join(c.replace('_', ' ') for c in present)}.")
        fr = sorted(frames[(scene, w)])
        docs.append({"id": f"{scene}_{w:03d}", "text": text,
                     "meta": {"scene": scene, "location": m["location"], "night": m["night"],
                              "t_start": t0, "t_end": t0 + WINDOW, "frame": fr[len(fr) // 2],
                              "classes": ",".join(sorted(by_cls)),
                              "counts": json.dumps({c: len(ids) for c, ids in sorted(by_cls.items())}),
                              "track_ids": json.dumps({c: sorted(ids) for c, ids in sorted(by_cls.items())})}})
    return docs


def main():
    events = [json.loads(l) for p in sorted(glob.glob("data/tracks/scene-*.jsonl")) for l in open(p)]
    docs = window_docs(events, scene_meta())
    model = SentenceTransformer(EMBEDDER)
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient(path="data/chroma")
    if "windows" in [c.name for c in client.list_collections()]:
        client.delete_collection("windows")
    col = client.create_collection("windows", metadata={"hnsw:space": "cosine"})
    col.add(ids=[d["id"] for d in docs], documents=[d["text"] for d in docs],
            embeddings=emb, metadatas=[d["meta"] for d in docs])
    print(f"{len(events)} events -> {len(docs)} windows indexed in data/chroma")
    print("example:", docs[len(docs) // 2]["text"])

    for q, where in [("buses at a bus stop at night", None),
                     ("many pedestrians crossing", {"location": "boston-seaport"})]:
        r = col.query(query_embeddings=model.encode([q], normalize_embeddings=True).tolist(),
                      n_results=3, where=where)
        print(f"\nQ: {q}  filter={where}")
        for doc, dist in zip(r["documents"][0], r["distances"][0]):
            print(f"  {1 - dist:.2f}  {doc[:150]}")


if __name__ == "__main__":
    main()