"""Small JSON / JSON-lines helpers shared by the pipeline, eval and agent code."""
import json
from pathlib import Path


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path, **kwargs):
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)


def read_jsonl(path):
    """One JSON object per line; blank lines are skipped."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r, **kwargs) + "\n" for r in rows)
