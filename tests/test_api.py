"""API tests that run without the model servers (VLM calls are stubbed)."""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenesense.agent import tools
from scenesense.serve import api

client = TestClient(api.app)


@pytest.fixture
def fake_logs(tmp_path, monkeypatch):
    rows = [{"scene": "scene-0001", "t": t, "track_id": i, "class": c}
            for i, c, t in [(1, "car", 0.0), (2, "car", 4.0), (3, "bus", 2.0)]]
    (tmp_path / "scene-0001.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(tools, "TRACKS_DIR", tmp_path)
    tools._events.cache_clear()
    yield
    tools._events.cache_clear()


def _jpeg(w=64, h=32):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, "JPEG")
    return buf.getvalue()


def test_count_ok_and_validation(fake_logs):
    r = client.get("/count", params={"object_class": "cars", "scene": "scene-0001"})
    assert r.status_code == 200 and r.json()["count"] == 2
    r = client.get("/count", params={"object_class": "dog", "scene": "scene-0001"})
    assert r.status_code == 422 and "Unknown object class" in r.json()["detail"]


def test_vqa_rejects_untrained_question_before_calling_vlm(monkeypatch):
    monkeypatch.setattr(tools, "vlm_on_image", lambda *a, **k: pytest.fail("VLM must not be called"))
    r = client.post("/vqa", files={"image": ("x.jpg", _jpeg(), "image/jpeg")},
                    data={"question": "What colour is the bus?"})
    assert r.status_code == 422 and "Unsupported VLM question" in r.json()["detail"]


def test_vqa_rejects_non_image():
    r = client.post("/vqa", files={"image": ("x.txt", b"hello", "text/plain")},
                    data={"question": "Is there a bus in the image?"})
    assert r.status_code == 400


def test_vqa_passes_image_size_to_vlm(monkeypatch):
    seen = {}

    def fake(data, question, width, height, mime):
        seen.update(width=width, height=height, mime=mime)
        return {"question": question, "question_type": "exists", "answer": "yes"}

    monkeypatch.setattr(tools, "vlm_on_image", fake)
    r = client.post("/vqa", files={"image": ("x.jpg", _jpeg(64, 32), "image/jpeg")},
                    data={"question": "Is there a bus in the image?"})
    assert r.status_code == 200 and r.json()["answer"] == "yes"
    assert seen == {"width": 64, "height": 32, "mime": "image/jpeg"}


def test_health_reports_degraded_when_servers_down(monkeypatch):
    monkeypatch.setattr(tools, "VLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(api.loop, "LLM_URL", "http://127.0.0.1:9/v1")
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and not r.json()["vlm"]["ok"]
