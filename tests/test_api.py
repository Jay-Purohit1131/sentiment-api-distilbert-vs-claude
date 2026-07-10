"""Tests for the FastAPI sentiment service.

Importing `app` loads the fine-tuned model into memory (via predict.py), so
these tests require the ./distilbert-imdb folder present and take a few seconds
to start up.
"""
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_shape():
    r = client.post("/predict", json={"text": "An absolute masterpiece from start to finish."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"positive", "negative"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_clear_negative():
    r = client.post("/predict", json={"text": "A boring, predictable mess. Awful film."})
    assert r.status_code == 200
    assert r.json()["label"] == "negative"


def test_predict_empty_text_rejected():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422        # Pydantic min_length=1 rejects it


def test_predict_missing_field_rejected():
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_batch():
    r = client.post("/predict-batch", json={"texts": ["Loved it.", "Hated it."]})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert all("label" in item and "confidence" in item for item in body)