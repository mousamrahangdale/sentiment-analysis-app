"""
Basic smoke tests.

`test_health` always runs. The two prediction tests are skipped automatically
if their dependency isn't available (no checkpoint copied yet / Ollama not
running) so `pytest` stays green on a fresh checkout — remove the skip once
you've completed setup.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "distilbert_loaded" in body
    assert "ollama_reachable" in body


def test_local_model_prediction():
    health = client.get("/api/v1/health").json()
    if not health["distilbert_loaded"]:
        pytest.skip("DistilBERT checkpoint not present yet — see saved_models/distilbert_sentiment/README.md")

    resp = client.post(
        "/api/v1/sentiment/analyze",
        json={"text": "This works perfectly, I'm really happy with it.", "source": "local_model"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("Bad", "Neutral", "Good")
    assert 0.0 <= body["confidence"] <= 1.0


def test_llm_prediction():
    health = client.get("/api/v1/health").json()
    if not health["ollama_reachable"]:
        pytest.skip("Ollama not reachable — run `ollama serve` and pull a model first.")

    resp = client.post(
        "/api/v1/sentiment/analyze",
        json={"text": "This is the worst experience I've had with an app.", "source": "llm"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("Bad", "Neutral", "Good")


def test_empty_text_rejected():
    resp = client.post("/api/v1/sentiment/analyze", json={"text": "   ", "source": "local_model"})
    assert resp.status_code == 422
