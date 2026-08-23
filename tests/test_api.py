"""
Tests for the sentiment analysis API.

Design notes:
- We mock `get_distilbert_service` / `get_llm_service` at the router module
  level rather than loading the real DistilBERT checkpoint or calling real
  Hugging Face / Groq APIs. This keeps tests fast, deterministic, and
  runnable in CI without secrets or GPU/model files.
- We do NOT use `with TestClient(app) as client:` (which would trigger
  FastAPI's lifespan and try to warm-load the real model) — a plain
  `TestClient(app)` skips startup/shutdown events, which is what we want
  for route-level unit tests.
- `get_settings` is wired into the route via FastAPI's `Depends(get_settings)`,
  so it CANNOT be swapped with `unittest.mock.patch("...get_settings")` —
  FastAPI resolves the dependency callable it captured at route-definition
  time, not a fresh module-attribute lookup per request. To override it we
  use `app.dependency_overrides`, which is the mechanism FastAPI provides
  for exactly this.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import app
from backend.services.distilbert_service import ModelNotFoundError

client = TestClient(app)

ANALYZE_URL = "/api/v1/sentiment/analyze"
HEALTH_URL = "/api/v1/health"


# ---------------------------------------------------------------------------
# Fixtures: canned successful responses matching each service's predict() shape
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_distilbert_result():
    return {
        "source": "local_model",
        "engine": "distilbert-finetuned",
        "label": "Good",
        "confidence": 0.97,
        "probabilities": {"Bad": 0.01, "Neutral": 0.02, "Good": 0.97},
        "reason": None,
        "latency_ms": 12.3,
        "text_length": 20,
    }


@pytest.fixture
def fake_llm_result():
    return {
        "source": "llm",
        "engine": "huggingface:cardiffnlp/twitter-roberta-base-sentiment-latest",
        "label": "Bad",
        "confidence": 0.94,
        "probabilities": {"Bad": 0.94, "Neutral": 0.05, "Good": 0.01},
        "reason": "The text expresses clear frustration.",
        "latency_ms": 1450.7,
        "text_length": 20,
    }


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Belt-and-suspenders: make sure no test leaks an override into the next one."""
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /api/v1/sentiment/analyze — local_model source
# ---------------------------------------------------------------------------

class TestAnalyzeLocalModel:
    def test_success(self, fake_distilbert_result):
        with patch("backend.routers.sentiment.get_distilbert_service") as mock_get:
            mock_get.return_value.predict.return_value = fake_distilbert_result

            resp = client.post(ANALYZE_URL, json={"text": "This app is great!", "source": "local_model"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "local_model"
        assert data["label"] == "Good"
        assert data["confidence"] == pytest.approx(0.97)
        assert data["probabilities"] == {"Bad": 0.01, "Neutral": 0.02, "Good": 0.97}
        assert data["reason"] is None
        assert data["text_length"] == fake_distilbert_result["text_length"]
        assert data["latency_ms"] == pytest.approx(fake_distilbert_result["latency_ms"])

    def test_model_not_found_returns_503(self):
        with patch("backend.routers.sentiment.get_distilbert_service") as mock_get:
            mock_get.side_effect = ModelNotFoundError("checkpoint missing at saved_models/distilbert_sentiment")

            resp = client.post(ANALYZE_URL, json={"text": "Some text", "source": "local_model"})

        assert resp.status_code == 503
        assert "checkpoint" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /api/v1/sentiment/analyze — llm source
# ---------------------------------------------------------------------------

class TestAnalyzeLLM:
    def test_success(self, fake_llm_result):
        with patch("backend.routers.sentiment.get_llm_service") as mock_get:
            mock_get.return_value.predict.return_value = fake_llm_result

            resp = client.post(ANALYZE_URL, json={"text": "This app is terrible!", "source": "llm"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "llm"
        assert data["label"] == "Bad"
        assert data["reason"] == "The text expresses clear frustration."
        assert data["engine"].startswith("huggingface:")

    def test_llm_failure_returns_502(self):
        with patch("backend.routers.sentiment.get_llm_service") as mock_get:
            mock_get.return_value.predict.side_effect = RuntimeError("HF Inference Providers unreachable")

            resp = client.post(ANALYZE_URL, json={"text": "Some text", "source": "llm"})

        assert resp.status_code == 502
        assert "hugging face" in resp.json()["detail"].lower()

    def test_reason_can_be_null_without_failing_the_request(self, fake_llm_result):
        # Reason generation is best-effort (see llm_service._generate_reason) —
        # a null reason must not be treated as an error by the route.
        degraded_result = {**fake_llm_result, "reason": None}
        with patch("backend.routers.sentiment.get_llm_service") as mock_get:
            mock_get.return_value.predict.return_value = degraded_result

            resp = client.post(ANALYZE_URL, json={"text": "Some text", "source": "llm"})

        assert resp.status_code == 200
        assert resp.json()["reason"] is None


# ---------------------------------------------------------------------------
# Request validation (schema-level, no service should even be called)
# ---------------------------------------------------------------------------

class TestAnalyzeValidation:
    def test_empty_text_returns_422(self):
        resp = client.post(ANALYZE_URL, json={"text": "   ", "source": "local_model"})
        assert resp.status_code == 422

    def test_missing_text_field_returns_422(self):
        resp = client.post(ANALYZE_URL, json={"source": "local_model"})
        assert resp.status_code == 422

    def test_invalid_source_returns_422(self):
        resp = client.post(ANALYZE_URL, json={"text": "Some text", "source": "not_a_real_source"})
        assert resp.status_code == 422

    def test_text_over_max_length_returns_422(self):
        # `get_settings` is injected into the route via `Depends(get_settings)`,
        # so FastAPI resolves it through the DI system rather than a plain
        # module-attribute lookup. `unittest.mock.patch("...get_settings")`
        # would silently do nothing here — use `dependency_overrides` instead,
        # which is what FastAPI actually consults per-request.
        base_settings = get_settings()
        app.dependency_overrides[get_settings] = lambda: base_settings.model_copy(
            update={"max_text_length": 10}
        )

        # The local_model service must never be reached if validation fails
        # first — patch it too so a regression here fails loudly instead of
        # quietly hitting a real (likely missing) checkpoint.
        with patch("backend.routers.sentiment.get_distilbert_service") as mock_get:
            resp = client.post(ANALYZE_URL, json={"text": "a" * 50, "source": "local_model"})
            mock_get.assert_not_called()

        assert resp.status_code == 422
        assert "too long" in resp.json()["detail"].lower()

    def test_service_not_called_when_validation_fails(self):
        # Guards against a regression where validation happens after the
        # (expensive) model call instead of before it.
        with patch("backend.routers.sentiment.get_distilbert_service") as mock_get:
            client.post(ANALYZE_URL, json={"text": "", "source": "local_model"})
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# /api/v1/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_healthy_when_model_loaded_and_hf_reachable(self):
        base_settings = get_settings()
        app.dependency_overrides[get_settings] = lambda: base_settings.model_copy(
            update={"huggingfacehub_api_token": "fake-token-for-test"}
        )

        with patch("backend.main.get_distilbert_service") as mock_distilbert, \
             patch("httpx.AsyncClient.get") as mock_http_get:
            mock_distilbert.return_value = object()  # loads without raising
            mock_http_get.return_value.status_code = 200

            resp = client.get(HEALTH_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["distilbert_loaded"] is True
        assert data["ollama_reachable"] is True

    def test_reports_hf_unreachable_when_no_token_set(self):
        base_settings = get_settings()
        app.dependency_overrides[get_settings] = lambda: base_settings.model_copy(
            update={"huggingfacehub_api_token": ""}
        )

        with patch("backend.main.get_distilbert_service") as mock_distilbert:
            mock_distilbert.return_value = object()

            resp = client.get(HEALTH_URL)

        assert resp.status_code == 200
        assert resp.json()["ollama_reachable"] is False

    def test_reports_distilbert_not_loaded(self):
        with patch("backend.main.get_distilbert_service") as mock_distilbert:
            mock_distilbert.side_effect = ModelNotFoundError("checkpoint missing")

            resp = client.get(HEALTH_URL)

        assert resp.status_code == 200
        assert resp.json()["distilbert_loaded"] is False
