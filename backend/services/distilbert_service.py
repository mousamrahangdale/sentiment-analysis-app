"""
Loads YOUR already fine-tuned DistilBERT checkpoint (the one saved by
`model.save_pretrained("saved_models/distilbert_sentiment")` in the notebook)
and serves predictions from it.

The model is loaded exactly once per process (singleton via functools.lru_cache)
so repeated requests don't reload weights from disk.
"""

import logging
import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import Settings, get_settings
from backend.preprocessing import clean_text

logger = logging.getLogger("sentiment.distilbert")


class ModelNotFoundError(RuntimeError):
    """Raised when the fine-tuned checkpoint folder is missing/empty."""


class DistilBertSentimentService:
    def __init__(self, settings: Settings):
        model_path = Path(settings.distilbert_model_path)

        if not model_path.exists() or not any(model_path.iterdir()):
            raise ModelNotFoundError(
                f"No fine-tuned DistilBERT checkpoint found at '{model_path}'. "
                "Copy the folder produced by model.save_pretrained(...) / "
                "tokenizer.save_pretrained(...) in your notebook into this path "
                "(config.py -> distilbert_model_path)."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = settings.max_sequence_length
        self.id2label = settings.id2label

        logger.info("Loading fine-tuned DistilBERT from %s on %s", model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info("DistilBERT checkpoint loaded successfully.")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        start = time.perf_counter()

        cleaned = clean_text(text)
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        # DistilBERT's forward() doesn't accept token_type_ids (unlike BERT) -
        # some fast tokenizers still emit it, so drop it if present.
        inputs.pop("token_type_ids", None)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        label = self.id2label[pred_idx]
        probabilities = {self.id2label[i]: float(round(p, 4)) for i, p in enumerate(probs)}

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "source": "local_model",
            "engine": "distilbert-finetuned",
            "label": label,
            "confidence": float(round(probs[pred_idx], 4)),
            "probabilities": probabilities,
            "reason": None,
            "latency_ms": round(latency_ms, 2),
            "text_length": len(text),
        }


@lru_cache
def get_distilbert_service() -> DistilBertSentimentService:
    """FastAPI dependency: builds the service once, reuses it for every request."""
    return DistilBertSentimentService(get_settings())
