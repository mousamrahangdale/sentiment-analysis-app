"""
Hybrid sentiment service:
  1. cardiffnlp/twitter-roberta-base-sentiment-latest -> fast, reliable
     label + confidence (dedicated classifier, no prompting needed).
  2. A small chat model -> given the text AND the already-decided label,
     writes one short sentence explaining why. This is a much easier task
     than classification (justification, not judgment), so a small model
     is fast and reliable here even though it wasn't for classification.

If step 2 fails or times out, we still return the classification result
with reason=None rather than failing the whole request — the reason is a
nice-to-have, not the core signal.
"""

import logging
import time
from functools import lru_cache

from huggingface_hub import InferenceClient

from backend.config import Settings, get_settings

logger = logging.getLogger("sentiment.llm")

_LABEL_MAP = {
    "negative": "Bad",
    "neutral": "Neutral",
    "positive": "Good",
}

_REASON_SYSTEM_PROMPT = (
    "You are given a piece of text and its already-determined sentiment label. "
    "Write ONE short, natural sentence (max 20 words) explaining why the text "
    "carries that sentiment. Do not mention confidence scores or restate the "
    "label itself. Respond with only the sentence — no quotes, no prefix."
)


class LLMSentimentService:
    def __init__(self, settings: Settings):
        if not settings.huggingfacehub_api_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is not set. Get a free token at "
                "https://huggingface.co/settings/tokens and put it in your .env file."
            )

        self.settings = settings

        self.classifier = InferenceClient(
            model=settings.hf_model_repo_id,
            token=settings.huggingfacehub_api_token,
            provider=settings.hf_provider,
            timeout=settings.llm_timeout_seconds,
        )
        self.reasoner = InferenceClient(
            model=settings.hf_reason_model_repo_id,
            token=settings.huggingfacehub_api_token,
            provider=settings.hf_reason_provider,
            timeout=settings.llm_timeout_seconds,
        )

    def predict(self, text: str) -> dict:
        start = time.perf_counter()

        # -- Step 1: classify (fast, reliable) --
        results = self.classifier.text_classification(text, top_k=None)
        best = max(results, key=lambda r: r["score"])
        label = _LABEL_MAP.get(best["label"].lower(), "Neutral")
        probabilities = {
            _LABEL_MAP.get(r["label"].lower(), r["label"]): round(r["score"], 4)
            for r in results
        }

        # -- Step 2: explain (best-effort, never blocks the response) --
        reason = self._generate_reason(text, label)

        latency_ms = (time.perf_counter() - start) * 1000

        result = {
            "source": "llm",
            "engine": f"huggingface:{self.settings.hf_model_repo_id}",
            "label": label,
            "confidence": round(best["score"], 4),
            "probabilities": probabilities,
            "reason": reason,
            "latency_ms": round(latency_ms, 2),
            "text_length": len(text),
        }

        logger.debug("Final result dict being returned: %s", result)
        return result

    def _generate_reason(self, text: str, label: str) -> str | None:
        logger.debug(
            "Calling reasoner model=%s provider=%s",
            self.settings.hf_reason_model_repo_id,
            self.settings.hf_reason_provider,
        )
        try:
            response = self.reasoner.chat_completion(
                messages=[
                    {"role": "system", "content": _REASON_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f'Text: "{text}"\nSentiment label: {label}',
                    },
                ],
                max_tokens=self.settings.llm_max_new_tokens,
                temperature=self.settings.llm_temperature,
            )
            reason = response.choices[0].message.content.strip()
            logger.debug("Reason generated successfully: %r", reason)
            return reason
        except Exception:
            logger.warning("Reason generation failed; returning label without reason.", exc_info=True)
            return None


@lru_cache
def get_llm_service() -> LLMSentimentService:
    return LLMSentimentService(get_settings())
