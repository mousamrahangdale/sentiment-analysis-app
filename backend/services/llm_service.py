"""
Sentiment classification using an open-source LLM, orchestrated through
LangChain. Supports two interchangeable providers, picked via
LLM_PROVIDER in .env:

  - "groq"   -> hosted, free-tier API serving open-weight models
               (Llama 3.x etc.) — no local GPU/RAM needed, works on
               free hosting (Render, etc).
  - "ollama" -> fully local, self-hosted, zero API key, needs
               `ollama serve` running on the same machine.

Both paths return the same JSON shape, so nothing else in the app
(routers, schemas, frontend) needs to know which one is active.
"""

import json
import logging
import re
import time
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.config import Settings, get_settings

logger = logging.getLogger("sentiment.llm")

_SYSTEM_PROMPT = """You are a strict sentiment-classification engine.
Classify the sentiment of the user's text into exactly one of: Bad, Neutral, Good.

Respond with ONLY a single JSON object, no prose before or after, in this exact shape:
{{"label": "Bad" | "Neutral" | "Good", "confidence": <float between 0 and 1>, "reason": "<one short sentence>"}}
"""

_HUMAN_PROMPT = "Text:\n\"\"\"{text}\"\"\""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_chat_model(settings: Settings):
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. Get a free key at "
                "https://console.groq.com/keys and set it in .env (local) or as a "
                "secret in your deployment platform."
            )
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        ), f"groq:{settings.groq_model}"

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        ), f"ollama:{settings.ollama_model}"

    raise RuntimeError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use 'groq' or 'ollama').")


class LLMSentimentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm, self.engine_name = _build_chat_model(settings)

        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
        )
        # Chain: prompt -> chat model -> raw string. We parse JSON ourselves
        # below with a regex fallback, since not every model obeys
        # "JSON only" perfectly and we don't want a brittle parser to 500.
        self.chain = prompt | self.llm | StrOutputParser()

    def predict(self, text: str) -> dict:
        start = time.perf_counter()

        raw = self.chain.invoke({"text": text})
        parsed = self._parse_json(raw)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "source": "llm",
            "engine": self.engine_name,
            "label": parsed["label"],
            "confidence": parsed["confidence"],
            "probabilities": None,
            "reason": parsed.get("reason"),
            "latency_ms": round(latency_ms, 2),
            "text_length": len(text),
        }

    @staticmethod
    def _parse_json(raw: str) -> dict:
        candidate = raw.strip()
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            match = _JSON_BLOCK_RE.search(candidate)
            if not match:
                logger.warning("LLM did not return parseable JSON: %r", raw)
                return {"label": "Neutral", "confidence": 0.0, "reason": "Could not parse model output."}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("LLM JSON block still invalid: %r", raw)
                return {"label": "Neutral", "confidence": 0.0, "reason": "Could not parse model output."}

        label = str(data.get("label", "Neutral")).strip().title()
        if label not in ("Bad", "Neutral", "Good"):
            label = "Neutral"

        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return {"label": label, "confidence": confidence, "reason": data.get("reason")}


@lru_cache
def get_llm_service() -> LLMSentimentService:
    return LLMSentimentService(get_settings())