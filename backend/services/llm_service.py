"""
Sentiment classification using an open-source LLM served locally by Ollama,
orchestrated through LangChain.

Why Ollama: it runs fully open-source weights (Llama 3.x, Qwen2.5, Mistral, ...)
on your own machine with zero API keys / cloud cost — a good fit for a
"local model vs open source LLM" comparison in the same app.

If you'd rather point this at a hosted OSS-model endpoint (Groq, Together,
Fireworks, vLLM server, etc.) instead of local Ollama, only this file needs to
change: swap ChatOllama for the equivalent LangChain chat model class.
"""

import json
import logging
import re
import time
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from backend.config import Settings, get_settings

logger = logging.getLogger("sentiment.llm")

_SYSTEM_PROMPT = """You are a strict sentiment-classification engine.
Classify the sentiment of the user's text into exactly one of: Bad, Neutral, Good.

Respond with ONLY a single JSON object, no prose before or after, in this exact shape:
{{"label": "Bad" | "Neutral" | "Good", "confidence": <float between 0 and 1>, "reason": "<one short sentence>"}}
"""

_HUMAN_PROMPT = "Text:\n\"\"\"{text}\"\"\""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMSentimentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
        )
        # Chain: prompt -> chat model -> raw string. We parse JSON ourselves
        # below with a regex fallback, since not every local model obeys
        # "JSON only" perfectly and we don't want a brittle parser to 500.
        self.chain = prompt | self.llm | StrOutputParser()

    def predict(self, text: str) -> dict:
        start = time.perf_counter()

        raw = self.chain.invoke({"text": text})
        parsed = self._parse_json(raw)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "source": "llm",
            "engine": f"ollama:{self.settings.ollama_model}",
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
