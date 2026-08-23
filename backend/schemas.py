from typing import Literal, Optional

from pydantic import BaseModel, Field

SentimentSource = Literal["local_model", "llm"]
SentimentLabel = Literal["Bad", "Neutral", "Good"]


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Sentence or paragraph to analyze")
    source: SentimentSource = Field(
        ...,
        description="'local_model' -> your fine-tuned DistilBERT checkpoint, "
        "'llm' -> open-source LLM via LangChain",
    )


class SentimentResponse(BaseModel):
    source: SentimentSource
    engine: str                       # e.g. "distilbert-finetuned" or "huggingface_open-source-model"
    label: SentimentLabel
    confidence: float                 # 0..1
    probabilities: Optional[dict[str, float]] = None   # only for local_model
    reason: Optional[str] = None      # only for llm
    latency_ms: float
    text_length: int


class HealthResponse(BaseModel):
    status: str
    distilbert_loaded: bool
    hf_reachable: bool
