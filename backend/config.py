"""
Centralised app configuration.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # -- App metadata --
    app_name: str = "Sentiment Intelligence API"
    app_version: str = "1.0.0"
    debug: bool = False

    # -- CORS --
    allowed_origins: list[str] = ["*"]

    # -- Local fine-tuned DistilBERT model --
    distilbert_model_path: str = str(BASE_DIR / "saved_models" / "distilbert_sentiment")
    max_sequence_length: int = 96
    id2label: dict[int, str] = {0: "Bad", 1: "Neutral", 2: "Good"}

    # -- Hugging Face Inference Providers --
    huggingfacehub_api_token: str = ""

    # Dedicated classifier: fast, reliable label + confidence.
    hf_model_repo_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    hf_provider: str = "hf-inference"

    # Small chat model used ONLY to explain an already-decided label —
    # much easier task than classification, so a small/fast model is fine.
    hf_reason_model_repo_id: str = "HuggingFaceTB/SmolLM3-3B"
    hf_reason_provider: str = "hf-inference"
    llm_temperature: float = 0.3
    llm_max_new_tokens: int = 60  # short sentence only
    llm_timeout_seconds: int = 15  # fail fast, reason is a nice-to-have

    # -- Request limits --
    max_text_length: int = 3000


@lru_cache
def get_settings() -> Settings:
    return Settings()