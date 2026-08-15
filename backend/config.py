"""
Centralised app configuration.

All tunables live here and are overridable via environment variables / .env
so the exact same code runs unchanged across local dev, CI, and (later)
a Docker container.
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
    max_sequence_length: int = 96  # must match training-time max_length
    id2label: dict[int, str] = {0: "Bad", 1: "Neutral", 2: "Good"}

    # -- Open source LLM via LangChain + Ollama --
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 30

    # -- Request limits --
    max_text_length: int = 3000


@lru_cache
def get_settings() -> Settings:
    """Settings are cheap to build but env parsing happens once per process."""
    return Settings()
