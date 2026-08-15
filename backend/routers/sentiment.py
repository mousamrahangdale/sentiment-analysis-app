import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.config import Settings, get_settings
from backend.schemas import SentimentRequest, SentimentResponse
from backend.services.distilbert_service import (
    DistilBertSentimentService,
    ModelNotFoundError,
    get_distilbert_service,
)
from backend.services.llm_service import LLMSentimentService, get_llm_service

logger = logging.getLogger("sentiment.router")

router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])


@router.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(
    payload: SentimentRequest,
    settings: Settings = Depends(get_settings),
):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"Text too long ({len(text)} chars). Max is {settings.max_text_length}.",
        )

    if payload.source == "local_model":
        try:
            service: DistilBertSentimentService = get_distilbert_service()
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        result = service.predict(text)

    else:  # "llm"
        try:
            llm_service: LLMSentimentService = get_llm_service()
            result = llm_service.predict(text)
        except Exception as exc:  # noqa: BLE001 - surface a clean 502 to the client
            logger.exception("LLM inference failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "Could not reach the open-source LLM. Is Ollama running "
                    f"('ollama serve') and is the model pulled ('ollama pull {settings.ollama_model}')?"
                ),
            ) from exc

    return SentimentResponse(**result)
