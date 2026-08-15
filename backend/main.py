import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.routers.sentiment import router as sentiment_router
from backend.schemas import HealthResponse
from backend.services.distilbert_service import ModelNotFoundError, get_distilbert_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentiment.app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-start the DistilBERT model at boot instead of on the first request,
    # so the first user doesn't pay the load-time latency.
    settings = get_settings()
    try:
        get_distilbert_service()
        logger.info("DistilBERT checkpoint pre-loaded at '%s'.", settings.distilbert_model_path)
    except ModelNotFoundError as exc:
        logger.warning(
            "Startup warning: %s (the /api/v1/sentiment/analyze 'local_model' "
            "route will return 503 until the checkpoint is in place).",
            exc,
        )
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sentiment_router)

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    async def health():
        distilbert_loaded = True
        try:
            get_distilbert_service()
        except ModelNotFoundError:
            distilbert_loaded = False

        ollama_reachable = False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                ollama_reachable = resp.status_code == 200
        except httpx.HTTPError:
            ollama_reachable = False

        return HealthResponse(
            status="ok",
            distilbert_loaded=distilbert_loaded,
            ollama_reachable=ollama_reachable,
        )

    # Serve the simple frontend at "/" so the whole app is a single `uvicorn` process.
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
