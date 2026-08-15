# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# System deps for torch/transformers (kept minimal - CPU-only wheel below)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install everything except torch from requirements.txt, then install the
# CPU-only torch build explicitly - the plain PyPI "torch" wheel bundles
# CUDA libraries (~2GB+) that free-tier CPU hosts don't need and don't have
# the disk/RAM budget for.
RUN grep -v '^torch' requirements.txt > requirements.nocuda.txt \
    && pip install --no-cache-dir -r requirements.nocuda.txt \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY backend/ backend/
COPY frontend/ frontend/
COPY saved_models/ saved_models/

# Hugging Face Spaces expects the app on port 7860; most other platforms
# (Render, Railway, Fly.io) inject $PORT - default covers both.
ENV PORT=7860
EXPOSE 7860

# Non-root user (best practice, also required by some platforms e.g. HF Spaces)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]