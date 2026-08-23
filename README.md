# Signal Lab — Sentiment Console

[![Run Tests](https://github.com/mousamrahangdale/sentiment-analysis-app/actions/workflows/test.yml/badge.svg)](https://github.com/mousamrahangdale/sentiment-analysis-app/actions/workflows/test.yml)

A production-shaped sentiment analysis app that lets you compare **your own
fine-tuned DistilBERT model** against a **hybrid open-source pipeline**
(dedicated sentiment classifier + small LLM for reasoning) on the same text,
side by side.

**Live demo:** https://sentiment-analysis-app-gqpz.onrender.com
*(free-tier instance — first request after inactivity can take ~50s to wake up)*

```
┌─────────────┐     POST /api/v1/sentiment/analyze     ┌──────────────────────────┐
│  Frontend   │ ───────────────────────────────────────▶│    FastAPI backend      │
│ (index.html)│      { text, source }                   │                          │
└─────────────┘                                          │  source=local_model      │
                                                           │   -> your DistilBERT    │
                                                           │  source=llm              │
                                                           │   -> cardiffnlp RoBERTa  │
                                                           │      (label+confidence) │
                                                           │   -> Groq LLM            │
                                                           │      (short reason)      │
                                                           └──────────────────────────┘
```

## Project layout

```
sentiment-analysis-project/
├── notebooks/
│   └── ChatGPT_Sentiment_Analysis_file.ipynb   # training pipeline (see below)
├── backend/
│   ├── main.py              # FastAPI app, CORS, health check, serves frontend
│   ├── config.py            # env-driven settings (pydantic-settings)
│   ├── schemas.py           # request/response models
│   ├── routers/sentiment.py # POST /api/v1/sentiment/analyze
│   └── services/
│       ├── distilbert_service.py   # loads YOUR saved checkpoint, predicts
│       └── llm_service.py          # Hugging Face Inference Providers, predicts
├── frontend/                # plain HTML/CSS/JS console UI
├── saved_models/distilbert_sentiment/   # <- your checkpoint files live here
├── tests/test_api.py
├── .github/workflows/test.yml   # CI: runs the test suite on every push/PR
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── .env.example
```

Nothing here is retrained at request time — `distilbert_service.py` loads the
exact checkpoint produced by `notebooks/ChatGPT_Sentiment_Analysis_file.ipynb`
with `model.save_pretrained(...)` / `tokenizer.save_pretrained(...)`.

## Training pipeline (`notebooks/ChatGPT_Sentiment_Analysis_file.ipynb`)

This notebook is where the DistilBERT checkpoint in `saved_models/` came
from. It's a one-time offline step, not something the running app does —
re-run it only if you want to retrain on new/updated data. It covers:

1. **Data loading & cleaning** — reads the labeled tweet dataset (Kaggle),
   drops nulls, maps text labels to `{Bad: 0, Neutral: 1, Good: 2}`, and
   applies a `clean_text()` function (strips URLs, mentions, etc.).
2. **EDA** — class distribution plot, word clouds per sentiment class.
3. **Baseline model** — TF-IDF + Logistic Regression, hyperparameter-tuned
   with Optuna, saved to `saved_models/logreg_sentiment_pipeline.joblib`.
4. **DistilBERT fine-tuning** — `distilbert-base-uncased` fine-tuned for
   3-class sentiment classification (train/val loss curves, confusion
   matrix, attention heatmap visualization included).
5. **Saving the checkpoint** — `model.save_pretrained(...)` and
   `tokenizer.save_pretrained(...)` write to `saved_models/distilbert_sentiment/`,
   which is exactly what `backend/services/distilbert_service.py` loads.
6. **A/B testing** — compares the Logistic Regression baseline against the
   fine-tuned DistilBERT model on held-out validation data.

To retrain: open the notebook (Colab or local Jupyter), point it at your
updated CSV, run all cells, then copy the resulting
`saved_models/distilbert_sentiment/` folder into this project — no backend
code changes needed, since `distilbert_service.py` just loads whatever
checkpoint is in that path.

## How the "llm" engine works

Instead of a single chat model doing zero-shot sentiment classification
(slower, less reliable), this app uses a **hybrid two-model pipeline**:

1. **[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)**
   — a dedicated 3-class sentiment classifier (negative/neutral/positive),
   fine-tuned on ~124M tweets. Fast, reliable, no prompt engineering needed.
2. **A small chat model on Groq** — given the text *and* the already-decided
   label, writes one short sentence explaining why. This is a much easier
   task than classification, so a small/fast model works well here.

If step 2 fails or times out, the app still returns the classification
result with `reason: null` rather than failing the whole request.

## 1. Prerequisites

- Python 3.10+
- A Hugging Face account + API token with **"Make calls to Inference
  Providers"** permission (Fine-grained token) —
  [get one here](https://huggingface.co/settings/tokens)
- Docker Desktop (only if you want to run via `docker compose`)

## 2. Get the code into VS Code

Clone or unzip the project, then in VS Code: **File → Open Folder →**
`sentiment-analysis-project`. Install the Python extension if prompted.

## 3. Copy your trained model in

From your notebook (Colab or local), grab everything inside
`saved_models/distilbert_sentiment/` (config.json, model weights,
tokenizer files — produced by the cell that does `model.save_pretrained(...)`)
and copy it into this project's:

```
saved_models/distilbert_sentiment/
```

Large weight files (`.safetensors`) are tracked with **Git LFS** — run
`git lfs install` once per machine before committing.

## 4. Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
```

## 5. Configure your `.env`

```dotenv
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxx

HF_MODEL_REPO_ID=cardiffnlp/twitter-roberta-base-sentiment-latest
HF_PROVIDER=hf-inference

HF_REASON_MODEL_REPO_ID=openai/gpt-oss-20b
HF_REASON_PROVIDER=groq
LLM_TEMPERATURE=0.3
LLM_MAX_NEW_TOKENS=200
LLM_TIMEOUT_SECONDS=35
```

## 6. Run the app

```bash
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** — that's the whole app (frontend is served
by the same FastAPI process). API docs: **http://localhost:8000/docs**.

Check both engines are wired up correctly:

```bash
curl http://localhost:8000/api/v1/health
```

## 7. Try it

In the console UI: paste a sentence or paragraph, pick **your DistilBERT**
or **Open Source LLM**, and hit **Check Sentiment**.

Or via curl:

```bash
curl -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The new update finally fixed the sync bug.", "source": "local_model"}'

curl -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The new update finally fixed the sync bug.", "source": "llm"}'
```

## 8. Run tests

```bash
pytest
```

All backend routes are covered with mocked services (no real model load or
network call required), so the suite runs in a couple of seconds.

## 9. Continuous Integration

Every push and pull request to `main` triggers a GitHub Actions workflow
(`.github/workflows/test.yml`) that installs dependencies and runs the full
`pytest` suite in a clean Ubuntu container — see the badge at the top of
this README for the current status, or check the
[Actions tab](https://github.com/mousamrahangdale/sentiment-analysis-app/actions)
for run history and logs.

## 10. Run with Docker

```bash
docker compose up --build
```

Open **http://localhost:8000**. The `.env` file is injected at runtime
(`env_file:` in `docker-compose.yml`) — it is never baked into the image.

## 11. Deploy

The app ships as a single Docker image (see `Dockerfile`), so it deploys
cleanly to any container platform (Render, Railway, Fly.io, Hugging Face
Spaces). Set the same environment variables from `.env` in your platform's
dashboard — do not commit `.env` to git.

This project is currently deployed on **Render** (see the live demo link
above), with auto-deploy enabled on every push to `main`.

## Design notes

- **Singletons, not per-request loads.** Both the DistilBERT model and the
  Hugging Face inference clients are built once (`functools.lru_cache`) and
  reused — the DistilBERT model is also warm-loaded at app startup so the
  first user request isn't slow.
- **One endpoint, one contract.** `/api/v1/sentiment/analyze` takes a
  `source` field rather than exposing two different endpoints, so the
  frontend (and any future client) has a single, consistent response shape
  regardless of which engine answered.
- **Classification and reasoning are separate concerns.** A dedicated
  classifier is used for the label/confidence (fast, reliable, no prompting),
  and a small chat model is used *only* to justify an already-decided label
  (an easier task than classification) — reasoning failures degrade
  gracefully instead of failing the whole request.
- **CPU-only Docker image.** The `Dockerfile` installs the CPU-only PyTorch
  wheel explicitly, avoiding the ~2GB of bundled CUDA libraries that
  free-tier hosts don't need or have room for.
