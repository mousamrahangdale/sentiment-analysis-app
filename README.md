# Signal Lab — Sentiment Console

A production-shaped sentiment analysis app that lets you compare **your own
fine-tuned DistilBERT model** against an **open-source LLM (via LangChain +
Ollama)** on the same text, side by side.

```
┌─────────────┐     POST /api/v1/sentiment/analyze     ┌───────────────────┐
│  Frontend   │ ───────────────────────────────────────▶│  FastAPI backend  │
│ (index.html)│      { text, source }                   │                    │
└─────────────┘                                          │  source=local_model│
                                                           │   -> your DistilBERT
                                                           │  source=llm         │
                                                           │   -> LangChain+Ollama│
                                                           └───────────────────┘
```

## Project layout

```
sentiment-analysis-project/
├── backend/
│   ├── main.py              # FastAPI app, CORS, health check, serves frontend
│   ├── config.py            # env-driven settings (pydantic-settings)
│   ├── schemas.py           # request/response models
│   ├── preprocessing.py     # clean_text() — identical to training notebook
│   ├── routers/sentiment.py # POST /api/v1/sentiment/analyze
│   └── services/
│       ├── distilbert_service.py   # loads YOUR saved checkpoint, predicts
│       └── llm_service.py          # LangChain + ChatOllama, predicts
├── frontend/                # plain HTML/CSS/JS console UI
├── saved_models/distilbert_sentiment/   # <- put your checkpoint files here
├── tests/test_api.py
├── requirements.txt
└── .env.example
```

Nothing here is retrained — `distilbert_service.py` loads the exact checkpoint
your notebook produced with `model.save_pretrained(...)` /
`tokenizer.save_pretrained(...)`, and reuses your notebook's `clean_text()`
so preprocessing matches training exactly.

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed, for the open-source LLM path

## 2. Get the code into VS Code

Unzip the project, then in VS Code: **File → Open Folder →**
`sentiment-analysis-project`. Install the Python extension if prompted.

## 3. Copy your trained model in

From your notebook (Colab or local), grab everything inside
`saved_models/distilbert_sentiment/` (config.json, model weights,
tokenizer files — produced by cell that does `model.save_pretrained(...)`)
and copy it into this project's:

```
saved_models/distilbert_sentiment/
```

(Download the folder as a zip from Colab, or `!zip -r model.zip saved_models/distilbert_sentiment` and download, then unzip here.)

## 4. Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
```

## 5. Set up the open-source LLM (Ollama)

```bash
# in a separate terminal, keep this running
ollama serve

# pull a small, fast open-source model (one-time download)
ollama pull llama3.2
```

Want a different model? Change `OLLAMA_MODEL` in `.env` to any model you've
pulled (e.g. `qwen2.5:7b`, `mistral`).

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

In the console UI: paste a sentence or paragraph, pick **CH.A (your
DistilBERT)** or **CH.B (open-source LLM)**, and hit **Run Analysis**.

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

## Design notes

- **Singletons, not per-request loads.** Both the DistilBERT model and the
  LangChain/Ollama client are built once (`functools.lru_cache`) and reused —
  the model is also warm-loaded at app startup so the first user request isn't slow.
- **One endpoint, one contract.** `/api/v1/sentiment/analyze` takes a
  `source` field rather than exposing two different endpoints, so the frontend
  (and any future client) has a single, consistent response shape regardless
  of which engine answered.
- **LLM output is untrusted text.** The LLM is prompted to return strict JSON,
  but the service still parses defensively (regex fallback + safe defaults)
  instead of trusting the model's formatting blindly — local models don't
  always obey instructions as reliably as hosted frontier ones.
- **Swappable LLM backend.** Only `backend/services/llm_service.py` talks to
  Ollama. Point it at a different LangChain chat model (Groq, vLLM, Together,
  etc.) and nothing else in the app needs to change.

## Coming next

Docker packaging (once you've confirmed everything runs cleanly above) — say
the word and I'll add a `Dockerfile` + `docker-compose.yml` (backend + Ollama
as two services) on top of this exact layout.
