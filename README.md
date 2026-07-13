# sentiment-api-distilbert-vs-claude

<!-- CI badge goes here once GitHub Actions is set up (Phase 6):
[![CI](https://github.com/<your-username>/sentiment-api-distilbert-vs-claude/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/sentiment-api-distilbert-vs-claude/actions/workflows/ci.yml)
-->

A production sentiment-analysis service — a fine-tuned DistilBERT model served over a
Dockerised FastAPI on a live public URL — plus a rigorous, reproducible head-to-head
between the fine-tuned model and a frontier LLM (Claude) on the same task.

**🔗 Live demo:** _coming in Phase 4_ · **📊 Fine-tune vs LLM verdict:** _coming in Phase 5_

---

## The problem

A startup receives **10,000 product reviews a day** and wants to classify their
sentiment automatically. They face a build-vs-buy decision that every team ships with
LLMs now has to make:

> Should we **fine-tune a small model we host ourselves**, or just **call an LLM API**?

This project answers that question **with numbers, not opinions** — measuring accuracy,
latency, cost per 1,000 requests, and maintenance burden for each approach, and stating
a clear "when each wins" verdict.

## Approach

1. **Cheap baseline first** — TF-IDF + Logistic Regression. Establishes the floor before
   reaching for anything heavier.
2. **Fine-tuned DistilBERT** — trained on the IMDB reviews dataset via the Hugging Face
   `Trainer`, reported as an F1 lift over the baseline. Model weights hosted on the
   [Hugging Face Hub](https://huggingface.co/INEED2PPP/sentiment-distilbert-imdb).
3. **Served in production** — FastAPI with `POST /predict` (Pydantic-validated),
   `/health`, and auto-generated `/docs`, containerised with Docker.
4. **The showdown** — a reproducible eval harness runs both DistilBERT and Claude against
   a hand-labelled gold set and outputs the tradeoff table below.

## Dataset

IMDB 50K movie reviews (binary sentiment: positive / negative).
_Aspect-level sentiment is a documented stretch goal._

## Results

_Filled in as the project progresses._

| Model | Accuracy | F1 | Latency (p50) | Cost / 1k | Maintenance |
|-------|----------|----|---------------|-----------|-------------|
| TF-IDF + LogReg (baseline) | 0.893 | 0.893 | — | — | — |
| Fine-tuned DistilBERT | 0.937 | 0.937 | — | — | — |
| Claude (Anthropic API) | — | — | — | — | — |

**Verdict:** _when-each-wins analysis goes here (Phase 5)._

## API

_Endpoints documented here once built (Phase 3)._ FastAPI auto-serves interactive docs at
`/docs`.

## Setup

### Option A: Local virtual environment

```bash
# 1. Clone
git clone https://github.com/<your-username>/sentiment-api-distilbert-vs-claude.git
cd sentiment-api-distilbert-vs-claude

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key (needed for the Claude comparison, Phase 5)
cp .env.example .env             # then edit .env and add your key

# 5. Run the API
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Option B: Docker

```bash
# Build the image
docker build -t sentiment-api .

# Run it (binds to $PORT, defaults to 8000)
docker run -p 8000:8000 -e PORT=8000 sentiment-api

# Test it
curl http://localhost:8000/health
```

Model weights are pulled automatically from the Hugging Face Hub at container
startup — no local model files are baked into the image.

Once running (either option), visit `http://localhost:8000/docs` for interactive
API docs.

## Project structure

```
.
├── Dockerfile
├── .dockerignore
├── src/          # baseline, model, and FastAPI app code
├── tests/        # pytest suite (runs in CI)
├── notebooks/    # EDA and fine-tuning experiments
├── data/         # datasets (gitignored; regenerated from scripts)
├── requirements.txt
└── README.md
```

## Tech

`transformers` · `datasets` · `scikit-learn` · `FastAPI` · `pydantic` ·
`anthropic` · `Docker` · `GitHub Actions` · `pytest`


## License
Released under the [MIT License](LICENSE).
