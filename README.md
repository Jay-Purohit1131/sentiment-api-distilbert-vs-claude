# sentiment-api-distilbert-vs-claude

<!-- CI badge goes here once GitHub Actions is set up (Phase 6):
[![CI](https://github.com/Jay-Purohit1131/sentiment-api-distilbert-vs-claude/actions/workflows/ci.yml/badge.svg)](https://github.com/Jay-Purohit1131/sentiment-api-distilbert-vs-claude/actions/workflows/ci.yml)
-->

A production sentiment-analysis service — a fine-tuned DistilBERT model served over a
FastAPI app (containerised with Docker) — plus an interactive live demo and a rigorous,
reproducible head-to-head between the fine-tuned model and a frontier LLM (Claude).

## Live demo
Try it: [Sentiment Analysis Demo](https://huggingface.co/spaces/INEED2PPP/sentiment-demo)
_(Free ZeroGPU tier — the Space sleeps when idle; the first request may take a few seconds to wake and allocate a GPU.)_

<img width="1294" height="403" alt="Screenshot 2026-07-15 at 4 25 53 PM" src="https://github.com/user-attachments/assets/e4c45553-a095-4b9d-9ca9-bb52e50a60b2" />

**📊 [Fine-tune vs LLM verdict below](#results)** — the project's centerpiece.

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

Measured on a hand-labelled 50-example gold set (`data/gold_set.csv`),
deliberately weighted toward hard cases (sarcasm, mixed sentiment, negation,
setup-then-twist). Both models were run through one reproducible harness
(`run_eval.py`).

| Model | Accuracy | F1 | Latency (p50) | Cost / 1k | Maintenance |
|-------|----------|----|---------------|-----------|-------------|
| TF-IDF + LogReg (baseline) | 0.893 | 0.893 | <1 ms | $0 | self-hosted |
| Fine-tuned DistilBERT | 0.937* | 0.937* | ~6 ms | $0 (self-hosted) | you host, patch, scale it |
| Claude (Haiku 4.5) | 0.86 | 0.82 | ~800 ms | ~$0.13 | zero — API handles it |

<sub>*DistilBERT accuracy/F1 differ by test set: 0.937 on the full IMDB test split,
0.76 on the hard-weighted gold set above. The gold set is intentionally adversarial.</sub>

### Where each model wins (per-category accuracy on the gold set)

| Category | DistilBERT | Claude |
|----------|-----------|--------|
| Clear positive | 1.00 | 1.00 |
| Clear negative | 1.00 | 1.00 |
| Sarcasm | 0.00 | 1.00 |
| Mixed sentiment | 0.60 | 0.80 |
| Faint praise | 0.75 | 0.00 |
| Negation | 1.00 | 0.75 |
| Terse | 0.67 | 0.67 |
| Setup-then-twist | 0.00 | 1.00 |

**Verdict — when each wins:**

On ordinary reviews (the bulk of real traffic), the fine-tuned DistilBERT
**matches Claude at ~130× the speed and zero marginal cost** — both score 100%
on clear-cut sentiment. Claude's advantage is entirely concentrated in
*rhetorically tricky* inputs: it read **every** sarcastic and setup-then-twist
review correctly where DistilBERT scored **0%**, because sarcasm can be handled
with a prompt instruction but not by a model that keys on surface words.

But Claude is not uniformly better. On faint-praise reviews it scored 0% —
prompting it toward a decisive verdict backfired on genuinely lukewarm text,
where DistilBERT's calibration held up.

**Recommendation for the 10k-reviews/day stakeholder:** self-host the fine-tuned
model as the default — it's effectively free and sub-10ms. Route to an LLM only
if the review stream is heavy in sarcasm or irony, where the accuracy gap
justifies the cost and latency. For most sentiment workloads, fine-tuning wins.

<sub>Gold-set categories hold only 4–5 examples each, so per-category figures show
direction, not statistical significance.</sub>

## API

- `POST /predict` — classify one review. Pydantic-validated; returns `{label, confidence}`.
- `POST /predict-batch` — classify a list of reviews.
- `GET /health` — liveness check.

Interactive docs auto-served at `/docs`.

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
├── src/
│   ├── app.py                  # FastAPI service (/predict, /predict-batch, /health)
│   ├── predict.py              # DistilBERT inference (loads the fine-tuned model)
│   └── claude_classifier.py    # Claude sentiment classifier (the comparison model)
├── notebooks/
│   ├── 01-EDA.ipynb            # class balance, length distribution, quality checks
│   ├── 02-Split.ipynb          # dedup + stratified train/val/test split (fixed seed)
│   ├── 03-Baseline.ipynb       # TF-IDF + Logistic Regression baseline
│   └── 04-finetune-distilbert.ipynb   # fine-tuning on Colab GPU (run record)
├── tests/
│   └── test_api.py             # pytest suite for the API (runs in CI)
├── data/
│   └── gold_set.csv            # 50 hand-labelled reviews (ground truth for the eval)
│                               #   IMDB splits are gitignored — regenerate via notebook 02
├── results/
│   ├── baseline_metrics.json   # TF-IDF baseline scores
│   ├── distilbert_metrics.json # fine-tuned model scores
│   ├── comparison.json         # DistilBERT vs Claude aggregate metrics
│   └── predictions.csv         # per-review predictions from both models
├── run_eval.py                 # eval harness: runs both models over the gold set
├── Dockerfile
├── .dockerignore
├── .env.example                # template; copy to .env and add your API key
├── requirements.txt
├── LICENSE
└── README.md

# Not in the repo (gitignored, regenerated or downloaded):
#   distilbert-imdb/     fine-tuned weights — from the HF Hub or notebook 04
#   data/*.csv           IMDB dataset + splits — regenerated by notebooks 01–02
#   .env                 your secrets (ANTHROPIC_API_KEY)
#   .venv/               virtual environment
```

## Tech

`transformers` · `datasets` · `scikit-learn` · `FastAPI` · `pydantic` ·
`anthropic` · `Docker` · `GitHub Actions` · `pytest`


## License
Released under the [MIT License](LICENSE).
