# Triage Chatbot

[![CI/CD](https://github.com/AmmarMaamoun/Triage_chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/AmmarMaamoun/Triage_chatbot/actions/workflows/ci.yml)

**A mental health early-triage chatbot that classifies conversational risk severity in real time, using emotion detection, entity extraction, and a clinically-informed scoring engine.**

---

## Introduction

Demand for mental health support outpaces available therapy capacity, and many people wait too long before anyone even knows they need help. This project explores a narrow, honest slice of that problem: **signal detection, not diagnosis** — a five-question structured conversation that flags who may need immediate attention, who should be followed up, and who is currently stable.

A user answers five short questions in a chat widget. The backend runs each response through an emotion classifier and a keyword-based crisis detector, combines them into a single risk score, and returns a severity category with a tailored recommendation — instantly, with no clinician in the loop for the triage step itself.

This is a compact but layered NLP system: two ML models (transformer-based emotion classification, spaCy NER) feeding a hand-tuned scoring engine, wrapped in a FastAPI service and shipped as a container to Cloud Run. For a technical reviewer, it's worth reading as an example of combining off-the-shelf ML with domain-specific rule logic — not just calling a model and returning its output.

**This tool does not diagnose or treat.** It is a triage aid, explicitly designed to defer to human clinicians and emergency services — a stance built into the product itself, not just this README (see `index.html`).

## Architecture

A message goes through three stages before a severity category comes back: emotion classification, keyword-based crisis detection, and a weighted risk score.

```
Chat Widget (index.html)
        │  POST /message
        ▼
   FastAPI (main.py)
        │
        ├─▶ spaCy (en_core_web_sm)         — entity extraction
        ├─▶ transformers pipeline           — emotion classification
        │   (distilbert-base-uncased-emotion)
        └─▶ Risk Engine                     — keyword rules + weighted score
                        │
                        ▼
        triage_test_results.csv (session log)
```

### Component breakdown

| Layer | Responsibility |
|---|---|
| `main.py` | FastAPI app — model loading, `/message` scoring endpoint, `/result_chart`, `/index` |
| `index.html` | Chat widget UI — five fixed questions, a locked crisis-branch MCQ, and a live severity heat bar |
| `Dockerfile` | Container build, pinned to a specific `en_core_web_sm` wheel for reproducibility |

### How the risk score is built

**Emotion classification.** Each message is scored by a DistilBERT emotion model; each emotion label maps to a base risk weight (`joy` low, `fear` high), weighted by the model's confidence per label.

**Keyword crisis detection.** Three keyword tiers (`STRONG_CRISIS`, `MEDIUM_CRISIS`, `MILD_NEGATIVE`) add to the score on match. A strong-crisis phrase (e.g. "kill myself") floors the score at CRISIS regardless of the model's emotion read — the rule-based layer overrides the ML layer by design, not by accident.

**Locked Q1.** The first question is a fixed multiple-choice self-harm screen. Selecting the most severe option forces a CRISIS classification immediately, client-side, before the rest of the conversation even runs — a deliberate fail-safe independent of the scoring pipeline.

**Category thresholds.** The final 0–100 score maps to `WELLNESS → MILD → MODERATE → URGENT → CRISIS`, each paired with a plain-language explanation and a link to an appropriate resource (self-help content up to crisis hotlines).

## Technology

- **Backend:** Python 3.10, FastAPI, Gunicorn/Uvicorn
- **NLP:** spaCy (`en_core_web_sm`), Hugging Face `transformers` (DistilBERT emotion model), PyTorch
- **Data:** pandas, CSV-based session logging
- **Frontend:** Static HTML/JS chat widget, Tailwind (CDN)
- **Deployment:** Docker → Google Cloud Run

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn` | API framework and server |
| `transformers`, `torch` | Emotion classification pipeline |
| `spacy` | Named-entity extraction |
| `pandas`, `numpy` | Session data handling for `/result_chart` |
| `python-multipart` | Form/request parsing |
| `gunicorn` | Production process manager (used in the container) |

## Usage

Open `/index` to load the chat widget. It asks five fixed questions; the first is a multiple-choice self-harm screen that can short-circuit straight to a crisis response. The remaining four are free text, sent together to `POST /message`, which returns a risk score, category, and recommendation. `GET /result_chart` returns the emotion breakdown behind the most recent entry, for the widget's severity heat bar.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Deploying (Cloud Run)

```bash
docker build -t triage-chatbot .
docker tag triage-chatbot europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/triage-repo/triage-chatbot:latest
docker push europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/triage-repo/triage-chatbot:latest

gcloud run deploy triage-chatbot \
  --image europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/triage-repo/triage-chatbot:latest \
  --platform managed --region europe-west1 \
  --allow-unauthenticated --memory=4Gi --timeout=15m
```

## CI/CD

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request:

1. **Lint & test** — `ruff check .` and `pytest` on Python 3.10 with pip caching. The suite (`tests/`) covers the risk-category thresholds, the `STRONG_CRISIS` keyword override that forces CRISIS regardless of the emotion model, the locked-Q1 crisis branch, and the `/` and `/message` endpoints. spaCy and the transformers pipeline are stubbed in `tests/conftest.py`, so CI never installs torch or downloads models.
2. **Deploy** (pushes to `main` only, after tests pass) — builds the Docker image, pushes it to Artifact Registry (`europe-west1-docker.pkg.dev/<project>/triage-repo/triage-chatbot`, tagged with the commit SHA and `latest`) and deploys it to Cloud Run with the flags below.

The deploy job only runs when GCP credentials are configured as repository secrets; without them it is skipped and the workflow still passes. To enable it, add `GCP_PROJECT_ID` plus either `GCP_SA_KEY` (service-account JSON key) or `GCP_WIF_PROVIDER` and `GCP_WIF_SERVICE_ACCOUNT` (Workload Identity Federation). The service account needs Artifact Registry Writer, Cloud Run Admin and Service Account User.

Run the same checks locally:

```bash
pip install -r requirements-ci.txt
ruff check .
pytest
```

## Why this project stands out

- **Combines ML with an explicit safety override.** The keyword-rule layer can force a CRISIS classification regardless of what the emotion model says — a conscious choice not to let a black-box model be the only thing standing between a user and a crisis resource.
- **A fail-safe outside the ML path entirely.** The Q1 crisis branch triggers client-side, before the backend or any model runs — the system degrades safely even if the API is down.
- **Two real ML components, not a single API call.** Transformer-based emotion classification and NER are combined into a purpose-built scoring function, not surfaced as raw model output.
- **Domain-aware framing.** The product explicitly states what it isn't (diagnosis, treatment) as clearly as what it is — a product decision as much as an ethical one.
- **Container-first from the start.** Dockerfile pins system and model dependencies precisely enough to deploy the same build to Cloud Run without surprises.
