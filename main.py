import os
import csv
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import spacy
from transformers import pipeline


# Public demo: don't persist what strangers type unless explicitly enabled.
STORE_MESSAGES = os.getenv("STORE_MESSAGES", "false").lower() == "true"

# Per-IP rate limit for /message (in-memory; fine for a small demo service)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
_hits: dict[str, deque] = defaultdict(deque)


# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Mental Health Triage Chatbot")


# --------------------------
# Health Check
# --------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------
# Load Models
# --------------------------
@app.on_event("startup")
def load_models():
    global nlp, classifier

    nlp = spacy.load("en_core_web_sm")

    classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        framework="pt",
        return_all_scores=True
    )


# --------------------------
# Message Schema
# --------------------------
class MessageIn(BaseModel):
    user_id: str = Field(max_length=100)
    message: str = Field(min_length=1, max_length=4000)


# --------------------------
# Q1 MCQ (LOCKED)
# --------------------------
Q1_OPTIONS = [
    "No, never",
    "Yes, but fleeting thoughts",
    "Yes, and I’ve thought how I might do it",
    "Yes, and I’ve taken steps recently"
]


# --------------------------
# Keyword Groups
# --------------------------
STRONG_CRISIS = [
    "kill myself", "suicide", "end my life", "better off dead"
]

MEDIUM_CRISIS = [
    "end it", "end it all", "no point", "I'm done", "no reason"
]

MILD_NEGATIVE = [
    "alone", "isolated", "hopeless", "tired", "can't sleep", "no energy"
]


# --------------------------
# Emotion → Risk Mapping
# --------------------------
RISK_MAP = {
    "joy": 10,
    "love": 10,
    "neutral": 20,
    "surprise": 30,
    "sadness": 60,
    "anger": 50,
    "fear": 70,
    "disgust": 55
}

NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "disgust"}
POSITIVE_EMOTIONS = {"joy", "neutral", "surprise"}


# --------------------------
# Category
# --------------------------
def classify_risk(score: float) -> str:
    if score >= 75:
        return "CRISIS"
    elif score >= 65:
        return "URGENT"
    elif score >= 40:
        return "MODERATE"
    elif score >= 20:
        return "MILD"
    return "WELLNESS"


# --------------------------
# Recommendation
# --------------------------
def triage_recommendation(category: str) -> str:
    links = {
        "WELLNESS": "https://www.mind.org.uk/information-support/tips-for-everyday-living/wellbeing/",
        "MILD": "https://www.helpguide.org/articles/depression/coping-with-depression.htm",
        "MODERATE": "https://www.apa.org/topics/depression",
        "URGENT": "https://www.nimh.nih.gov/health/topics/depression",
        "CRISIS": "https://findahelpline.com"
    }

    texts = {
        "WELLNESS": "You seem generally okay. Maintain healthy routines.",
        "MILD": "You may be experiencing mild distress. Self-help strategies are recommended.",
        "MODERATE": "Your responses show moderate distress. Talking to a therapist soon is advised.",
        "URGENT": "You indicate significant emotional difficulty. Please seek support.",
        "CRISIS": "Your responses suggest you may be in crisis. Contact crisis support immediately."
    }

    text = f"{texts[category]}<br><br><a href='{links[category]}' target='_blank'>Tips to get better</a>"
    if category in ("URGENT", "CRISIS"):
        text += (
            "<br><br><strong>In the UAE:</strong> emergency 999 (police) or 998 (ambulance); "
            "mental health support line 800 HOPE (800 4673), Arabic and English."
        )
    return text


# --------------------------
# NLP
# --------------------------
def detect_emotions(text: str):
    return classifier(text)[0]


def extract_entities(text: str):
    doc = nlp(text)
    return [(ent.label_, ent.text) for ent in doc.ents]


# --------------------------
# Risk Engine
# --------------------------
def calculate_risk(emotions, text: str):
    text_lower = text.lower()

    # Probability-weighted average of per-emotion risk. The classifier returns
    # every label with scores summing to ~1, so divide by the total score
    # (not the label count, which shrank the emotion signal ~6x).
    total = sum(e["score"] for e in emotions)
    weighted = sum(RISK_MAP.get(e["label"].lower(), 40) * e["score"] for e in emotions)
    score = weighted / total if total else 0

    def hits(keywords):
        return sum(k.lower() in text_lower for k in keywords)

    score += 25 * hits(STRONG_CRISIS)
    score += 15 * hits(MEDIUM_CRISIS)
    score += 10 * hits(MILD_NEGATIVE)

    if hits(STRONG_CRISIS):
        score = max(score, 75)
    elif not hits(MEDIUM_CRISIS):
        # Emotion and mild keywords alone can reach URGENT, never CRISIS:
        # CRISIS needs an explicit crisis phrase (or the Q1 screen).
        score = min(score, 74)

    # 0-80 keeps WELLNESS (<20) reachable; 80 is the top of the UI heat bar.
    return max(0, min(score, 80))


# --------------------------
# API: Message
# --------------------------
def _client_ip(request: Request) -> str:
    # Cloud Run puts the real client first in X-Forwarded-For
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _check_rate_limit(ip: str):
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, please wait a minute.")
    window.append(now)


@app.post("/message")
def receive_msg(msg: MessageIn, request: Request):
    _check_rate_limit(_client_ip(request))
    session_id = str(uuid.uuid4())
    text = msg.message.strip()

    # ---- Q1 MCQ ENFORCEMENT ----
    q1_answer = next((q for q in Q1_OPTIONS if q in text), None)
    forced_crisis = q1_answer == "Yes, and I’ve taken steps recently"

    emotions = detect_emotions(text)
    entities = extract_entities(text)

    risk_score = calculate_risk(emotions, text)

    if forced_crisis:
        risk_score = 80

    category = classify_risk(risk_score)

    out = {
        "session_id": session_id,
        "user_id": msg.user_id,
        "clean_message": text,
        "risk_score": round(risk_score, 2),
        "triage_category": category,
        "entities": entities,
        "timestamp": datetime.utcnow().isoformat()
    }

    if STORE_MESSAGES:
        file = "triage_test_results.csv"
        exists = os.path.isfile(file)
        with open(file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(out)

    return {
        **out,
        # Emotion breakdown for this request only (was a shared /result_chart read)
        "emotions": {e["label"]: round(float(e["score"]), 4) for e in emotions},
        "recommendation": triage_recommendation(category)
    }


# --------------------------
# Index Page
# --------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
