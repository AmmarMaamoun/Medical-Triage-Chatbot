import os
import csv
import uuid
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import spacy
from transformers import pipeline
import pandas as pd


# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Mental Health Triage Chatbot")


# --------------------------
# Health Check
# --------------------------
@app.get("/")
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
    user_id: str
    message: str


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
        "CRISIS": "https://www.opencounseling.com/suicide-hotlines"
    }

    texts = {
        "WELLNESS": "You seem generally okay. Maintain healthy routines.",
        "MILD": "You may be experiencing mild distress. Self-help strategies are recommended.",
        "MODERATE": "Your responses show moderate distress. Talking to a therapist soon is advised.",
        "URGENT": "You indicate significant emotional difficulty. Please seek support.",
        "CRISIS": "Your responses suggest you may be in crisis. Contact crisis support immediately."
    }

    return f"{texts[category]}<br><br><a href='{links[category]}' target='_blank'>Tips to get better</a>"


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

    weighted = 0
    for e in emotions:
        base = RISK_MAP.get(e["label"].lower(), 40)
        weighted += base * e["score"]

    score = weighted / len(emotions)

    score += 25 * sum(k in text_lower for k in STRONG_CRISIS)
    score += 15 * sum(k in text_lower for k in MEDIUM_CRISIS)
    score += 10 * sum(k in text_lower for k in MILD_NEGATIVE)

    if any(k in text_lower for k in STRONG_CRISIS):
        score = max(score, 75)

    return max(20, min(score, 80))


# --------------------------
# API: Message
# --------------------------
@app.post("/message")
def receive_msg(msg: MessageIn):
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

    file = "triage_test_results.csv"
    exists = os.path.isfile(file)

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(out)

    return {
        **out,
        "recommendation": triage_recommendation(category)
    }


# --------------------------
# Heat Bar Endpoint (REAL)
# --------------------------
@app.get("/result_chart")
def result_chart():
    csv_path = "triage_test_results.csv"
    if not os.path.exists(csv_path):
        return {"error": "no_data"}

    df = pd.read_csv(csv_path)
    last = df.iloc[-1]

    emotions = detect_emotions(last["clean_message"])
    scores = {e["label"]: float(e["score"]) for e in emotions}

    return {
        "labels": list(scores.keys()),
        "values": list(scores.values()),
        "score": float(last["risk_score"]),
        "category": last["triage_category"]
    }


# --------------------------
# Index Page
# --------------------------
@app.get("/index", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
