'''''''''
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import spacy
import os
from transformers import pipeline

# --- Force PyTorch backend ---
os.environ["USE_TF"] = "0"  # Disable TensorFlow, use PyTorch backend

# --- Initialize FastAPI app ---
app = FastAPI(title="Mental Health Triage Chatbot MVP")

# --- Load NLP models once at startup ---
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# --- Define message schema ---
class MessageIn(BaseModel):
    user_id: str
    message: str

# --- Core NLP Functions ---
def detect_intent_bert(text: str):
    """
    Use pretrained DistilBERT emotion classifier to detect emotional intent.
    """
    result = classifier(text)[0]
    intent = result["label"]
    confidence = result["score"]
    return intent, confidence

def extract_entities(text: str):
    """
    Use spaCy to extract entities (like PERSON, DATE, ORG, etc.)
    """
    doc = nlp(text)
    return [(ent.label_, ent.text) for ent in doc.ents]

def process_message(message: str):
    """
    Process user message → detect intent + extract entities
    """
    intent, confidence = detect_intent_bert(message)
    entities = extract_entities(message)
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities
    }

# --- API Endpoints ---
@app.post("/message")
def receive_message(msg: MessageIn):
    clean_text = msg.message.strip()
    nlp_result = process_message(clean_text)

    return {
        "user_id": msg.user_id,
        "clean_message": clean_text,
        "intent": nlp_result["intent"],
        "confidence": round(nlp_result["confidence"], 3),
        "entities": nlp_result["entities"],
        "timestamp": datetime.utcnow().isoformat()
    }

# --- Test Route ---
@app.get("/test")
def test():
    sample_text = "Hi, I feel anxious and haven't been sleeping well."
    result = process_message(sample_text)
    return {
        "sample_input": sample_text,
        "output": result
    }
'''''''''''
'''''''''''
import os
os.environ["USE_TF"] = "0"  # Force PyTorch backend (avoid TensorFlow errors)

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import spacy
from transformers import pipeline
import requests
import csv

# --- Initialize FastAPI app ---
app = FastAPI(title="Mental Health Triage Chatbot MVP")

# --- Load NLP models once at startup ---
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# --- Define message schema ---
class MessageIn(BaseModel):
    user_id: str
    message: str

# --- Crisis keyword lists ---
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end it", "can't live",
    "better off dead", "no point", "die", "end my life"
]
SUPPORT_KEYWORDS = ["alone", "no one", "nobody", "isolated", "hopeless"]
FUNCTIONAL_KEYWORDS = ["can't sleep", "tired", "no energy", "skip class", "fail", "can't focus"]

# --- Emotion-to-risk mapping ---
RISK_MAP = {
    "joy": 10,
    "neutral": 20,
    "surprise": 30,
    "sadness": 60,
    "anger": 50,
    "fear": 70,
    "disgust": 55
}

# --- Triage category thresholds ---
def classify_risk(score: float) -> str:
    if score >= 85:
        return "CRISIS"
    elif score >= 65:
        return "URGENT"
    elif score >= 40:
        return "MODERATE"
    elif score >= 20:
        return "MILD"
    else:
        return "WELLNESS"

def triage_recommendation(category: str) -> str:
    recs = {
        "CRISIS": "Connect to crisis support immediately (988 / campus hotline).",
        "URGENT": "Schedule counselor within 48 hours.",
        "MODERATE": "Schedule counselor within 1–2 weeks.",
        "MILD": "Provide self-help resources; optional counseling.",
        "WELLNESS": "Offer preventive mental health and wellness programs."
    }
    return recs.get(category, "No recommendation available.")

# --- NLP and logic pipeline ---
def detect_intent_bert(text: str):
    """Use pretrained DistilBERT emotion classifier."""
    result = classifier(text)[0]
    intent = result["label"]
    confidence = result["score"]
    return intent, confidence

def extract_entities(text: str):
    """Extract named entities (PERSON, DATE, ORG, etc.)"""
    doc = nlp(text)
    return [(ent.label_, ent.text) for ent in doc.ents]

def calculate_risk_score(intent: str, text: str) -> float:
    """Calculate overall triage risk score (0–100)"""
    base_score = RISK_MAP.get(intent.lower(), 40)
    text_lower = text.lower()

    # Add keyword bonuses
    crisis_bonus = any(word in text_lower for word in CRISIS_KEYWORDS)
    support_bonus = any(word in text_lower for word in SUPPORT_KEYWORDS)
    functional_bonus = any(word in text_lower for word in FUNCTIONAL_KEYWORDS)

    total = base_score
    if crisis_bonus:
        total += 30
    if functional_bonus:
        total += 10
    if support_bonus:
        total += 10

    return min(total, 100)

def process_message(message: str):
    """Main pipeline: emotion → risk → triage"""
    intent, confidence = detect_intent_bert(message)
    entities = extract_entities(message)
    risk_score = calculate_risk_score(intent, message)
    triage_category = classify_risk(risk_score)
    recommendation = triage_recommendation(triage_category)
    
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "risk_score": risk_score,
        "triage_category": triage_category,
        "recommendation": recommendation
    }

# --- API Endpoints ---
@app.post("/message")
def receive_message(msg: MessageIn):
    clean_text = msg.message.strip()
    nlp_result = process_message(clean_text)
    
    result = {
        "user_id": msg.user_id,
        "clean_message": clean_text,
        "intent": nlp_result["intent"],
        "confidence": round(nlp_result["confidence"], 3),
        "risk_score": nlp_result["risk_score"],
        "triage_category": nlp_result["triage_category"],
        "recommendation": nlp_result["recommendation"],
        "entities": nlp_result["entities"],
        "timestamp": datetime.utcnow().isoformat()
    }

    # --- Log to CSV ---
    csv_filename = "triage_test_results.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return result

# --- Test Route ---
@app.get("/test")
def test():
    sample_text = "I feel hopeless and can’t sleep. I’ve been thinking about ending it all."
    result = process_message(sample_text)
    return {"sample_input": sample_text, "output": result}

# --- Auto-Test Route ---
@app.get("/run_tests")
def run_tests():
    """Runs predefined test cases and logs results to CSV."""
    test_cases = [
        "I feel great lately. Been sleeping well and doing yoga.",
        "I'm a bit stressed but managing fine.",
        "I feel anxious every night, can't focus on studying.",
        "I’m so tired all the time. Nothing feels enjoyable anymore.",
        "I can’t stop worrying, it’s ruining my sleep.",
        "I’ve been thinking about killing myself.",
        "I want to die but I don’t have a plan.",
        "Everyone would be better off without me.",
        "I cut myself last week, it helped me cope.",
        "Sometimes I wish I could just disappear forever."
    ]

    results = []
    for i, msg in enumerate(test_cases, 1):
        payload = {"user_id": f"test_user_{i}", "message": msg}
        try:
            response = requests.post("http://127.0.0.1:8000/message", json=payload, timeout=10)
            data = response.json()
            results.append(data)
        except Exception as e:
            results.append({"user_id": f"test_user_{i}", "error": str(e)})

    return {"message": "✅ All tests completed and logged to CSV.", "results": results}
'''''''''
'''''''''
import os
os.environ["USE_TF"] = "0"  # Force PyTorch backend (avoid TensorFlow errors)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
import spacy
from transformers import pipeline
import requests
import csv
import pandas as pd
import plotly.express as px
import random

# --- Initialize FastAPI app ---
app = FastAPI(title="Mental Health Triage Chatbot MVP")

# --- Load NLP models once at startup ---
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# --- Define message schema ---
class MessageIn(BaseModel):
    user_id: str
    message: str

# --- Crisis keyword lists ---
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end it", "can't live",
    "better off dead", "no point", "die", "end my life"
]
SUPPORT_KEYWORDS = ["alone", "no one", "nobody", "isolated", "hopeless"]
FUNCTIONAL_KEYWORDS = ["can't sleep", "tired", "no energy", "skip class", "fail", "can't focus"]

# --- Emotion-to-risk mapping ---
RISK_MAP = {
    "joy": 10,
    "neutral": 20,
    "surprise": 30,
    "sadness": 60,
    "anger": 50,
    "fear": 70,
    "disgust": 55
}

# --- Triage category thresholds ---
def classify_risk(score: float) -> str:
    if score >= 85:
        return "CRISIS"
    elif score >= 65:
        return "URGENT"
    elif score >= 40:
        return "MODERATE"
    elif score >= 20:
        return "MILD"
    else:
        return "WELLNESS"

def triage_recommendation(category: str) -> str:
    recs = {
        "CRISIS": "Connect to crisis support immediately (988 / campus hotline).",
        "URGENT": "Schedule counselor within 48 hours.",
        "MODERATE": "Schedule counselor within 1–2 weeks.",
        "MILD": "Provide self-help resources; optional counseling.",
        "WELLNESS": "Offer preventive mental health and wellness programs."
    }
    return recs.get(category, "No recommendation available.")

# --- NLP and logic pipeline ---
def detect_intent_bert(text: str):
    """Use pretrained DistilBERT emotion classifier."""
    result = classifier(text)[0]
    intent = result["label"]
    confidence = result["score"]
    return intent, confidence

def extract_entities(text: str):
    """Extract named entities (PERSON, DATE, ORG, etc.)"""
    doc = nlp(text)
    return [(ent.label_, ent.text) for ent in doc.ents]

def calculate_risk_score(intent: str, text: str) -> float:
    """Calculate overall triage risk score (0–100)"""
    base_score = RISK_MAP.get(intent.lower(), 40)
    text_lower = text.lower()

    crisis_bonus = any(word in text_lower for word in CRISIS_KEYWORDS)
    support_bonus = any(word in text_lower for word in SUPPORT_KEYWORDS)
    functional_bonus = any(word in text_lower for word in FUNCTIONAL_KEYWORDS)

    total = base_score
    if crisis_bonus:
        total += 30
    if functional_bonus:
        total += 10
    if support_bonus:
        total += 10

    return min(total, 100)

def process_message(message: str):
    """Main pipeline: emotion → risk → triage"""
    intent, confidence = detect_intent_bert(message)
    entities = extract_entities(message)
    risk_score = calculate_risk_score(intent, message)
    triage_category = classify_risk(risk_score)
    recommendation = triage_recommendation(triage_category)
    
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "risk_score": risk_score,
        "triage_category": triage_category,
        "recommendation": recommendation
    }

# --- API Endpoints ---
@app.post("/message")
def receive_message(msg: MessageIn):
    clean_text = msg.message.strip()
    nlp_result = process_message(clean_text)
    
    result = {
        "user_id": msg.user_id,
        "clean_message": clean_text,
        "intent": nlp_result["intent"],
        "confidence": round(nlp_result["confidence"], 3),
        "risk_score": nlp_result["risk_score"],
        "triage_category": nlp_result["triage_category"],
        "recommendation": nlp_result["recommendation"],
        "entities": nlp_result["entities"],
        "timestamp": datetime.utcnow().isoformat()
    }

    csv_filename = "triage_test_results.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return result

# --- Test Route ---
@app.get("/test")
def test():
    sample_text = "I feel hopeless and can’t sleep. I’ve been thinking about ending it all."
    result = process_message(sample_text)
    return {"sample_input": sample_text, "output": result}

# --- Auto-Test Route (100 diverse messages) ---
@app.get("/run_tests")
def run_tests():
    """Runs 100 diverse test cases and logs results to CSV."""
    wellness_msgs = [
        "I feel amazing today!", "I'm doing fine.", "I'm happy and motivated.",
        "I've been productive and calm.", "Feeling good about life lately.",
        "Everything's balanced, sleeping well.", "Enjoying time with friends and family.",
        "Doing yoga and meditation daily.", "My mental health feels strong.",
        "Life is stable and peaceful."
    ]

    mild_msgs = [
        "I've been a bit tired recently.", "Some stress but manageable.",
        "I get anxious sometimes before exams.", "Slight trouble sleeping this week.",
        "Feeling off but still managing.", "Minor tension lately.",
        "Worrying more than usual.", "Sometimes I lose focus easily.",
        "Feeling a bit down, not serious.", "Mild anxiety at night."
    ]

    moderate_msgs = [
        "I’ve been feeling anxious and can’t focus on work.",
        "Constant worrying is draining me.", "Feeling depressed most days.",
        "Trouble concentrating and sleeping well.", "Emotions up and down lately.",
        "I feel like nothing is enjoyable anymore.", "Can't relax even after trying.",
        "Losing interest in hobbies.", "Everything feels heavy lately.",
        "I’m so tired even after sleeping enough."
    ]

    urgent_msgs = [
        "I’ve been having dark thoughts lately.", "Feeling hopeless about my future.",
        "Anxiety is ruining my studies.", "Crying for no reason, can't stop.",
        "I can’t eat or sleep properly anymore.", "Too anxious to go outside.",
        "My stress is unbearable recently.", "I'm isolating myself from everyone.",
        "I think I might need therapy soon.", "Can’t control panic anymore."
    ]

    crisis_msgs = [
        "I want to end it all.", "I'm thinking about killing myself.",
        "Everyone would be better off without me.", "I’ve lost the will to live.",
        "Can't handle this pain anymore.", "I want to die.",
        "I've thought about ending my life.", "I feel like disappearing forever.",
        "I cut myself yesterday.", "No reason to keep going anymore."
    ]

    all_msgs = (wellness_msgs * 5) + (mild_msgs * 5) + (moderate_msgs * 5) + (urgent_msgs * 5) + (crisis_msgs * 5)
    all_msgs = all_msgs[:100]
    random.shuffle(all_msgs)

    results = []
    for i, msg in enumerate(all_msgs, 1):
        payload = {"user_id": f"test_user_{i}", "message": msg}
        try:
            response = requests.post("http://127.0.0.1:8000/message", json=payload, timeout=10)
            data = response.json()
            results.append(data)
        except Exception as e:
            results.append({"user_id": f"test_user_{i}", "error": str(e)})

    return {"message": "✅ 100 tests completed and logged to CSV.", "results_logged": len(results)}

# --- Dashboard Route ---
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Visual dashboard using data from CSV."""
    csv_filename = "triage_test_results.csv"
    if not os.path.exists(csv_filename):
        return HTMLResponse("<h3>No data yet. Run /run_tests first.</h3>")

    df = pd.read_csv(csv_filename)

    fig1 = px.pie(df, names="triage_category", title="Triage Category Distribution")
    fig2 = px.bar(df, x="intent", y="risk_score", title="Average Risk Score by Intent",
                  color="triage_category", barmode="group")

    chart1_html = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""
    <html>
    <head><title>Triage Dashboard</title></head>
    <body style="font-family: Arial; margin: 30px;">
        <h1>🧠 Mental Health Triage Dashboard</h1>
        <p>Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        {chart1_html}
        {chart2_html}
        <p>Data source: <b>{csv_filename}</b></p>
        <p><a href="/run_tests">Run new test batch</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
'''''