# triage_chatbot.py

import os
os.environ["USE_TF"] = "0"  # Force PyTorch backend (avoid TensorFlow errors)
from transformers import AutoTokenizer, AutoModelForCausalLM
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
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)
# --- Generative empathy model ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# --- Define message schema ---
class MessageIn(BaseModel):
    user_id: str
    message: str

# --- Crisis keyword lists ---


CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end it", "can't live",
    "better off dead", "no point", "end it all", "die", "end my life",
    "end things", "commit suicide", "take my own life", "off myself",
    "do it", "check out for good", "make myself not exist", "end this life",
    "do myself in", "kill myself tonight", "end everything", "end myself",
    "finish myself off", "end my suffering", "end the pain", "end my existence",
    "stop my heart", "make it all stop", "end my story", "take the easy way out",
    "check out early", "bow out for good", "call it quits on life", "end the struggle",
    "finish this", "end my time here", "quit life", "end my battle", "end my journey",
    "stop breathing on purpose", "make myself dead", "end my consciousness",
    "terminate my life", "finish my life", "end my being", "do away with myself",
    "destroy myself", "annihilate myself", "erase myself", "remove myself from existence",
    "cancel myself", "nullify my life", "extinguish my life", "snuff myself out",
    "put an end to myself", "bring about my death", "cause my own death",
    "engineer my demise", "orchestrate my end", "become deceased by my own hand",
    "self-terminate", "self-destruct", "auto-annihilate", "make myself cease to be",
    "become unalived by myself", "unalive myself", "delete myself",
    "log out of life for good", "hit the final quit button", "press the off switch on me",
    "pull my own plug", "turn out my own light", "blow my own candle out",
    "end my own movie", "close my own book", "write my own ending", "make my final exit",
    "take my final bow", "leave this world on purpose", "depart by my own choice",
    "exit stage left by myself", "check out permanently", "cash in my own chips",
    "fold my own hand", "end the game myself", "leave the building for good",
    "take a permanent vacation", "go to sleep forever", "never wake up on purpose",
    "end my waking hours", "silence myself permanently", "end my own suffering",
    "stop my own pain", "put myself out of my misery", "give myself a final peace",
    "find a permanent solution", "take a drastic way out", "make a final decision about me",
    "take irreversible action", "do the unthinkable to myself", "carry out the plan on myself",
    "go through with it on myself", "act on these thoughts about me",
    "follow through on this feeling", "make the dark thought a reality",
    "turn the idea into action for me", "do what I've been thinking about",
    "give in to the dark urge about me"
]

SUPPORT_KEYWORDS = [
    "alone", "no one", "nobody", "isolated", "hopeless"
]

loneliness_keywords = [
    "alone", "no one", "nobody", "isolated", "hopeless", "all alone",
    # ... rest of loneliness keywords
]

FUNCTIONAL_KEYWORDS = [
    "can't sleep", "tired", "no energy", "skip class", "fail", "can't focus",
    "exhausted", "drained", "fatigued", "worn out", "burnt out", "running on empty",
    # ... rest of functional keywords
]

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
    if score >= 75:
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
    """
    For return_all_scores=True:
    classifier(text) returns a list containing a list of all emotions.
    We pick the emotion with the highest score.
    """
    full_output = classifier(text)[0]  # list of {label, score}

    # Find highest-scoring emotion
    best = max(full_output, key=lambda x: x["score"])

    return best["label"], best["score"]


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
        total += 25
    if functional_bonus:
        total += 10
    if support_bonus:
        total += 10

    return min(total, 100)
def generate_empathy(text: str) -> str:
    """
    Generates ONE short empathic sentence.
    No follow-up questions, no advice.
    """
    prompt = (
        f"The user said: '{text}'. "
        "Write a single short sentence showing empathy. "
        "Do NOT ask any questions. "
        "Do NOT ask for more details. "
        "Do NOT give advice. "
        "Just reflect their feelings with warmth."
    )

    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = gen_model.generate(
        inputs["input_ids"],
        max_new_tokens=40,
        temperature=0.7,
        top_p=0.9
    )

    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure absolutely no question marks
    response = response.replace("?", "")

    return response

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
class EmpathyIn(BaseModel):
    message: str

@app.post("/empathy")
def empathy_api(msg: EmpathyIn):
    reply = generate_empathy(msg.message)
    return {"empathy": reply}

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

# --- Dashboard Route (HEATMAP VERSION) ---
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Visual dashboard using HEATMAP instead of pie/bar charts."""
    csv_filename = "triage_test_results.csv"
    if not os.path.exists(csv_filename):
        return HTMLResponse("<h3>No data yet. Run /run_tests first.</h3>")

    df = pd.read_csv(csv_filename)

    # Create a pivot table: rows = intent, columns = triage category, values = average risk score
    heatmap_data = df.pivot_table(
        index="intent",
        columns="triage_category",
        values="risk_score",
        aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Reds",
        title="Triage Heatmap: Avg Risk Score by Intent & Category"
    )

    heatmap_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html_content = f"""
    <html>
    <head><title>Triage Dashboard (Heatmap)</title></head>
    <body style="font-family: Arial; margin: 30px;">
        <h1>🧠 Mental Health Triage Heatmap Dashboard</h1>
        <p>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        {heatmap_html}
        <p>Data source: <b>{csv_filename}</b></p>
        <p><a href="/run_tests">Run new test batch</a></p>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)
# --- Chat UI Route (SAFE VERSION) ---
@app.get("/results", response_class=HTMLResponse)
def results_page():
    print(" /results endpoint called!")

    csv_filename = "triage_test_results.csv"
    if not os.path.exists(csv_filename):
        return HTMLResponse("<h3>No results found. Please complete the chat first.</h3>")

    df = pd.read_csv(csv_filename)

    if df.empty:
        return HTMLResponse("<h3>No data found.</h3>")

    # ---------------------------
    # 1) Get last user entry
    # ---------------------------
    last_row = df.iloc[-1]
    text = last_row["clean_message"]

    print("Running classifier...")
    probs = classifier(text, return_all_scores=True)[0]

    # Convert classifier output to dictionary
    emotion_scores = {item["label"]: item["score"] for item in probs}

    # Convert to DataFrame for radar
    radar_df = pd.DataFrame({
        "emotion": list(emotion_scores.keys()),
        "score": list(emotion_scores.values())
    })

    # Close radar loop
    radar_df = pd.concat([radar_df, radar_df.iloc[[0]]], ignore_index=True)

    categories = radar_df["emotion"].tolist()
    values = radar_df["score"].tolist()

    import plotly.graph_objects as go

    # ---------------------------
    # 2) Build Radar Chart
    # ---------------------------
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color="indigo")
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Emotional Breakdown (Radar Chart)"
    )

    radar_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # ------------------------------------------------
    # 3) “Feels Like You” Similarity Score (NEW)
    # ------------------------------------------------
    import numpy as np

    # Load emotion vectors for all users
    all_scores = []

    for _, row in df.iterrows():
        temp_probs = classifier(row["clean_message"], return_all_scores=True)[0]
        vec = [item["score"] for item in temp_probs]
        all_scores.append(vec)

    all_scores = np.array(all_scores)

    # Compute population mean vector
    population_mean = np.mean(all_scores, axis=0)

    # User vector
    user_vec = np.array(list(emotion_scores.values()))

    # Cosine similarity
    similarity = np.dot(user_vec, population_mean) / (
        np.linalg.norm(user_vec) * np.linalg.norm(population_mean)
    )

    feels_like_pct = round(similarity * 100)

    # ------------------------------------------------
    # 4) Return HTML
    # ------------------------------------------------
    return HTMLResponse(f"""
    <html>
    <body style="font-family: Arial; padding: 30px;">

        <h2> Your Emotional Analysis</h2>

        {radar_html}

        <h3 style="margin-top: 30px;">🌟 Feels Like You Score</h3>
        <p style="font-size: 1.1rem;">
            You share a similar emotional profile with 
            <b>{feels_like_pct}%</b> of people.
        </p>

        <br>
        <a href="/chat" style="color: indigo;">⬅ Back to Chat</a>
    </body>
    </html>
    """)

# --- Chat UI Route (SAFE, FINAL) ---
@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    # Path to chat.html in same folder as INPUT.py
    file_path = os.path.join(os.path.dirname(__file__), "chat.html")

    print(" Checking for chat.html at:", file_path)

    if not os.path.exists(file_path):
        return HTMLResponse(
            f"""
            <h3> chat.html NOT FOUND</h3>
            <p>Looked for file at:</p>
            <pre>{file_path}</pre>
            <p>Please place chat.html in the SAME FOLDER as INPUT.py</p>
            """,
            status_code=404
        )

    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    return HTMLResponse(html)




