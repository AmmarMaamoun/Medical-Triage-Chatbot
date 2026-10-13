import csv


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_serves_chat_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Triage Assistant" in resp.text


def test_message_returns_triage_result(client, emotions):
    emotions([{"label": "joy", "score": 0.9}, {"label": "sadness", "score": 0.1}])
    resp = client.post("/message", json={"user_id": "u1", "message": "  I went out today  "})
    assert resp.status_code == 200
    body = resp.json()

    assert body["user_id"] == "u1"
    assert body["clean_message"] == "I went out today"
    assert body["triage_category"] == "MILD"
    assert body["entities"] == [["DATE", "today"]]
    assert "Tips to get better" in body["recommendation"]
    assert body["session_id"]
    assert body["emotions"] == {"joy": 0.9, "sadness": 0.1}


def test_message_strong_crisis_keyword_forces_crisis(client, emotions):
    emotions([{"label": "joy", "score": 1.0}])
    resp = client.post("/message", json={"user_id": "u2", "message": "I want to end my life"})
    body = resp.json()
    assert body["triage_category"] == "CRISIS"
    assert "findahelpline.com" in body["recommendation"]
    assert "800 HOPE" in body["recommendation"]


def test_message_q1_most_severe_answer_forces_crisis(client, emotions):
    emotions([{"label": "joy", "score": 1.0}])
    resp = client.post(
        "/message",
        json={"user_id": "u3", "message": "Yes, and I’ve taken steps recently"},
    )
    body = resp.json()
    assert body["risk_score"] == 80
    assert body["triage_category"] == "CRISIS"


def test_messages_not_stored_by_default(client, tmp_path):
    client.post("/message", json={"user_id": "u5", "message": "hello"})
    assert not (tmp_path / "triage_test_results.csv").exists()


def test_message_is_logged_to_csv_when_enabled(client, tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "STORE_MESSAGES", True)
    client.post("/message", json={"user_id": "u4", "message": "hello"})
    with open(tmp_path / "triage_test_results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["user_id"] == "u4"


def test_message_rejects_missing_fields(client):
    resp = client.post("/message", json={"message": "no user id"})
    assert resp.status_code == 422


def test_message_rejects_overlong_input(client):
    resp = client.post("/message", json={"user_id": "u6", "message": "x" * 4001})
    assert resp.status_code == 422


def test_rate_limit_returns_429(client, monkeypatch):
    import main

    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    for _ in range(2):
        assert client.post("/message", json={"user_id": "u7", "message": "hi"}).status_code == 200
    assert client.post("/message", json={"user_id": "u7", "message": "hi"}).status_code == 429
