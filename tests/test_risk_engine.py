import pytest

import main


@pytest.mark.parametrize(
    "score, expected",
    [
        (80, "CRISIS"),
        (75, "CRISIS"),
        (74.99, "URGENT"),
        (65, "URGENT"),
        (64.99, "MODERATE"),
        (40, "MODERATE"),
        (39.99, "MILD"),
        (20, "MILD"),
        (19.99, "WELLNESS"),
        (0, "WELLNESS"),
    ],
)
def test_classify_risk_thresholds(score, expected):
    assert main.classify_risk(score) == expected


def test_strong_crisis_keyword_overrides_positive_emotion():
    # The emotion model reads the text as pure joy, but the rule layer must win.
    joy = [{"label": "joy", "score": 1.0}]
    score = main.calculate_risk(joy, "Honestly I want to kill myself")
    assert score >= 75
    assert main.classify_risk(score) == "CRISIS"


@pytest.mark.parametrize("phrase", main.STRONG_CRISIS)
def test_every_strong_crisis_phrase_forces_crisis(phrase):
    joy = [{"label": "joy", "score": 1.0}]
    score = main.calculate_risk(joy, f"I keep thinking {phrase.upper()}")
    assert main.classify_risk(score) == "CRISIS"


def test_neutral_text_without_keywords_is_not_crisis():
    joy = [{"label": "joy", "score": 1.0}]
    score = main.calculate_risk(joy, "Had a nice walk in the park")
    assert main.classify_risk(score) != "CRISIS"


def test_score_is_clamped_between_20_and_80():
    fear = [{"label": "fear", "score": 1.0}]
    many = " ".join(main.STRONG_CRISIS + main.MEDIUM_CRISIS + main.MILD_NEGATIVE)
    assert main.calculate_risk(fear, many) == 80

    joy = [{"label": "joy", "score": 0.01}]
    assert main.calculate_risk(joy, "fine") == 20


def test_mild_negative_keywords_raise_score():
    sad = [{"label": "sadness", "score": 0.5}]
    base = main.calculate_risk(sad, "ok")
    raised = main.calculate_risk(sad, "I feel alone and hopeless")
    assert raised > base
