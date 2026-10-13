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


def test_score_is_capped_at_80():
    fear = [{"label": "fear", "score": 1.0}]
    many = " ".join(main.STRONG_CRISIS + main.MEDIUM_CRISIS + main.MILD_NEGATIVE)
    assert main.calculate_risk(fear, many) == 80


def _all_labels(**scores):
    """Full classifier output: every label, probabilities summing to 1."""
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    rest = (1 - sum(scores.values())) / (len(labels) - len(scores))
    return [{"label": lab, "score": scores.get(lab, rest)} for lab in labels]


def test_happy_text_can_reach_wellness():
    score = main.calculate_risk(_all_labels(joy=0.99), "Had a great day with friends")
    assert main.classify_risk(score) == "WELLNESS"


def test_strong_sadness_is_not_downplayed():
    # Regression: dividing by the label count turned 99.9% sadness into the minimum score.
    score = main.calculate_risk(_all_labels(sadness=0.999), "I feel low")
    assert main.classify_risk(score) == "MODERATE"


def test_sadness_plus_negative_keywords_escalates_to_urgent():
    score = main.calculate_risk(_all_labels(sadness=0.95), "I feel hopeless and alone")
    assert main.classify_risk(score) == "URGENT"


def test_medium_crisis_keywords_match_regardless_of_case():
    joy = [{"label": "joy", "score": 1.0}]
    assert main.calculate_risk(joy, "honestly i'm done") > main.calculate_risk(joy, "honestly fine")


def test_mild_negative_keywords_raise_score():
    sad = [{"label": "sadness", "score": 0.5}]
    base = main.calculate_risk(sad, "ok")
    raised = main.calculate_risk(sad, "I feel alone and hopeless")
    assert raised > base


def test_emotion_alone_never_reaches_crisis():
    fear = _all_labels(fear=0.999)
    score = main.calculate_risk(fear, "I feel hopeless, alone and can't sleep")
    assert main.classify_risk(score) == "URGENT"
