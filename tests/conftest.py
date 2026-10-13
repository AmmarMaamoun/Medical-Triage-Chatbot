"""Stub the heavy ML dependencies before main.py is imported.

main.py imports spaCy and transformers at module level and loads both models
on startup. In CI we replace them with tiny fakes so the tests exercise the
real API and risk engine without installing torch or downloading models.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeEntity:
    def __init__(self, label, text):
        self.label_ = label
        self.text = text


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


def fake_nlp(text):
    return FakeDoc([FakeEntity("DATE", "today")] if "today" in text else [])


# Emotion scores returned by the fake classifier; tests can overwrite this.
FAKE_EMOTIONS = [{"label": "joy", "score": 1.0}]


def fake_classifier(text):
    return [list(FAKE_EMOTIONS)]


spacy_stub = types.ModuleType("spacy")
spacy_stub.load = lambda name: fake_nlp
sys.modules.setdefault("spacy", spacy_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.pipeline = lambda *args, **kwargs: fake_classifier
sys.modules.setdefault("transformers", transformers_stub)


@pytest.fixture
def emotions(monkeypatch):
    """Set the emotion scores the fake classifier returns for one test."""

    def _set(scores):
        monkeypatch.setattr(sys.modules[__name__], "FAKE_EMOTIONS", scores)

    return _set


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import main

    # /message appends to a CSV in the working directory; keep it out of the repo.
    monkeypatch.chdir(tmp_path)
    with TestClient(main.app) as c:
        yield c
