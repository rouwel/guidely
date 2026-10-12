"""Tests for the pure pipeline helpers in backend/rag.py (no torch, no I/O)."""

from types import SimpleNamespace

import pytest

from backend import rag


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def test_build_chunks_splits_long_text_into_multiple_chunks():
    long_text = "word " * 4000  # far beyond one 500-token window
    chunks = rag.build_chunks(long_text, FakeTokenizer())
    assert len(chunks) > 1
    # nothing is lost: all original characters survive across the chunks
    assert sum(len(chunk) for chunk in chunks) >= len(long_text)


def test_build_chunks_never_exceeds_window():
    long_text = "word " * 2000
    chunks = rag.build_chunks(long_text, FakeTokenizer(), chunk_tokens=500, overlap=100)
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_build_chunks_short_text_stays_single_chunk():
    assert rag.build_chunks("short", FakeTokenizer()) == ["short"]


def test_build_chunks_folds_tiny_trailing_chunk_into_previous():
    # 21 tokens at chunk=10/overlap=2 makes windows of 10, 10 and 5 - the 5-token
    # tail is below MIN_CHUNK_TOKENS, so it is merged into the chunk before it.
    chunks = rag.build_chunks("x" * 21, FakeTokenizer(), chunk_tokens=10, overlap=2)
    assert len(chunks) == 2
    assert "x" in chunks[-1]


def test_as_record_dict_keeps_file_and_text():
    assert rag.as_record({"file": "a.txt", "text": "hi"}) == {"file": "a.txt", "text": "hi"}


def test_as_record_missing_file_defaults_to_unknown():
    assert rag.as_record({"text": "hi"}) == {"file": "unknown", "text": "hi"}


def test_as_record_legacy_bare_string():
    assert rag.as_record("raw chunk") == {"file": "unknown", "text": "raw chunk"}


def test_answer_question_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        rag.answer_question("q", [{"file": "a", "text": "t"}])


def test_answer_question_empty_sources_needs_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = rag.answer_question("any question", [])
    assert result["answer"]
    assert result["sources"] == []


class _StubCompletions:
    @staticmethod
    def create(model=None, temperature=None, messages=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer text.【2†L4-L7】"))]
        )


class _StubChat:
    completions = _StubCompletions()


class _StubOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = _StubChat()


def test_answer_question_strips_citation_markers(monkeypatch):
    monkeypatch.setattr(rag, "OpenAI", _StubOpenAI)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    result = rag.answer_question("q", [{"file": "a.txt", "text": "t"}])
    assert result["answer"] == "Answer text."
    assert result["sources"] == [{"file": "a.txt", "text": "t"}]


def test_answer_question_sends_sources_to_model(monkeypatch):
    captured = {}

    class _RecordingCompletions:
        @staticmethod
        def create(model=None, temperature=None, messages=None):
            captured["messages"] = messages
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class _RecordingChat:
        completions = _RecordingCompletions()

    class _RecordingOpenAI:
        def __init__(self, **kwargs):
            self.chat = _RecordingChat()

    monkeypatch.setattr(rag, "OpenAI", _RecordingOpenAI)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    rag.answer_question("the question", [{"file": "b.txt", "text": "the text"}])
    user_message = captured["messages"][-1]["content"]
    assert "b.txt" in user_message
    assert "the text" in user_message
    assert "the question" in user_message