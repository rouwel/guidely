"""Shared fakes: no torch, no network, isolated disk state per test."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import metrics as metrics_module
from backend import rag as rag_module
from backend import store as store_module
from backend.main import app
from backend.routes import documents as documents_routes
from backend.routes import search as search_routes


class FakeTokenizer:
    """Character-based stand-in for the embedding model's tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def fake_embed(texts, *_args, **_kwargs):
    """Deterministic 16-dim vectors so FAISS works without a real model.

    The vector is built from the first characters of the text and always padded
    to 16 dims, keeping the index dimension independent of the text length.
    """
    text_list = [texts] if isinstance(texts, str) else texts
    result = np.zeros((len(text_list), 16), dtype="float32")
    for row, text in enumerate(text_list):
        for column, char in enumerate(text[:16]):
            result[row, column] = ord(char)
    return result


class _Completions:
    @staticmethod
    def create(model=None, temperature=None, messages=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A short answer.【1†L1-L3】"))]
        )


class _Chat:
    completions = _Completions()


class FakeOpenAI:
    """A stand-in OpenAI client whose create() returns a canned, marked-up answer."""

    def __init__(self, *args, **kwargs):
        self.chat = _Chat()


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Redirect disk state to a scratch dir and substitute every heavy/slow piece."""
    monkeypatch.setattr(store_module, "INDEX_PATH", tmp_path / "chunks.index")
    monkeypatch.setattr(store_module, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(store_module, "MANIFEST_PATH", tmp_path / "documents.json")
    monkeypatch.setattr(metrics_module, "LOG_DIR", tmp_path / "logs")

    monkeypatch.setattr(documents_routes, "embed", fake_embed)
    monkeypatch.setattr(documents_routes, "window", FakeTokenizer)
    monkeypatch.setattr(search_routes, "embed", fake_embed)
    monkeypatch.setattr(rag_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    # Reset the in-memory counters so tests are order-independent.
    metrics_module.metrics.searches = 0
    metrics_module.metrics.uploads = 0
    metrics_module.metrics.cache_hits = 0
    metrics_module.metrics.latencies_ms.clear()
    metrics_module.metrics.failures = {key: 0 for key in metrics_module.metrics.failures}
    metrics_module.metrics.last_index = {"chunks": 0, "duration_seconds": 0.0, "throughput": 0.0}
    yield


@pytest.fixture()
def client():
    return TestClient(app)