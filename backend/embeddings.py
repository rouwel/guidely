"""Lazy embedding-model access.

The sentence-transformer pulls in torch, which takes minutes to import on slow
disks. Keeping it behind get_model() means uvicorn binds instantly, /health
stays fast, and the test suite can substitute its own embedder.
"""

import os

import numpy as np

# bge-small-en-v1.5 reads 512 tokens, so the 500-token chunks from rag.py sit
# inside the window. Override with EMBEDDING_MODEL to swap without code changes.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

_model = None


def get_model():
    """Return the shared SentenceTransformer, loading it on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def is_loaded() -> bool:
    """True once the model has been loaded at least once in this process."""
    return _model is not None


def window():
    """A tokenizer for rag.build_chunks - the model's own tokenizer."""
    return get_model().tokenizer


def embed(texts: str | list[str]) -> np.ndarray:
    """Encode a str or a list of str into a float32 numpy array for FAISS."""
    text_list = [texts] if isinstance(texts, str) else texts
    return get_model().encode(text_list).astype("float32")