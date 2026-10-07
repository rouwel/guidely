"""Liveness and in-process metrics endpoints."""

from fastapi import APIRouter

from .. import embeddings
from ..metrics import metrics
from ..store import INDEX_PATH, SAMPLE_DIR, load_manifest

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": len(load_manifest()),
        "index_present": INDEX_PATH.exists(),
        "model_loaded": embeddings.is_loaded(),
        "sample_documents": len(list(SAMPLE_DIR.glob("*.txt"))),
    }


@router.get("/metrics")
def metrics_endpoint():
    return metrics.report()