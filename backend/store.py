"""On-disk state: chunk records, the FAISS index and the document manifest.

Everything survives a restart. All paths live here so the writer (upload) and
the reader (search) cannot drift apart, as they did when each held its own path.
"""

import hashlib
import json
from pathlib import Path

import faiss

DATA_DIR = Path(__file__).resolve().parent / "data"
LOG_DIR = Path(__file__).resolve().parent / "logs"
SAMPLE_DIR = DATA_DIR / "sample-docs"
INDEX_PATH = DATA_DIR / "chunks.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
MANIFEST_PATH = DATA_DIR / "documents.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> list:
    """The current manifest, or [] when nothing has been uploaded yet."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest: list) -> None:
    temp = MANIFEST_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(MANIFEST_PATH)


def load_records() -> list:
    """The stored chunk records, or [] when nothing has been uploaded yet."""
    if not CHUNKS_PATH.exists():
        return []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_index():
    """The stored FAISS index (raises FileNotFoundError/RuntimeError when missing)."""
    return faiss.read_index(str(INDEX_PATH))


def store_blocking(records: list, embeddings) -> int:
    """Write records + their embeddings as a chunk/index pair.

    Both files are written to temporary names and then swapped in, so a crash
    between the two writes can never leave an index with no chunks file or a
    half-written json file behind.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    temp_index = INDEX_PATH.with_suffix(".index.tmp")
    temp_chunks = CHUNKS_PATH.with_suffix(".json.tmp")
    faiss.write_index(index, str(temp_index))
    temp_chunks.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    temp_index.replace(INDEX_PATH)
    temp_chunks.replace(CHUNKS_PATH)
    return len(records)