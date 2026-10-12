"""Pydantic shapes for the chunk records and the upload manifest.

They give the API a single, documented JSON contract: a stored chunk is always
{"file", "text"}, and each upload earns one manifest entry so GET /documents
can list what is inside the index without guessing.
"""

from pydantic import BaseModel


class ChunkRecord(BaseModel):
    file: str
    text: str
    distance: float | None = None


class ManifestEntry(BaseModel):
    filename: str
    size: int
    hash: str
    chunks: int
    duration_seconds: float
    chunks_per_second: float
    uploaded_at: str
    cached: bool