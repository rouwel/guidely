"""Document routes: upload, list, delete and reindex.

Upload adds a document to the running corpus and rebuilds the index, keeps a
manifest so GET /documents can list what is indexed, and skips re-embedding a
file whose bytes were already seen (hash match).
"""

import io
import time
from datetime import UTC, datetime

from fastapi import APIRouter, File, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .. import rag, store
from ..embeddings import embed, window
from ..metrics import metrics
from ..models.record import ManifestEntry

router = APIRouter()

store.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def extract_text(filename: str, contents: bytes) -> str:
    """Turn an uploaded file into plain text (PDF text layer or UTF-8 text)."""
    if filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(contents))
            if reader.is_encrypted:
                raise ValueError("This PDF is password protected")
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except PdfReadError as error:
            raise ValueError("This PDF could not be read") from error

        if not text:
            raise ValueError(
                "This PDF has no selectable text - it is probably a scan or an image"
            )
        return text

    return contents.decode("utf-8")


def rebuild_index(records: list) -> None:
    """Embed every chunk and write the index plus chunks file atomically."""
    embeddings = embed([record["text"] for record in records])
    store.store_blocking(records, embeddings)


def _manifest_entry(filename: str, contents: bytes, digest: str, total_chunks: int, duration: float) -> dict:
    return ManifestEntry(
        filename=filename,
        size=len(contents),
        hash=digest,
        chunks=total_chunks,
        duration_seconds=round(duration, 3),
        chunks_per_second=round(total_chunks / duration, 2) if duration else 0.0,
        uploaded_at=_now(),
        cached=False,
    ).model_dump()


@router.post("/upload/documents")
async def upload_document(file: UploadFile = File(...)):
    # FIX: file.filename is Optional in FastAPI, so a client sending no filename
    # would raise AttributeError inside endswith() instead of getting a 400.
    allowed = (".txt", ".pdf")
    if not file.filename or not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400, detail="Only .txt and .pdf files are supported"
        )

    contents = await file.read()

    try:
        document = extract_text(file.filename, contents)
    except UnicodeDecodeError:
        # FIX: `from None` hides the UnicodeDecodeError traceback from the response.
        raise HTTPException(
            status_code=400, detail="The file must be UTF-8 encoded text"
        ) from None
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    digest = store.sha256(contents)
    manifest = store.load_manifest()

    # Re-uploading the same bytes is a no-op: nothing is re-embedded, the index is
    # not rebuilt, and the existing manifest entry is returned as-is.
    existing = next((entry for entry in manifest if entry["hash"] == digest), None)
    if existing:
        metrics.record_cache_hit()
        return {
            "filename": file.filename,
            "message": "This document was already embedded (unchanged content)",
            "chunks_stored": existing["chunks"],
            "cached": True,
        }

    records = store.load_records() + [
        {"file": file.filename, "text": chunk} for chunk in rag.build_chunks(document, window())
    ]

    started = time.perf_counter()
    rebuild_index(records)
    duration = time.perf_counter() - started
    metrics.record_upload(len(records), duration)

    # A re-upload of the same *name* replaces that name's entry, so the list never
    # fills with stale versions of one file.
    manifest = [entry for entry in manifest if entry["filename"] != file.filename]
    manifest.append(_manifest_entry(file.filename, contents, digest, len(records), duration))
    store.save_manifest(manifest)

    return {
        "filename": file.filename,
        "message": "Document embedded and stored successfully",
        "chunks_stored": len(records),
        "cached": False,
    }


@router.get("/documents")
def list_documents():
    manifest = store.load_manifest()
    return {"count": len(manifest), "documents": manifest}


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    manifest = store.load_manifest()
    remaining = [entry for entry in manifest if entry["filename"] != filename]
    if len(remaining) == len(manifest):
        raise HTTPException(
            status_code=404, detail=f"No document named '{filename}' is indexed"
        )

    remaining_records = [
        record for record in store.load_records() if record.get("file", "unknown") != filename
    ]
    if remaining_records:
        rebuild_index(remaining_records)
    else:
        store.CHUNKS_PATH.unlink(missing_ok=True)
        store.INDEX_PATH.unlink(missing_ok=True)

    store.save_manifest(remaining)
    return {
        "message": f"Deleted '{filename}' and rebuilt the index",
        "remaining_documents": len(remaining),
    }


@router.post("/documents/reindex")
def reindex():
    records = store.load_records()
    if not records:
        raise HTTPException(status_code=404, detail="No document has been uploaded yet")

    started = time.perf_counter()
    rebuild_index(records)
    duration = time.perf_counter() - started
    metrics.record_upload(len(records), duration)

    return {
        "message": "Index rebuilt",
        "chunks_stored": len(records),
        "duration_seconds": round(duration, 3),
    }