"""Search route: retrieve chunks, ask the model, log what happened.

Retrieval alone is not the deliverable: the retrieved chunks go to the model,
which writes the answer the UI shows above the sources it used.
"""

import json
import time

from fastapi import APIRouter, HTTPException, Query
from openai import APIStatusError, APITimeoutError

from .. import rag, store
from ..embeddings import embed
from ..metrics import metrics

router = APIRouter()


@router.get("/search")
def search(question: str = Query(..., min_length=1), top_k: int = Query(3, ge=1, le=20)):
    started = time.perf_counter()

    try:
        index = store.read_index()
    except (FileNotFoundError, RuntimeError):
        # FIX: a missing index is a RuntimeError from the FAISS C++ layer, not a
        # FileNotFoundError, so the old handler never fired. Both are now treated
        # as "nothing uploaded". `from None` stops the traceback chaining.
        metrics.record_failure("missing_index")
        raise HTTPException(
            status_code=404, detail="No document has been uploaded yet"
        ) from None

    try:
        chunks = [rag.as_record(record) for record in store.load_records()]
    except json.JSONDecodeError as error:
        # FIX: an unreadable chunks file used to escape as a bare 500. Say what is
        # wrong and how to fix it, since the only remedy is a fresh upload.
        metrics.record_failure("corrupt_chunks")
        raise HTTPException(
            status_code=500,
            detail="The stored chunks file is unreadable - upload the document again",
        ) from error

    # FIX: if the index holds more vectors than the chunks file has entries, the
    # lookup below raises IndexError and surfaces as an opaque 500.
    if index.ntotal != len(chunks):
        metrics.record_failure("index_mismatch")
        raise HTTPException(
            status_code=500,
            detail="The stored index and chunks do not match - upload the document again",
        )

    question_embedding = embed([question])
    distances, indices = index.search(question_embedding, k=top_k)

    results = []
    for distance, index_position in zip(distances[0], indices[0]):
        # -1 means FAISS did not find a result (fewer vectors than top_k)
        if index_position == -1:
            continue
        record = chunks[index_position]
        results.append({**record, "distance": float(distance)})

    try:
        answered = rag.answer_question(question, results)
    except ValueError as error:
        # No API key configured - a setup problem, not a bad request.
        metrics.record_failure("missing_key")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except APITimeoutError as error:
        metrics.record_failure("timeout")
        raise HTTPException(
            status_code=504, detail="The language model timed out, try again"
        ) from error
    except APIStatusError as error:
        if error.status_code == 429:
            # The free tier throttles hard. Tell the user it is a quota reset,
            # not a bug, so they wait a minute instead of replaying the search.
            metrics.record_failure("rate_limited")
            raise HTTPException(
                status_code=503,
                detail="The free model is rate-limited, wait a minute and try again",
            ) from error
        # Bad key or an empty credit balance both arrive as a status error.
        metrics.record_failure("provider")
        raise HTTPException(
            status_code=502,
            detail=f"The language model rejected the request (HTTP {error.status_code})",
        ) from error

    metrics.record_search(
        (time.perf_counter() - started) * 1000, question, len(results)
    )
    return {"question": question, "answer": answered["answer"], "sources": answered["sources"]}