"""Guidely - retrieval-augmented Q&A over a small game's documents.

Module layout:

    rag.py        chunking + answer drafting (no I/O, no FastAPI)
    embeddings.py lazy sentence-transformer access (torch loads on first use)
    store.py      on-disk chunks, index, manifest and the log directory
    metrics.py    counters for /metrics plus CSV logs of queries and failures
    models/       Pydantic record shapes
    routes/       documents, search and health routers

Run with:  uv run uvicorn backend.main:app
"""

import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .routes import documents, health, search
from .store import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("guidely")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=1)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(handler)

app = FastAPI(title="Guidely", version="1.0.0")
app.include_router(documents.router)
app.include_router(health.router)
app.include_router(search.router)


# FIX: any exception we did not anticipate used to return Starlette's plain-text
# "Internal Server Error", which the frontend could only report as "Request failed
# with status 500". Keep the traceback in the app log, but send a readable message.
@app.exception_handler(Exception)
async def unhandled_error(request: Request, error: Exception):
    logger.exception("unhandled error while handling %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error - check the api log for the traceback"},
    )


# FastAPI's default validation error returns a list of objects; log it while
# keeping the same JSON shape the frontend already knows how to read.
@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    logger.warning("validation error on %s: %s", request.url.path, error)
    details = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/")
def home():
    return {"message": "Guidely API - interactive docs at /docs"}