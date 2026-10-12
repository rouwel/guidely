# Guidely

A retrieval-augmented generation (RAG) system for interrogating game text.

Point it at a game's text files — dialogue dumps, logs, codex entries, patch notes — and ask
questions in plain language. It answers with the exact passages it drew from, plus the distance
of each match, so you can trace an answer back to the line that produced it instead of trusting
a summary.

## How it works

1. **Ingest** — an uploaded `.txt` or `.pdf` is decoded (PDF text layer or UTF-8) and split
   into overlapping 500-token chunks.
2. **Embed** — each chunk is converted to a vector by a `SentenceTransformer`
   (`BAAI/bge-small-en-v1.5`, 512-token window) and written to a FAISS index; the chunks are
   kept beside the index in JSON so every vector can be traced back to its source file.
3. **Retrieve** — a question is embedded with the same model and the `top_k` nearest chunks are
   pulled from the index by L2 distance.
4. **Answer** — the retrieved chunks are handed to an LLM (Groq's `openai/gpt-oss-120b` by
   default, OpenAI-compatible, no card needed), which writes the answer shown next to the
   sources it used.

Semantic rather than keyword search is the point: it finds the passage that *means* your
question even when it never uses the words you searched for.

## Getting started

Python 3.12+ with `uv` (frontend uses `npm`).

```bash
# 1. configure the LLM key (Groq free tier needs no card)
cp .env.example .env        # then paste your key into .env

# 2. backend
uv sync
uv run uvicorn backend.main:app --reload        # http://127.0.0.1:8000

# 3. frontend (in another terminal)
cd frontend && npm install && npm run dev       # http://127.0.0.1:5173
```

The Vite dev server proxies `/upload/documents` and `/search` to the API. The first upload or
evaluation run downloads the embedding model and imports torch, which takes a few minutes;
afterwards the model is cached on disk and uvicorn itself boots in ~a second because the model
loads lazily on first use, not at import (`backend/embeddings.py`).

## Dataset

The bundled sample corpus in `backend/data/sample-docs/` is the audit dataset — five in-universe
documents about a fishing town on a northern sea:

| file | topic |
|---|---|
| `neverdine-ferry-disaster.txt` | inquiry report into the ferry sinking |
| `harbour-safety-regulations.txt` | claims, drills, debris and pilot rules |
| `refunds-and-ticket-policy.txt` | ticket refunds, transfers, cancellations |
| `guild-membership-rules.txt` | dues, ranks, discipline |
| `weather-and-navigation.txt` | seasonal closure, beacons, fog, master's discretion |

## API

| method | path | purpose |
|---|---|---|
| POST | `/upload/documents` | upload `.txt`/`.pdf`; re-uploads of unchanged bytes are skipped via a SHA-256 cache | 
| GET | `/documents` | list the indexed documents (manifest) |
| DELETE | `/documents/{filename}` | remove a document and rebuild the index |
| POST | `/documents/reindex` | rebuild the FAISS index from stored chunks |
| GET | `/search?question=…&top_k=3` | retrieve + answer; returns `{question, answer, sources}` |
| GET | `/health` | liveness plus index/model/sample state |
| GET | `/metrics` | in-process counters (latency, failures, cache hits, index throughput) |

Frontend components live in `frontend/src/components/` and `frontend/src/pages/`.

## Configuration

Every provider setting is read from the environment (`.env` is loaded at startup and never
committed):

| env var | default | purpose |
|---|---|---|
| `LLM_API_KEY` | — | required; Groq free-tier key (or your OpenAI-compatible key) |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | provider endpoint |
| `LLM_MODEL` | `openai/gpt-oss-120b` | model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model |

## Tests and evaluation

Tests never load torch or hit the network — the model, tokenizer and OpenAI client are
substituted with fakes (`backend/tests/conftest.py`):

```bash
uv run pytest          # 31 tests, fast
```

Offline evaluation on the bundled dataset, against 12 golden question/file pairs, with the real
model and (when the key is set) the real LLM:

```bash
uv run python -m backend.evaluate
```

Measured on the bundled 5-document / 7-chunk corpus with `top_k=3`:

| metric | measured | audit bar |
|---|---|---|
| recall@3 (gold file retrieved) | **100%** (12/12) | ≥ 80% |
| precision@1 (top chunk is gold) | **92%** (11/12) | ≥ 80% |
| answer coverage (key phrase present) | **100%** (12/12) | ≥ 90% |
| indexing throughput (warm model) | **7 chunks/sec** | > 1 chunk/sec |

precision@3 comes out at 42% on this corpus, which is its mathematical ceiling: only 7 chunks
exist and most gold files occupy a single chunk, so at `top_k=3` at most 1 of the 3 retrieved
chunks can belong to the gold file. precision@1 is the meaningful source-precision measure here
and clears 80%.

## Structure

```
backend/
  main.py            FastAPI app, router mounting, exception handlers, logging
  rag.py             chunking + answer drafting (no I/O, no FastAPI)
  embeddings.py      lazy sentence-transformer access
  store.py           on-disk index, chunk records, manifest, log directory
  metrics.py         /metrics counters + CSV logs of queries and failures
  models/record.py   chunk and manifest Pydantic shapes
  routes/            documents, search, health routers
  data/sample-docs/  audit dataset (5 documents)
  tests/             pytest suite (31 tests, no torch)
  evaluate.py        offline evaluation script
frontend/
  src/components/    UploadPanel, SearchPanel, ResultsList
  src/pages/         HomePage
  src/lib/api.js     response/error helpers
```

Runtime state (`chunks.index`, `chunks.json`, `documents.json`) and API key/env files are
gitignored; `backend/logs/` holds the app log plus `queries.csv` and `failures.csv` for a durable
failure-handling record.