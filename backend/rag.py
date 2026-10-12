"""Chunking, answer drafting and stored-chunk helpers for guidely.

Split out of main.py so the pipeline rules read in one place rather than in the
middle of the request handling:

    build_chunks      slice text the way the embedding model reads it
    answer_question   turn retrieved chunks into an answer plus its sources
    as_record         normalise a stored chunk

None of these touch FastAPI. ValueError means the caller can turn it into a
clear 503; the provider's own errors are left alone so the route can tell a
missing key from a timeout from a rejected request.

The provider defaults to Groq, which speaks the OpenAI wire format, so the same
openai client works - only the base URL and the key differ. Every setting is
overridable through the environment (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL), so a
different account or model never needs a code change.
"""

import os
import re

from openai import OpenAI

# 500 tokens sits inside both the brief's 500-1000 range and the embedding
# model's 512-token window, so every word of a chunk is embedded - see
# backend/embeddings.py.
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

# A trailing chunk shorter than this is folded into the one before it, because a
# handful of tokens on its own embeds into noise.
MIN_CHUNK_TOKENS = 60

# Groq's free tier: an emergency-rescue 120B model, no card needed, roughly a
# thousand questions a day. Override any of these with the matching env var.
ANSWER_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
TIMEOUT_SECONDS = 30

PROMPT = (
    "You answer questions about a set of documents. Use only the numbered sources "
    "below. If they do not contain the answer, say so plainly instead of guessing. "
    "Keep it to a short paragraph and quote the wording you relied on. Do not add "
    "any citations, page numbers or reference markers to your answer."
)


def build_chunks(text, tokenizer, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Split text into overlapping chunks of roughly chunk_tokens tokens.

    Counting with the model's own tokenizer is the point: characters are not
    tokens, so a character-based split produces chunks of wildly different real
    length and quietly loses everything past the model's context window.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(1, chunk_tokens - overlap)

    windows = []
    start = 0
    while start < len(token_ids):
        windows.append(token_ids[start : start + chunk_tokens])
        if start + chunk_tokens >= len(token_ids):
            break
        start += step

    chunks = [tokenizer.decode(window) for window in windows]

    if len(chunks) > 1 and len(windows[-1]) < MIN_CHUNK_TOKENS:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()

    return chunks


def as_record(chunk):
    """Normalise a stored chunk to {"file", "text"}.

    Older indexes hold bare strings, so an index written before filenames were
    kept still searches instead of raising.
    """
    if isinstance(chunk, dict):
        return {"file": chunk.get("file", "unknown"), "text": chunk["text"]}
    return {"file": "unknown", "text": chunk}


def answer_question(question, sources):
    """Ask an LLM to answer question from the retrieved sources.

    `sources` is a list of {"file", "text"} records. Returns {"answer",
    "sources"}, where the sources are the same records that were sent to the
    model, so the caller can show the answer next to the text it came from.
    """
    if not sources:
        return {
            "answer": "Nothing in the indexed documents covers that question.",
            "sources": [],
        }

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is not set - export it or add it to .env")

    context = "\n\n".join(
        f"[{number}] file: {source['file']}\n{source['text']}"
        for number, source in enumerate(sources, start=1)
    )

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    # Some providers bolt automatic citation markers on to answers that came
    # from retrieved text. The sources are already shown next to the answer, so
    # the markers are only noise in the UI.
    answer = re.sub(r"\uff0c?\u3010[^\u3011]*\u3011|\u3010[^\u3011]*\u3011", "", answer).strip()

    return {"answer": answer, "sources": sources}