"""Token-aware chunking for guidely.

Split out of main.py so the chunking rules live in one readable place rather than
in the middle of the request handling.
"""

CHUNK_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

# A trailing chunk shorter than this is folded into the one before it, because a
# handful of tokens on its own embeds into noise.
MIN_CHUNK_TOKENS = 60


def build_chunks(text, tokenizer, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Split text into overlapping chunks of roughly chunk_tokens tokens.

    Counting with the model's own tokenizer is the point: characters are not tokens,
    so a character-based split produces chunks of wildly different real length and
    quietly loses everything past the model's context window.
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