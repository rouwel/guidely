# Guidely

A retrieval-augmented generation (RAG) system for interrogating game text.

Point it at a game's text files — dialogue dumps, logs, codex entries, patch notes — and
ask questions in plain language. It answers with the exact passages it drew from, plus the
distance of each match, so you can trace an answer back to the line that produced it instead
of trusting a summary.

This is aimed at the kind of reading ARG players do anyway: piecing together lore that the
game never states outright. Character histories, vanished events, factions, places that only
appear in one translation, references that resolve only when you have read all 40,000 words
at once.

## How it works

1. **Ingest** — an uploaded text file is decoded, then split into fixed-size chunks.
2. **Embed** — each chunk is converted to a vector by a `SentenceTransformer`
   (`all-MiniLM-L6-v2`) and written to a FAISS index; the chunks themselves are kept in JSON
   so every vector can be traced back to its source text.
3. **Retrieve** — a question is embedded with the same model and the nearest chunks are
   pulled from the index by L2 distance.
4. **Answer** — the retrieved chunks come back ranked, closest first.

Semantic rather than keyword search is the point: it finds the passage that *means* your
question even when it never uses the words you searched for. Ask "why did the ferry sink" and
you get the log entry about the harbour, not just every line containing "ferry".
