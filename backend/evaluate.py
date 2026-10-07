"""Offline evaluation of the retrieval pipeline on the bundled sample set.

Builds a real index from backend/data/sample-docs with the real embedding model,
then scores retrieval against golden question/file pairs. Question answering is
included when LLM_API_KEY is set, so a full run also measures answer quality.

Usage:
    uv run python -m backend.evaluate

First run loads torch and the model weights, which is slow (~minutes) on this
machine; afterwards the model is cached on disk.
"""

import time

import faiss
from dotenv import load_dotenv

from backend import rag
from backend.embeddings import embed, window
from backend.store import SAMPLE_DIR

# The API key lives in .env; load it so this script can answer questions too.
load_dotenv()

TOP_K = 3

# (question, expected file, phrase the answer should contain)
GOLD = [
    (
        "How long do people have to file a claim against the harbour authority?",
        "harbour-safety-regulations.txt",
        "fourteen days",
    ),
    (
        "Can I get a refund if the ferry never sailed because of weather?",
        "refunds-and-ticket-policy.txt",
        "full refund",
    ),
    (
        "Why did the ferry Neverdine sink?",
        "neverdine-ferry-disaster.txt",
        "storm",
    ),
    (
        "Who must be called when floating cargo is spotted in the sound?",
        "harbour-safety-regulations.txt",
        "harbour office",
    ),
    (
        "What happens to a guild member who does not pay the annual dues?",
        "guild-membership-rules.txt",
        "suspended",
    ),
    (
        "When are the beacon lights relit after the winter?",
        "weather-and-navigation.txt",
        "equinox",
    ),
    (
        "What did the inquiry blame for the sinking of the Neverdine?",
        "neverdine-ferry-disaster.txt",
        "negligence",
    ),
    (
        "Are ferry tickets transferable to another passenger?",
        "refunds-and-ticket-policy.txt",
        "transferable",
    ),
    (
        "How many abandonment drills must a passenger vessel run each year?",
        "harbour-safety-regulations.txt",
        "four",
    ),
    (
        "How does an apprentice become a journeyman?",
        "guild-membership-rules.txt",
        "journeyman",
    ),
    (
        "What should a master do when the sound is closed by fog?",
        "weather-and-navigation.txt",
        "fog",
    ),
    (
        "What happens if the ferry line cancels a crossing?",
        "refunds-and-ticket-policy.txt",
        "refund",
    ),
]


def main() -> None:
    documents = sorted(SAMPLE_DIR.glob("*.txt"))
    print(f"== Guidely evaluation on {len(documents)} sample documents ==")
    for path in documents:
        print(f"   - {path.name}")

    records = []
    for path in documents:
        text = path.read_text(encoding="utf-8")
        records += [
            {"file": path.name, "text": chunk} for chunk in rag.build_chunks(text, window())
        ]
    print(f"\n{len(records)} chunks in the corpus")

    # Embed one throwaway sentence first so the model-load cost (torch import +
    # weights) never lands in the throughput measurement. Throughput is then the
    # real embedding rate for the corpus.
    embed(["warmup"])
    started = time.perf_counter()
    embeddings = embed([record["text"] for record in records])
    indexed_seconds = time.perf_counter() - started
    throughput = len(records) / indexed_seconds
    print(f"indexing: {indexed_seconds:.2f}s -> {throughput:.0f} chunks/sec (model already loaded)")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    lines = []
    hits = 0
    top_1_ok = 0
    precision_sum = 0.0
    answer_ok = 0
    questions_with_answer = 0

    for question, expected_file, phrase in GOLD:
        question_vector = embed([question])
        distances, positions = index.search(question_vector, k=TOP_K)

        retrieved = []
        for distance, position in zip(distances[0], positions[0]):
            if position == -1:
                continue
            record = records[position]
            retrieved.append({**record, "distance": float(distance)})

        hit = any(record["file"] == expected_file for record in retrieved)
        top_1_ok += int(retrieved[0]["file"] == expected_file) if retrieved else 0
        precision = (
            sum(record["file"] == expected_file for record in retrieved) / len(retrieved)
            if retrieved
            else 0.0
        )
        hits += int(hit)
        precision_sum += precision

        covered = "no key"
        try:
            result = rag.answer_question(question, retrieved)
            covered = phrase.lower() in result["answer"].lower()
            questions_with_answer += 1
            answer_ok += int(bool(covered))
        except ValueError:
            covered = "no key"

        lines.append(f"| {question} | {expected_file} | {hit} | {precision:.2f} | {covered} |")

    answered = len(GOLD)
    print("\n| question | expected file | hit@3 | precision@3 | answer ok |")
    print("|---|---|---:|---:|---|")
    print("\n".join(lines))
    print(
        f"\nsummary: recall@{TOP_K} = {hits / answered:.0%} ({hits}/{answered}), "
        f"precision@1 = {top_1_ok / answered:.0%} ({top_1_ok}/{answered}), "
        f"source precision@{TOP_K} = {precision_sum / answered:.0%}"
    )
    if questions_with_answer:
        print(f"answer coverage = {answer_ok / questions_with_answer:.0%} ({answer_ok}/{questions_with_answer})")
    print(f"indexing throughput = {throughput:.0f} chunks/sec (model warm)")


if __name__ == "__main__":
    main()