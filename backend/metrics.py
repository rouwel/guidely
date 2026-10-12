"""In-memory counters for /metrics plus CSV logs of searches and failures.

Counters reset on restart on purpose: the report describes the current process.
Every search and every failure is also appended to a CSV file, so there is a
durable record the audit can read even after a restart.
"""

import statistics
from collections import deque
from datetime import UTC, datetime

from .store import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _csv_escape(value) -> str:
    return str(value).replace(",", " ").replace("\n", " ")


def _append_csv(filename: str, fields: list, row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / filename
    with open(path, "a", encoding="utf-8") as handle:
        if path.stat().st_size == 0:
            handle.write(",".join(fields) + "\n")
        handle.write(",".join(_csv_escape(row[field]) for field in fields) + "\n")


class Metrics:
    def __init__(self) -> None:
        self.searches = 0
        self.uploads = 0
        self.cache_hits = 0
        self.latencies_ms = deque(maxlen=500)
        self.failures = {
            "missing_index": 0,
            "corrupt_chunks": 0,
            "index_mismatch": 0,
            "missing_key": 0,
            "timeout": 0,
            "rate_limited": 0,
            "provider": 0,
        }
        self.last_index = {"chunks": 0, "duration_seconds": 0.0, "throughput": 0.0}

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_upload(self, chunks: int, duration_seconds: float) -> None:
        self.uploads += 1
        self.last_index = {
            "chunks": chunks,
            "duration_seconds": round(duration_seconds, 3),
            "throughput": round(chunks / duration_seconds, 2) if duration_seconds else 0.0,
        }

    def record_search(self, elapsed_ms: float, question: str, retrieved: int) -> None:
        self.searches += 1
        self.latencies_ms.append(elapsed_ms)
        _append_csv(
            "queries.csv",
            ["time", "question", "elapsed_ms", "retrieved"],
            {
                "time": _now(),
                "question": question,
                "elapsed_ms": round(elapsed_ms, 2),
                "retrieved": retrieved,
            },
        )

    def record_failure(self, kind: str, detail: str = "") -> None:
        self.failures[kind] = self.failures.get(kind, 0) + 1
        _append_csv(
            "failures.csv",
            ["time", "kind", "detail"],
            {"time": _now(), "kind": kind, "detail": detail},
        )

    def report(self) -> dict:
        latencies = sorted(self.latencies_ms)
        return {
            "searches": self.searches,
            "uploads": self.uploads,
            "cache_hits": self.cache_hits,
            "latency_ms": {
                "median": round(statistics.median(latencies), 2) if latencies else 0.0,
                "p95": (
                    round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 2)
                    if latencies
                    else 0.0
                ),
                "samples": len(latencies),
            },
            "failures": dict(self.failures),
            "last_index": dict(self.last_index),
        }


metrics = Metrics()