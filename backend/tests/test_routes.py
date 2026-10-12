"""End-to-end route tests against the FastAPI app with fakes substituted in."""

import json

from backend import store as store_module

DOC_TEXT = (
    "The ferry neverdine sank in a winter storm. Claims against the harbour "
    "authority must be filed within fourteen days."
)


def _upload(client, name="notes.txt", content=DOC_TEXT):
    return client.post(
        "/upload/documents",
        files={"file": (name, content.encode("utf-8"), "text/plain")},
    )


def test_upload_then_search_roundtrip(client):
    upload = _upload(client)
    assert upload.status_code == 200
    payload = upload.json()
    assert payload["cached"] is False
    assert payload["chunks_stored"] >= 1
    assert payload["filename"] == "notes.txt"

    docs = client.get("/documents").json()
    assert docs["count"] == 1
    assert docs["documents"][0]["filename"] == "notes.txt"

    response = client.get("/search", params={"question": "why did the ferry sink?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    # Citation markers from the stubbed provider are stripped before the UI sees them.
    assert body["answer"] == "A short answer."
    assert body["sources"]
    assert body["sources"][0]["file"] == "notes.txt"
    assert "distance" in body["sources"][0]


def test_upload_same_bytes_is_cached(client):
    first = _upload(client, "a.txt")
    second = _upload(client, "b.txt")  # different name, identical content
    assert second.status_code == 200
    body = second.json()
    assert body["cached"] is True
    assert body["chunks_stored"] == first.json()["chunks_stored"]
    assert client.get("/metrics").json()["cache_hits"] == 1


def test_upload_multiple_documents_then_delete(client):
    _upload(client, "a.txt")
    _upload(client, "b.txt", content="Guild dues are payable at the spring festival.")
    assert client.get("/documents").json()["count"] == 2

    response = client.delete("/documents/a.txt")
    assert response.status_code == 200

    remaining = client.get("/documents").json()
    assert remaining["count"] == 1
    assert remaining["documents"][0]["filename"] == "b.txt"

    # The index was rebuilt from the surviving chunks, so search cites only b.txt.
    body = client.get("/search", params={"question": "guild"}).json()
    assert {source["file"] for source in body["sources"]} == {"b.txt"}


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/documents/missing.txt").status_code == 404


def test_delete_last_document_clears_index(client):
    _upload(client, "a.txt")
    assert client.delete("/documents/a.txt").status_code == 200
    assert client.get("/documents").json()["count"] == 0
    assert client.get("/search", params={"question": "anything"}).status_code == 404


def test_search_before_upload_returns_404(client):
    assert client.get("/search", params={"question": "hello"}).status_code == 404


def test_search_empty_question_returns_422(client):
    assert client.get("/search", params={"question": ""}).status_code == 422


def test_search_missing_question_param_returns_422(client):
    assert client.get("/search").status_code == 422


def test_search_without_api_key_returns_503(client, monkeypatch):
    _upload(client)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert client.get("/search", params={"question": "hello"}).status_code == 503


def test_reindex_rebuilds_index(client):
    _upload(client)
    response = client.post("/documents/reindex")
    assert response.status_code == 200
    assert response.json()["chunks_stored"] >= 1
    assert client.get("/search", params={"question": "ferry"}).status_code == 200


def test_reindex_without_documents_returns_404(client):
    assert client.post("/documents/reindex").status_code == 404


def test_upload_rejects_bad_extension(client):
    response = client.post(
        "/upload/documents",
        files={"file": ("notes.docx", b"hello", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "txt and .pdf" in response.json()["detail"]


def test_upload_rejects_non_utf8_text(client):
    response = client.post(
        "/upload/documents",
        files={"file": ("bad.txt", b"\xff\xfe\x00", "text/plain")},
    )
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_upload_rejects_scan_pdf_without_text(client):
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF"
    )
    response = client.post(
        "/upload/documents",
        files={"file": ("scan.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 400


def test_index_chunk_mismatch_returns_500(client):
    _upload(client)
    records = json.loads(store_module.CHUNKS_PATH.read_text(encoding="utf-8"))
    records.append({"file": "ghost.txt", "text": "a vector exists with no record"})
    store_module.CHUNKS_PATH.write_text(json.dumps(records), encoding="utf-8")
    assert client.get("/search", params={"question": "hi"}).status_code == 500


def test_corrupt_chunks_file_returns_500(client):
    _upload(client)
    store_module.CHUNKS_PATH.write_text("{not json", encoding="utf-8")
    response = client.get("/search", params={"question": "hi"})
    assert response.status_code == 500
    assert "unreadable" in response.json()["detail"]


def test_health_reports_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] == 0
    assert body["index_present"] is False
    assert body["model_loaded"] is False
    assert body["sample_documents"] > 0


def test_metrics_report_shape(client):
    client.get("/metrics")
    body = client.get("/metrics").json()
    assert set(body["failures"]) == {
        "missing_index",
        "corrupt_chunks",
        "index_mismatch",
        "missing_key",
        "timeout",
        "rate_limited",
        "provider",
    }
    assert body["latency_ms"]["samples"] == 0
    assert body["last_index"]["throughput"] == 0.0


def test_metrics_count_searches_and_failures(client):
    assert client.get("/search", params={"question": "boom"}).status_code == 404
    _upload(client)
    response = client.get("/search", params={"question": "when is the refund? top_k ignored", "top_k": 2})
    assert response.status_code == 200

    body = client.get("/metrics").json()
    # Only the second search completed; the first failed before a lookup count.
    assert body["searches"] == 1
    assert body["failures"]["missing_index"] == 1
    assert body["latency_ms"]["samples"] == 1


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Guidely" in response.json()["message"]