import { useState } from "react";

// FastAPI puts string errors in `detail`, but validation errors arrive as a list of
// objects, so normalise both into something we can drop straight into the UI.
function readError(payload, response) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join("; ");
  return `Request failed with status ${response.status}`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState({ state: "idle", message: "", chunks: null });

  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [search, setSearch] = useState({ state: "idle", results: [], error: "" });

  async function handleUpload(event) {
    event.preventDefault();
    if (!file) return;

    setUpload({ state: "uploading", message: "", chunks: null });
    setSearch({ state: "idle", results: [], error: "" });

    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch("/upload/documents", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(readError(payload, response));

      setUpload({
        state: "done",
        message: payload.message,
        chunks: payload.chunks_stored,
      });
    } catch (error) {
      setUpload({ state: "error", message: error.message, chunks: null });
    }
  }

  async function handleSearch(event) {
    event.preventDefault();
    if (!question.trim()) return;

    setSearch({ state: "searching", results: [], error: "" });

    const params = new URLSearchParams({ question, top_k: String(topK) });
    try {
      const response = await fetch(`/search?${params}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(readError(payload, response));

      setSearch({ state: "done", results: payload.results, error: "" });
    } catch (error) {
      setSearch({ state: "error", results: [], error: error.message });
    }
  }

  return (
    <main>
      <h1>Guidely</h1>
      <p className="tagline">Ask questions about a game's text.</p>

      <section>
        <h2>1. Upload a document</h2>
        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept=".txt"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <button type="submit" disabled={!file || upload.state === "uploading"}>
            {upload.state === "uploading" ? "Embedding..." : "Upload"}
          </button>
        </form>
        {file && <p className="hint">Selected: {file.name}</p>}
        {upload.state === "done" && (
          <p className="ok">
            {upload.message} ({upload.chunks} chunks stored)
          </p>
        )}
        {upload.state === "error" && <p className="error">{upload.message}</p>}
      </section>

      <section>
        <h2>2. Ask a question</h2>
        <form onSubmit={handleSearch}>
          <input
            type="text"
            value={question}
            placeholder="Why did the ferry sink?"
            onChange={(event) => setQuestion(event.target.value)}
          />
          <label>
            chunks
            <input
              type="number"
              min="1"
              max="20"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <button type="submit" disabled={search.state === "searching" || !question.trim()}>
            {search.state === "searching" ? "Searching..." : "Search"}
          </button>
        </form>
        {search.state === "error" && <p className="error">{search.error}</p>}
        {search.results.length > 0 && (
          <ol className="results">
            {search.results.map((result, index) => (
              <li key={index}>
                <p className="chunk">{result.chunk}</p>
                <span className="distance">distance {result.distance.toFixed(3)}</span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </main>
  );
}
