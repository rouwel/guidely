import { useState } from "react";
import SearchPanel from "../components/SearchPanel";
import UploadPanel from "../components/UploadPanel";
import { readError, readJson } from "../lib/api";

export default function HomePage() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState({ state: "idle", message: "", chunks: null });

  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [search, setSearch] = useState({ state: "idle", answer: "", sources: [], error: "" });

  async function handleUpload(event) {
    event.preventDefault();
    if (!file) return;

    setUpload({ state: "uploading", message: "", chunks: null });
    setSearch({ state: "idle", answer: "", sources: [], error: "" });

    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch("/upload/documents", { method: "POST", body });
      const result = await readJson(response);
      if (!response.ok) throw new Error(readError(result, response));

      setUpload({
        state: "done",
        message: result.payload.message,
        chunks: result.payload.chunks_stored,
      });
    } catch (error) {
      setUpload({ state: "error", message: error.message, chunks: null });
    }
  }

  async function handleSearch(event) {
    event.preventDefault();
    if (!question.trim()) return;

    setSearch({ state: "searching", answer: "", sources: [], error: "" });

    const params = new URLSearchParams({ question, top_k: String(topK) });
    try {
      const response = await fetch(`/search?${params}`);
      const result = await readJson(response);
      if (!response.ok) throw new Error(readError(result, response));

      setSearch({
        state: "done",
        answer: result.payload.answer ?? "",
        sources: result.payload.sources ?? [],
        error: "",
      });
    } catch (error) {
      setSearch({ state: "error", answer: "", sources: [], error: error.message });
    }
  }

  return (
    <main>
      <h1>Guidely</h1>
      <p className="tagline">Ask questions about a game's text.</p>
      <UploadPanel file={file} upload={upload} onFile={setFile} onSubmit={handleUpload} />
      <SearchPanel
        search={search}
        question={question}
        topK={topK}
        onQuestion={setQuestion}
        onTopK={setTopK}
        onSubmit={handleSearch}
      />
    </main>
  );
}