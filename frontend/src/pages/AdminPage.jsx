import { useState } from "react";
import UploadPanel from "../components/UploadPanel";
import { readError, readJson } from "../lib/api";

export default function AdminPage({ onNavigateHome }) {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState({ state: "idle", message: "", chunks: null });
  const [reindex, setReindex] = useState({ state: "idle", message: "", error: "" });

  async function handleUpload(event) {
    event.preventDefault();
    if (!file) return;

    setUpload({ state: "uploading", message: "", chunks: null });

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

  async function handleReindex() {
    setReindex({ state: "running", message: "", error: "" });

    try {
      const response = await fetch("/documents/reindex", { method: "POST" });
      const result = await readJson(response);
      if (!response.ok) throw new Error(readError(result, response));

      setReindex({
        state: "done",
        message: `${result.payload.message} (${result.payload.chunks_stored} chunks)`,
        error: "",
      });
    } catch (error) {
      setReindex({ state: "error", message: "", error: error.message });
    }
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>Admin</h1>
          <p className="tagline">Manage the indexed documents.</p>
        </div>
        <button type="button" className="secondary" onClick={onNavigateHome}>
          Back to search
        </button>
      </header>
      <UploadPanel file={file} upload={upload} onFile={setFile} onSubmit={handleUpload} />
      <section>
        <h2>Re-index documents</h2>
        <p className="hint">Rebuild the search index from every stored chunk.</p>
        <button type="button" onClick={handleReindex} disabled={reindex.state === "running"}>
          {reindex.state === "running" ? "Re-indexing..." : "Re-index documents"}
        </button>
        {reindex.state === "done" && <p className="ok">{reindex.message}</p>}
        {reindex.state === "error" && <p className="error">{reindex.error}</p>}
      </section>
    </main>
  );
}
