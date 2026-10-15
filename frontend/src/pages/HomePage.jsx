import { useState } from "react";
import SearchPanel from "../components/SearchPanel";
import { readError, readJson } from "../lib/api";

export default function HomePage({ onNavigateAdmin }) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [search, setSearch] = useState({ state: "idle", answer: "", sources: [], error: "" });

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
      <header className="topbar">
        <div>
          <h1>Guidely</h1>
          <p className="tagline">Ask questions about a game's text.</p>
        </div>
        <button type="button" className="secondary" onClick={onNavigateAdmin}>
          Add documents
        </button>
      </header>
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
