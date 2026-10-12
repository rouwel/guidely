import ResultsList from "./ResultsList";

export default function SearchPanel({ search, question, topK, onQuestion, onTopK, onSubmit }) {
  return (
    <section>
      <h2>2. Ask a question</h2>
      <form onSubmit={onSubmit}>
        <input
          type="text"
          value={question}
          placeholder="Why did the ferry sink?"
          onChange={(event) => onQuestion(event.target.value)}
        />
        <label>
          chunks
          <input
            type="number"
            min="1"
            max="20"
            value={topK}
            onChange={(event) => onTopK(Number(event.target.value))}
          />
        </label>
        <button type="submit" disabled={search.state === "searching" || !question.trim()}>
          {search.state === "searching" ? "Asking..." : "Ask"}
        </button>
      </form>
      {search.state === "error" && <p className="error">{search.error}</p>}
      {search.state === "done" && (
        <>
          <p className="answer">{search.answer}</p>
          <ResultsList sources={search.sources} />
        </>
      )}
    </section>
  );
}