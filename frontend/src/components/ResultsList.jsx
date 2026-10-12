export default function ResultsList({ sources }) {
  return (
    <>
      <h3>Sources</h3>
      <ol className="results">
        {sources.map((source, index) => (
          <li key={index}>
            <span className="filename">{source.file}</span>
            <p className="chunk">{source.text}</p>
            <span className="distance">distance {source.distance.toFixed(3)}</span>
          </li>
        ))}
      </ol>
    </>
  );
}