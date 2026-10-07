export default function UploadPanel({ file, upload, onFile, onSubmit }) {
  return (
    <section>
      <h2>1. Upload a document</h2>
      <form onSubmit={onSubmit}>
        <input
          type="file"
          accept=".txt,.pdf"
          onChange={(event) => onFile(event.target.files?.[0] ?? null)}
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
  );
}