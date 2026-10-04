# FIX: import block re-sorted to satisfy ruff's isort rule (I001)
import json
from pathlib import Path

import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from sentence_transformers import SentenceTransformer

app = FastAPI()

# FIX: stays at module scope on purpose - create_embeddings and search both need it.
# (It was previously local to create_embeddings, which made search raise NameError.)
# Note: this loads the model at import time, so every worker start pays the cost.
model_im_using = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FIX: the writer (create_embeddings) and the reader (search) previously hard-coded
# these paths separately, so they could drift apart. One constant, used by both.
INDEX_PATH = Path("chunks.index")
CHUNKS_PATH = Path("chunks.json")


@app.get("/")
def home():
    return {"message": "Hello World"}


# FIX: renamed from create_Chunks to snake_case, matching every other function.
# FIX: `chunk.append(chunks)` had the receiver and the target swapped. It raised
# AttributeError ('str' object has no attribute 'append') and left `chunks` empty,
# so every embedding so far was built from zero documents.
def create_chunks(doc: str, chunk_size: int) -> list[str]:
    chunks = []
    i = 0
    while i < len(doc):
        chunks.append(doc[i : i + chunk_size])
        i += chunk_size
    return chunks


# FIX: renamed from Create_Embeddings (PascalCase is for classes, not functions).
# FIX: the parameter is now `document` (str) instead of `contents`, so it no longer
# collides with the `contents` bytes variable in upload_document - passing the bytes
# to create_chunks was the easy mistake this naming invited.
def create_embeddings(document: str) -> int:
    # FIX: raise ValueError here, where the caller can report it, instead of failing
    # later on embeddings.shape[1] with an IndexError the caller never catches.
    if not document.strip():
        raise ValueError("The uploaded document is empty")

    chunks = create_chunks(document, 100)

    # Generate embeddings from chunks, as float32 for FAISS
    embeddings = model_im_using.encode(chunks).astype("float32")

    dimension = embeddings.shape[1]
    # FIX: faiss.Index.FlatL2 does not exist in faiss-cpu 1.15 - the index classes were
    # flattened out of the `Index` namespace, so every upload died with
    # AttributeError: type object 'Index' has no attribute 'FlatL2'.
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(INDEX_PATH))

    # FIX: file handle renamed from `file`, which shadowed the UploadFile parameter
    # name used elsewhere in this module.
    with open(CHUNKS_PATH, "w", encoding="utf-8") as chunks_file:
        json.dump(chunks, chunks_file, ensure_ascii=False, indent=2)

    return len(chunks)


# FIX: this route was nested inside create_embeddings, after its `return`, so the
# decorator never executed and POST /upload/documents was never registered. Dedented
# to module level. (The 9-space `# Read uploaded bytes` comment is gone with it.)
@app.post("/upload/documents")
async def upload_document(file: UploadFile = File(...)):
    # FIX: file.filename is Optional in FastAPI, so a client sending no filename
    # would raise AttributeError inside endswith() instead of getting a 400.
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    # Raw upload stays `contents` (bytes); the decoded text is `document` (str).
    contents = await file.read()

    try:
        document = contents.decode("utf-8")
    except UnicodeDecodeError:
        # FIX: `from None` hides the UnicodeDecodeError traceback from the response.
        raise HTTPException(
            status_code=400, detail="The file must be UTF-8 encoded text"
        ) from None

    try:
        number_of_chunks = create_embeddings(document)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "filename": file.filename,
        "message": "Document embedded and stored successfully",
        "chunks_stored": number_of_chunks,
    }


# FIX: search was a plain function with no decorator, so it was unreachable over
# HTTP and `question` had no way to arrive. It is now a route with a query param.
# FIX: faiss returns (distances, indices) - `positions` was a misleading name.
@app.get("/search")
def search(question: str, top_k: int = 2):
    try:
        index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "r", encoding="utf-8") as chunks_file:
            # Same name as in create_embeddings, for the same payload.
            chunks = json.load(chunks_file)
    except (FileNotFoundError, RuntimeError):
        # FIX: a missing index is a RuntimeError from the FAISS C++ layer, not a
        # FileNotFoundError, so the old handler never fired and a missing index
        # returned 500 instead of 404. Both are now treated as "nothing uploaded".
        # `from None` stops the FAISS/IO traceback chaining into the response.
        raise HTTPException(
            status_code=404, detail="No document has been uploaded yet"
        ) from None

    question_embedding = model_im_using.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, k=top_k)

    results = []
    for distance, index_position in zip(distances[0], indices[0]):
        # -1 means FAISS did not find a result (fewer vectors than top_k)
        if index_position == -1:
            continue

        results.append(
            {"chunk": chunks[index_position], "distance": float(distance)}
        )

    return {"question": question, "results": results}
