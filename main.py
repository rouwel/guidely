# FIX: import block re-sorted to satisfy ruff's isort rule (I001)
import io
import json
import logging
from pathlib import Path

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from openai import APIStatusError, APITimeoutError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

# Load the API key from .env into the process before anything reads it, so the
# server works the moment it starts instead of erroring on a missing key.
load_dotenv()

from rag import answer_question, as_record, build_chunks

logger = logging.getLogger(__name__)

app = FastAPI()


# FIX: any exception we did not anticipate used to return Starlette's plain-text
# "Internal Server Error", which the frontend could only report as "Request failed with
# status 500". Keep the traceback in the api log, but send a readable message instead.
@app.exception_handler(Exception)
async def unhandled_error(request, error):
    logger.exception("unhandled error while handling %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error - check the api log for the traceback"},
    )

# FIX: stays at module scope on purpose - create_embeddings and search both need it.
# (It was previously local to create_embeddings, which made search raise NameError.)
# Note: this loads the model at import time, so every worker start pays the cost.
# all-MiniLM-L6-v2 truncates at 256 tokens, so most of a 700-token chunk was
# never embedded. bge-small-en-v1.5 reads 512 tokens, and chunks are now 500,
# keeping the whole chunk inside the model's window.
model_im_using = SentenceTransformer("BAAI/bge-small-en-v1.5")

# FIX: the writer (create_embeddings) and the reader (search) previously hard-coded
# these paths separately, so they could drift apart. One constant, used by both.
INDEX_PATH = Path("chunks.index")
CHUNKS_PATH = Path("chunks.json")


@app.get("/")
def home():
    return {"message": "Hello World"}




# FIX: renamed from Create_Embeddings (PascalCase is for classes, not functions).
# FIX: the parameter is now `document` (str) instead of `contents`, so it no longer
# collides with the `contents` bytes variable in upload_document - passing the bytes
# to the chunker was the easy mistake this naming invited.
def create_embeddings(document: str, filename: str) -> int:
    # FIX: raise ValueError here, where the caller can report it, instead of failing
    # later on embeddings.shape[1] with an IndexError the caller never catches.
    if not document.strip():
        raise ValueError("The uploaded document is empty")

    chunks = build_chunks(document, model_im_using.tokenizer)

    # Store the file name beside every chunk, so a retrieved chunk can say where it
    # came from. Bare strings cannot be cited back to a document.
    records = [{"file": filename, "text": chunk} for chunk in chunks]

    # Generate embeddings from chunks, as float32 for FAISS
    embeddings = model_im_using.encode(chunks).astype("float32")

    dimension = embeddings.shape[1]
    # FIX: faiss.Index.FlatL2 does not exist in faiss-cpu 1.15 - the index classes were
    # flattened out of the `Index` namespace, so every upload died with
    # AttributeError: type object 'Index' has no attribute 'FlatL2'.
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    # FIX: both files used to be written straight to their final names. A crash between
    # the two writes left an index with no chunks file, and a crash *during* the json
    # write left half a file that json.load could never read again - both surfaced later
    # as an opaque 500 from /search. Write to temporary files first, then swap them in,
    # so the pair on disk is always the pair that was completely written.
    # FIX: file handle renamed from `file`, which shadowed the UploadFile parameter name.
    temp_index = INDEX_PATH.with_suffix(".index.tmp")
    temp_chunks = CHUNKS_PATH.with_suffix(".json.tmp")

    faiss.write_index(faiss_index, str(temp_index))
    with open(temp_chunks, "w", encoding="utf-8") as chunks_file:
        json.dump(records, chunks_file, ensure_ascii=False, indent=2)

    temp_index.replace(INDEX_PATH)
    temp_chunks.replace(CHUNKS_PATH)

    return len(records)


# FIX: PDFs are binary, so contents.decode("utf-8") could never work for them. Text
# extraction lives here so upload_document stays a thin route, and so the empty-result
# case (a scanned PDF with no text layer) gets its own message instead of looking like
# an empty upload.
def extract_text(filename: str, contents: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(contents))
            if reader.is_encrypted:
                raise ValueError("This PDF is password protected")
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except PdfReadError as error:
            raise ValueError("This PDF could not be read") from error

        if not text:
            raise ValueError(
                "This PDF has no selectable text - it is probably a scan or an image"
            )
        return text

    return contents.decode("utf-8")


# FIX: this route was nested inside create_embeddings, after its `return`, so the
# decorator never executed and POST /upload/documents was never registered. Dedented
# to module level. (The 9-space `# Read uploaded bytes` comment is gone with it.)
@app.post("/upload/documents")
async def upload_document(file: UploadFile = File(...)):
    # FIX: file.filename is Optional in FastAPI, so a client sending no filename
    # would raise AttributeError inside endswith() instead of getting a 400.
    # FIX: PDF is now accepted alongside txt, case-insensitively.
    allowed = (".txt", ".pdf")
    if not file.filename or not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400, detail="Only .txt and .pdf files are supported"
        )

    # Raw upload stays `contents` (bytes); the extracted text is `document` (str).
    contents = await file.read()

    try:
        document = extract_text(file.filename, contents)
    except UnicodeDecodeError:
        # FIX: `from None` hides the UnicodeDecodeError traceback from the response.
        raise HTTPException(
            status_code=400, detail="The file must be UTF-8 encoded text"
        ) from None
    except ValueError as error:
        # Unreadable, encrypted, or text-less PDF - reported as-is so the user knows why.
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        number_of_chunks = create_embeddings(document, file.filename)
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
def search(question: str = Query(..., min_length=1), top_k: int = Query(2, ge=1, le=20)):
    try:
        index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "r", encoding="utf-8") as chunks_file:
            # Records are {"file", "text"} now; as_record keeps older indexes working.
            chunks = [as_record(chunk) for chunk in json.load(chunks_file)]
    except (FileNotFoundError, RuntimeError):
        # FIX: a missing index is a RuntimeError from the FAISS C++ layer, not a
        # FileNotFoundError, so the old handler never fired and a missing index
        # returned 500 instead of 404. Both are now treated as "nothing uploaded".
        # `from None` stops the FAISS/IO traceback chaining into the response.
        raise HTTPException(
            status_code=404, detail="No document has been uploaded yet"
        ) from None
    except json.JSONDecodeError as error:
        # FIX: an unreadable chunks file used to escape as a bare 500. Say what is
        # wrong and how to fix it, since the only remedy is a fresh upload.
        raise HTTPException(
            status_code=500,
            detail="The stored chunks file is unreadable - upload the document again",
        ) from error

    # FIX: if the index holds more vectors than the chunks file has entries, the lookup
    # below raised IndexError and surfaced as an opaque 500. Catch the mismatch here,
    # where the message can tell the user to upload the document again.
    if index.ntotal != len(chunks):
        raise HTTPException(
            status_code=500,
            detail="The stored index and chunks do not match - upload the document again",
        )

    question_embedding = model_im_using.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, k=top_k)

    results = []
    for distance, index_position in zip(distances[0], indices[0]):
        # -1 means FAISS did not find a result (fewer vectors than top_k)
        if index_position == -1:
            continue

        record = chunks[index_position]
        results.append({**record, "distance": float(distance)})

    # Retrieval alone is not the deliverable: the retrieved chunks go to the model,
    # which writes the answer the UI shows above the sources it used.
    try:
        answered = answer_question(question, results)
    except ValueError as error:
        # No API key configured - a setup problem, not a bad request.
        raise HTTPException(status_code=503, detail=str(error)) from error
    except APITimeoutError as error:
        raise HTTPException(
            status_code=504, detail="The language model timed out, try again"
        ) from error
    except APIStatusError as error:
        if error.status_code == 429:
            # The free tier throttles hard. Tell the user it is a quota reset, not
            # a bug, so they wait a minute instead of replaying the search.
            raise HTTPException(
                status_code=503,
                detail="The free model is rate-limited, wait a minute and try again",
            ) from error
        # Bad key or an empty credit balance both arrive as a status error. Report
        # the code the provider sent instead of letting it surface as a bare 500.
        raise HTTPException(
            status_code=502,
            detail=f"The language model rejected the request (HTTP {error.status_code})",
        ) from error

    return {"question": question, "answer": answered["answer"], "sources": answered["sources"]}
