from fastapi import FastAPI, File,UploadFile
import json
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.post("/upload/documents")
def upload_document(file: UploadFile = File(...)):
    contents = file.read()
    file_size = len(contents)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": file_size,
        "message": "Document received successfully!"
    }

def create_Chunks(doc, chunk_size):
    chunks = []
    i = 0
    while i < len(doc):
        chunk = doc[i:i+chunk_size]

        chunk.append(chunks)
    return chunks

def Create_Embeddings(contents):
    chunks_1 = create_Chunks(contents, 100)
    #load an e0bedd5ng model
    model_im_using = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #Generate embeddings from chunks
    embeddings = model_im_using.encode(chunks_1)
    # Convert embeddings to float 32
    embeddings = embeddings.astype("float32")
    # create a faiss index
    dimension = embeddings.shape[1]
    faiss_index = faiss.Index.FlatL2(dimension)
    # save embeddings to index
    faiss_index.add(embeddings)
    # save index to disl
    faiss.write_index(faiss_index, "chunks.index")
    # save chunks to json file
    with open("chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks_1, file, ensure_ascii=False, indent=2)
