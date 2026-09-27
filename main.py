from fastapi import FastAPI, UploadFile, File

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
