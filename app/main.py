from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.ingestion import extract_text, chunk_text
from app.vectorestore import add_chunks, search_chunks, get_documents
from app.rag import generate_answer

app = FastAPI(title="Documents Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k : int = 4

@app.get("/")
def root():
    return{"status":"healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text(file.filename, contents)
    chunks = chunk_text(text)
    n = add_chunks(doc_id=file.filename, chunks = chunks)
    return{"filename": file.filename, "chunks_created": n}

@app.post("/query")
async def query_documents(request: QueryRequest):
    results = search_chunks(request.question, top_k=request.top_k)
    return generate_answer(request.question, results)
@app.get("/documents")
def list_documents():
    return {"documents": get_documents()}
