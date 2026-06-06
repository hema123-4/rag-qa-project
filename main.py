from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_chain import build_qa_chain
from embeddings import create_vectorstore
import shutil, os

app = FastAPI(
    title="RAG Document Q&A",
    description="Upload a PDF and ask questions with cited answers"
)

qa_chain = None
retriever = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain, retriever
    save_path = f"uploaded_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    create_vectorstore(save_path)
    qa_chain, retriever = build_qa_chain()   # ← unpack both
    os.remove(save_path)
    return {"message": "PDF processed successfully", "status": "ready"}

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if not qa_chain:
        return {"error": "Please upload a PDF first via /upload-pdf"}
    answer = qa_chain.invoke(req.question)
    sources = retriever.invoke(req.question)
    return {
    "question": req.question,
    "answer": " ".join(answer.replace("\\n", " ").replace("\n", " ").replace("**", "").replace('\\"', "'").replace('"', "'").split()).strip(),
    "sources": [doc.page_content[:150] for doc in sources]
}

@app.get("/")
def root():
    return {"message": "RAG Q&A API running. Visit /docs to test."}