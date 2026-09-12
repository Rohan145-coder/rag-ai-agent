from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_pipeline import ask, build_pipeline_from_pdf, ask_pdf

app = FastAPI(title="Ask My Company Reports API")

active_pipelines = {}


class Question(BaseModel):
    question: str


class PdfQuestion(BaseModel):
    session_id: str
    question: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/ask")
def ask_tesla(payload: Question):
    """Ask a question about the fixed Tesla 10-K dataset."""
    answer = ask(payload.question)
    return {"answer": answer}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, get back a session_id to use with /ask-pdf."""
    file_bytes = await file.read()
    pipeline = build_pipeline_from_pdf(file_bytes, source_name=file.filename)

    session_id = file.filename + "_" + str(len(file_bytes))
    active_pipelines[session_id] = pipeline

    return {"session_id": session_id, "message": f"Indexed {file.filename}"}


@app.post("/ask-pdf")
def ask_uploaded_pdf(payload: PdfQuestion):
    """Ask a question about a previously uploaded PDF, using its session_id."""
    pipeline = active_pipelines.get(payload.session_id)
    if pipeline is None:
        return {"error": "Session not found. Upload the PDF again."}

    answer = ask_pdf(payload.question, pipeline)
    return {"answer": answer}