import os
import hashlib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rag_pipeline import ask, build_pipeline_from_pdf, ask_pdf

app = FastAPI(title="Ask My Company Reports API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# In-memory store of uploaded-PDF pipelines, keyed by session_id.
# Backed by files on disk (in SESSIONS_DIR) so a session can be
# rebuilt if the backend restarts and the in-memory dict is empty.
active_pipelines = {}


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class PdfQuestion(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, max_length=500)


def get_pipeline(session_id):
    """Returns the pipeline for a session, rebuilding it from the
    saved PDF on disk if the backend was restarted and it's no
    longer in memory."""
    if session_id in active_pipelines:
        return active_pipelines[session_id]

    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.pdf")
    if os.path.exists(session_file):
        with open(session_file, "rb") as f:
            file_bytes = f.read()
        pipeline = build_pipeline_from_pdf(file_bytes, source_name=f"{session_id}.pdf")
        active_pipelines[session_id] = pipeline
        return pipeline

    return None


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
    """Upload a PDF, get back a session_id to use with /ask-pdf.
    The PDF is also saved to disk so this session survives a
    backend restart."""
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
    if len(file_bytes) > MAX_FILE_SIZE:
        size_mb = len(file_bytes) / 1_000_000
        return JSONResponse(
            status_code=400,
            content={"error": f"File too large ({size_mb:.1f}MB). Max size is 20MB."}
        )

    try:
        pipeline = build_pipeline_from_pdf(file_bytes, source_name=file.filename)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    # Hash the content for a stable, filename-safe session id
    session_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    active_pipelines[session_id] = pipeline

    with open(os.path.join(SESSIONS_DIR, f"{session_id}.pdf"), "wb") as f:
        f.write(file_bytes)

    return {"session_id": session_id, "message": f"Indexed {file.filename}"}


@app.post("/ask-pdf")
def ask_uploaded_pdf(payload: PdfQuestion):
    """Ask a question about a previously uploaded PDF, using its session_id."""
    pipeline = get_pipeline(payload.session_id)
    if pipeline is None:
        return {"error": "Session not found. Upload the PDF again."}

    answer = ask_pdf(payload.question, pipeline)
    return {"answer": answer}