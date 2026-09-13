import os
import io
import json
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from pypdf import PdfReader

load_dotenv()

# --- Fix: fail fast with a clear message if the API key is missing,
# instead of a confusing error later when the first LLM call is made ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Create a .env file in the project root with:\n"
        "GROQ_API_KEY=your_key_here\n"
        "Get a free key at https://console.groq.com/keys"
    )
groq_client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "tesla_10k"

# --- Load everything ONCE, not every time we ask a question ---
reader = PdfReader("data/tesla.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(full_text)
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def ask(question):
    """Takes a question, returns the final cited answer as a string.
    Wired to the fixed Tesla dataset. Used by main.py / run_eval.py / CI."""

    tokenized_question = question.split()
    bm25_scores = bm25.get_scores(tokenized_question)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:8]
    bm25_results = [chunks[i] for i in top_bm25_indices]

    question_embedding = embed_model.encode(question).tolist()
    vector_results_raw = collection.query(query_embeddings=[question_embedding], n_results=8)
    vector_results = vector_results_raw["documents"][0]

    combined_results = list(dict.fromkeys(bm25_results + vector_results))

    pairs = [[question, chunk] for chunk in combined_results]
    rerank_scores = reranker.predict(pairs)
    scored_chunks = list(zip(combined_results, rerank_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in scored_chunks[:5]]

    context_text = ""
    for i, chunk in enumerate(top_chunks):
        context_text += f"\n[Source {i+1}]:\n{chunk}\n"

    system_prompt = """You are a financial document assistant. Answer the user's question using ONLY the information in the sources provided below.

Rules:
1. Every fact in your answer MUST be followed by a citation like [Source 1] or [Source 2].
2. If the sources don't contain the answer, say "I cannot find this information in the provided documents" — do NOT guess or use outside knowledge.
3. Keep your answer concise and directly focused on the question.
"""
    user_prompt = f"Sources:\n{context_text}\n\nQuestion: {question}\n\nAnswer with citations:"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


# ============================================================
# Support for uploading ANY PDF (used by the Streamlit app)
# ============================================================

def build_pipeline_from_pdf(file_bytes, source_name="uploaded.pdf"):
    """Builds a fresh chunk index + BM25 index + Chroma collection for
    ANY uploaded PDF. Raises ValueError with a clear message for bad
    input (corrupted file, or a PDF with no extractable text) instead
    of crashing deep inside with a confusing traceback."""

    try:
        reader_ = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(
            f"Could not read '{source_name}' - it may be corrupted or password-protected."
        ) from e

    full_text_ = ""
    for page in reader_.pages:
        full_text_ += page.extract_text() or ""

    if not full_text_.strip():
        raise ValueError(
            f"No readable text found in '{source_name}'. It may be a scanned/image-only "
            "PDF without OCR, or genuinely empty."
        )

    pdf_chunks = chunk_text(full_text_)
    tokenized = [c.split() for c in pdf_chunks]
    pdf_bm25 = BM25Okapi(tokenized)

    temp_client = chromadb.EphemeralClient()
    collection_name = f"upload_{abs(hash(source_name + str(len(file_bytes))))}"
    temp_collection = temp_client.get_or_create_collection(name=collection_name)

    embeddings = embed_model.encode(pdf_chunks).tolist()
    temp_collection.add(
        documents=pdf_chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(pdf_chunks))]
    )

    return {"chunks": pdf_chunks, "bm25": pdf_bm25, "collection": temp_collection}


def retrieve_and_rerank(question, chunks_, bm25_, collection_, top_k=5, candidates=8):
    """Shared hybrid search + rerank logic, reusable by any pipeline variant."""
    tokenized_question = question.split()
    bm25_scores = bm25_.get_scores(tokenized_question)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidates]
    bm25_results = [chunks_[i] for i in top_bm25_indices]

    question_embedding = embed_model.encode(question).tolist()
    vector_results_raw = collection_.query(
        query_embeddings=[question_embedding], n_results=min(candidates, len(chunks_))
    )
    vector_results = vector_results_raw["documents"][0]

    combined_results = list(dict.fromkeys(bm25_results + vector_results))

    pairs = [[question, chunk] for chunk in combined_results]
    rerank_scores = reranker.predict(pairs)
    scored_chunks = list(zip(combined_results, rerank_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]


def ask_pdf(question, pipeline):
    """Answers a question about an uploaded PDF (plain text, no image handling)."""
    chunks_ = pipeline["chunks"]
    bm25_ = pipeline["bm25"]
    collection_ = pipeline["collection"]

    top_chunks = retrieve_and_rerank(question, chunks_, bm25_, collection_)

    context_text = ""
    for i, chunk in enumerate(top_chunks):
        context_text += f"\n[Source {i+1}]:\n{chunk}\n"

    system_prompt = """You are a document assistant. Answer the user's question using ONLY the information in the sources provided below.

Rules:
1. Every fact in your answer MUST be followed by a citation like [Source 1] or [Source 2].
2. If the sources don't contain the answer, say "I cannot find this information in the provided documents" — do NOT guess or use outside knowledge.
3. Keep your answer concise and directly focused on the question.
"""
    user_prompt = f"Sources:\n{context_text}\n\nQuestion: {question}\n\nAnswer with citations:"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


# ============================================================
# Agentic decision loop (used by test_agentic.py)
# ============================================================

def assess_retrieval(question, chunks_):
    """Judges retrieval as SUFFICIENT / INSUFFICIENT / AMBIGUOUS."""
    context_text = "\n\n".join(chunks_)
    prompt = f"""Given these document excerpts:

{context_text}

Question: "{question}"

Decide ONE of the following:
- SUFFICIENT: the excerpts clearly and unambiguously answer this question
- INSUFFICIENT: the excerpts don't contain the answer at all
- AMBIGUOUS: the excerpts contain MULTIPLE different, similarly-named figures or facts that could each answer this question, and the question's current wording doesn't make clear which one is meant

Respond with ONLY valid JSON, nothing else, in this exact shape:
{{"verdict": "SUFFICIENT" or "INSUFFICIENT" or "AMBIGUOUS", "clarifying_question": "a specific question to ask the user to disambiguate, or null if not AMBIGUOUS"}}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"verdict": "SUFFICIENT", "clarifying_question": None}

    return result


def rewrite_query(question):
    """Rewrites the question to be more specific/searchable."""
    prompt = f"""The following question did not retrieve good enough results from a document search:

"{question}"

Rewrite it as a more specific, keyword-rich search query that would retrieve better matching passages from a financial document. Reply with ONLY the rewritten query, nothing else."""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def ask_agentic(question, max_retries=1):
    """Agentic version of ask(): decides at runtime whether retrieval
    is sufficient, insufficient (retry), or ambiguous (ask user back)."""
    debug_info = {"retries_used": 0, "queries_tried": [question], "clarification_needed": False}

    current_question = question
    top_chunks = retrieve_and_rerank(current_question, chunks, bm25, collection)

    attempt = 0
    while attempt < max_retries:
        assessment = assess_retrieval(question, top_chunks)
        verdict = assessment.get("verdict", "SUFFICIENT")

        if verdict == "SUFFICIENT":
            break

        if verdict == "AMBIGUOUS":
            debug_info["clarification_needed"] = True
            clarifying_q = assessment.get("clarifying_question") or \
                "Could you clarify which specific figure you mean?"
            return clarifying_q, debug_info

        current_question = rewrite_query(question)
        debug_info["queries_tried"].append(current_question)
        top_chunks = retrieve_and_rerank(current_question, chunks, bm25, collection)
        attempt += 1

    debug_info["retries_used"] = attempt

    context_text = ""
    for i, chunk in enumerate(top_chunks):
        context_text += f"\n[Source {i+1}]:\n{chunk}\n"

    system_prompt = """You are a financial document assistant. Answer the user's question using ONLY the information in the sources provided below.

Rules:
1. Every fact in your answer MUST be followed by a citation like [Source 1] or [Source 2].
2. If the sources don't contain the answer, say "I cannot find this information in the provided documents" — do NOT guess or use outside knowledge.
3. Keep your answer concise and directly focused on the question.
"""
    user_prompt = f"Sources:\n{context_text}\n\nQuestion: {question}\n\nAnswer with citations:"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content, debug_info