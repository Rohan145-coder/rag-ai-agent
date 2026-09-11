import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from pypdf import PdfReader

# Step 1: Load the API key from .env
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Same setup as before ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] BASE_DIR resolved to: {BASE_DIR}")
print(f"[DEBUG] Full chroma_db path: {os.path.join(BASE_DIR, 'chroma_db')}")
print(f"[DEBUG] Does that path exist? {os.path.exists(os.path.join(BASE_DIR, 'chroma_db'))}")

reader = PdfReader("data/tesla.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = chunk_text(full_text)
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = client.get_or_create_collection(name="tesla_10k")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

question = input("Ask a question about Tesla's 10-K: ")

# --- Hybrid search ---
tokenized_question = question.split()
bm25_scores = bm25.get_scores(tokenized_question)
top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
bm25_results = [chunks[i] for i in top_bm25_indices]

question_embedding = model.encode(question).tolist()
vector_results_raw = collection.query(query_embeddings=[question_embedding], n_results=5)
vector_results = vector_results_raw["documents"][0]
print(f"\n[DEBUG] BM25 returned: {len(bm25_results)} chunks")
print(f"[DEBUG] Vector returned: {len(vector_results)} chunks")
if len(vector_results) == 0:
    print(f"[DEBUG] Raw vector response: {vector_results_raw}")

combined_results = list(dict.fromkeys(bm25_results + vector_results))

# --- Rerank ---
pairs = [[question, chunk] for chunk in combined_results]
rerank_scores = reranker.predict(pairs)
scored_chunks = list(zip(combined_results, rerank_scores))
scored_chunks.sort(key=lambda x: x[1], reverse=True)
top_chunks = [chunk for chunk, score in scored_chunks[:3]]
print("\n--- DEBUG: Chunks sent to LLM ---")
for i, chunk in enumerate(top_chunks):
    print(f"[Source {i+1}] {chunk[:150]}...")
print("--- END DEBUG ---\n")

# --- NEW: Build a prompt that forces citations ---
context_text = ""
for i, chunk in enumerate(top_chunks):
    context_text += f"\n[Source {i+1}]:\n{chunk}\n"

system_prompt = """You are a financial document assistant. Answer the user's question using ONLY the information in the sources provided below.

Rules:
1. Every fact in your answer MUST be followed by a citation like [Source 1] or [Source 2].
2. If the sources don't contain the answer, say "I cannot find this information in the provided documents" — do NOT guess or use outside knowledge.
3. Keep your answer concise and directly focused on the question.
"""

user_prompt = f"""Sources:
{context_text}

Question: {question}

Answer with citations:"""

# --- NEW: Call Groq ---
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print("\n--- FINAL ANSWER ---\n")
print(response.choices[0].message.content)