from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from pypdf import PdfReader

# --- Same setup as before ---
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
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tesla_10k")

# --- NEW: Load the reranker model ---
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker loaded.")

question = input("Ask a question about Tesla's 10-K: ")

# --- Hybrid search (same as before) ---
tokenized_question = question.split()
bm25_scores = bm25.get_scores(tokenized_question)
top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
bm25_results = [chunks[i] for i in top_bm25_indices]

question_embedding = model.encode(question).tolist()
vector_results_raw = collection.query(query_embeddings=[question_embedding], n_results=5)
vector_results = vector_results_raw["documents"][0]
combined_results = list(dict.fromkeys(bm25_results + vector_results))
print(f"\nBM25 alone: {len(bm25_results)} chunks, Vector alone: {len(vector_results)} chunks")
print(f"Hybrid search returned {len(combined_results)} candidate chunks.")
# --- NEW: Rerank the combined results ---
pairs = [[question, chunk] for chunk in combined_results]
rerank_scores = reranker.predict(pairs)

# Sort chunks by rerank score, highest first
scored_chunks = list(zip(combined_results, rerank_scores))
scored_chunks.sort(key=lambda x: x[1], reverse=True)

top_chunks = [chunk for chunk, score in scored_chunks[:3]]

print("\n--- Top 3 chunks AFTER reranking ---\n")
for i, (chunk, score) in enumerate(scored_chunks[:3]):
    print(f"[Rank {i+1}] score: {score:.4f}")
    print(chunk[:300] + "...")
    print("-" * 40)