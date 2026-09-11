from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from pypdf import PdfReader

# Step 1: Re-extract and re-chunk the PDF (BM25 needs the raw chunk list directly, unlike Chroma which stores it for us)
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

# Step 2: Build the BM25 index
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Step 3: Set up vector search (same as before)
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tesla_10k")

# Step 4: Take the question
question = input("Ask a question about Tesla's 10-K: ")

# Step 5: Run BM25 search
tokenized_question = question.split()
bm25_scores = bm25.get_scores(tokenized_question)
top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
bm25_results = [chunks[i] for i in top_bm25_indices]

# Step 6: Run vector search
question_embedding = model.encode(question).tolist()
vector_results_raw = collection.query(query_embeddings=[question_embedding], n_results=5)
vector_results = vector_results_raw["documents"][0]

# Step 7: Combine both result sets (removing duplicates)
combined_results = list(dict.fromkeys(bm25_results + vector_results))

print(f"\n--- BM25 found {len(bm25_results)} chunks, Vector found {len(vector_results)} chunks ---")
print(f"--- Combined (deduplicated): {len(combined_results)} chunks ---\n")

for i, chunk in enumerate(combined_results):
    print(f"[Chunk {i+1}]")
    print(chunk[:200] + "...")  # just show first 200 chars for readability
    print("-" * 40)