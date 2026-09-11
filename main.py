from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


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
print(f"Total chunks: {len(chunks)}")


model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tesla_10k")


for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print(f"Stored {len(chunks)} chunks in the vector database.")