from sentence_transformers import SentenceTransformer
import chromadb
model=SentenceTransformer("all-MiniLM-L6-v2")
client= chromadb.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(name="tesla_10k")
question=input("Ask a question about Tesla's 10-K: ")
question_embedding= model.encode(question).tolist()
result=collection.query(query_embeddings=[question_embedding] ,n_results=5)
for i ,chunk in enumerate(result["documents"][0]):
    print(f"[chunk {i+ 1}]")
    print(chunk)