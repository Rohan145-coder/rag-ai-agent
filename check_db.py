import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tesla_10k")

print(f"Number of chunks stored in the collection: {collection.count()}")