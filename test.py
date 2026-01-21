from chromadb import Client
from chromadb.config import Settings

CHROMA_DIR = r"D:/Interns/pdf_extraction_using_chromadb/chroma_store"

client = Client(Settings(
    chroma_db_impl="duckdb+parquet",  # same persistent DB
    persist_directory=CHROMA_DIR,
    anonymized_telemetry=False
))

collection = client.get_or_create_collection(
    name="pdf_markdown_embeddings",
    metadata={"embedding_model": "all-MiniLM-L6-v2"}
)

print("Total chunks in collection:", collection.count())

# Optional: view first 5 chunks
results = collection.query(query_texts=["example query"], n_results=5)
for i, doc in enumerate(results['documents'][0]):
    print(f"\nChunk {i+1}:\n{doc}")
    print("Metadata:", results['metadatas'][0][i])
