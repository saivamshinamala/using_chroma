import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uuid
from pathlib import Path
from typing import List

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MARKDOWN_DIR = r"D:/Interns/pdf_extraction_using_chromadb/data/markdowns"
CHROMA_DIR = r"D:/Interns/pdf_extraction_using_chromadb/chroma_store"
COLLECTION_NAME = "pdf_markdown_embeddings"
EMBEDDING_MODEL_PATH = r"all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 256

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# =========================
# MODEL
# =========================
print("🔹 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
print("✅ Model loaded successfully")

# =========================
# ✅ PERSISTENT CHROMA CLIENT (FIX)
# =========================
client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"embedding_model": "all-MiniLM-L6-v2"}
)

# =========================
# UTILITIES
# =========================
def load_markdown_files(md_dir: str) -> List[Path]:
    files = list(Path(md_dir).rglob("*.md"))
    print(f"📄 Found {len(files)} markdown files")
    return files

def chunk_markdown_stream(md_path: Path) -> List[str]:
    chunks = []
    current = ""

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            current += line
            if len(current) >= CHUNK_SIZE:
                chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:]

    if current:
        chunks.append(current.strip())

    return chunks

# =========================
# INGEST
# =========================
def ingest_markdown():
    md_files = load_markdown_files(MARKDOWN_DIR)
    if not md_files:
        print("❌ No markdown files found")
        return

    for md_path in tqdm(md_files, desc="📄 Markdown files"):
        print(f"\n✂️ Chunking {md_path.name}...")
        chunks = chunk_markdown_stream(md_path)
        print(f"🧩 Created {len(chunks)} chunks")

        ids, embeddings, metadatas = [], [], []

        for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE),
                      desc="🔹 Embedding",
                      leave=False):
            batch = chunks[i:i+EMBED_BATCH_SIZE]

            vecs = embedder.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            embeddings.extend(vecs.tolist())
            ids.extend(str(uuid.uuid4()) for _ in batch)
            metadatas.extend({
                "source": md_path.name,
                "chunk_index": i + j
            } for j in range(len(batch)))

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✅ Stored {len(ids)} chunks")

    print("🎉 DONE — Chroma persisted to disk")

if __name__ == "__main__":
    ingest_markdown()
