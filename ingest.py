# ingest.py
# This is the first script you run when setting up MedRAG.
# Think of it as "teaching" the system about medicine —
# we feed it documents, break them into pieces, convert them
# into numbers (embeddings), and store them so we can search later.
# You only need to run this once, or again when you add new documents.

import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, DATA_PATH
)


def load_documents(data_path: str) -> list[dict]:
    # Go through the data folder and read every .txt file we find.
    # We store both the filename and the text so we can track
    # where each piece of information originally came from.
    docs = []
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append({"filename": filename, "text": text})
            print(f"  Loaded: {filename} ({len(text)} chars)")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # We can't feed an entire document to the embedding model at once —
    # it has a size limit, and smaller chunks give more precise search results.
    # So we split the text into chunks of chunk_size characters.
    #
    # The overlap is important in medical text — a sentence describing
    # a symptom might start at the end of one chunk and finish at the
    # beginning of the next. Overlap makes sure we don't lose that context.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # slide forward but keep some overlap from the previous chunk
        start += chunk_size - overlap
    return chunks


def ingest():
    print("=== MedRAG Ingestion Pipeline ===\n")

    # Step 1: Load all the medical documents from our data folder
    print("Loading documents...")
    docs = load_documents(DATA_PATH)
    if not docs:
        # Nothing to process — remind the user to add documents first
        print("No .txt files found in data/sample_docs. Add documents and re-run.")
        return

    # Step 2: Break every document into smaller overlapping chunks
    print("\nChunking documents...")
    all_chunks = []
    all_metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            # We save the source filename and chunk number as metadata
            # so when we retrieve a chunk later, we know exactly where it came from
            all_metadata.append({"source": doc["filename"], "chunk_index": i})
        print(f"  {doc['filename']} → {len(chunks)} chunks")

    # Step 3: Convert every chunk into a vector (embedding)
    # The embedding model turns text into numbers that capture meaning —
    # so "heart attack" and "myocardial infarction" end up close together
    # in vector space even though the words are different
    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # Step 4: Store everything in ChromaDB
    # ChromaDB is our vector database — it saves the chunks, their embeddings,
    # and metadata so we can do fast similarity search later
    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # If we've run ingest before, clear the old data and start fresh
    # so we don't end up with duplicate chunks
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print("  Cleared existing collection — starting fresh.")

    collection = client.create_collection(CHROMA_COLLECTION_NAME)

    # Each chunk needs a unique ID — we use UUID so there's no collision
    ids = [str(uuid.uuid4()) for _ in all_chunks]
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
        ids=ids
    )

    print(f"\n✅ Done! {len(all_chunks)} chunks are now stored and ready to search.")


if __name__ == "__main__":
    ingest()