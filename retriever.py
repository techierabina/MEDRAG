# retriever.py
# This is the "search engine" of MedRAG.
# When a user asks a question, we don't send it straight to the LLM —
# we first search our ChromaDB for the most relevant chunks of medical text.
# Those chunks become the evidence the LLM uses to answer the question.
# This is what makes RAG different from a regular chatbot — the answers
# are grounded in real documents, not just the model's memory.

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, TOP_K


class Retriever:
    def __init__(self):
        # We load the same embedding model we used during ingestion.
        # This is critical — if we used a different model here,
        # the query vector and the stored chunk vectors would be in
        # completely different spaces and the search would be meaningless.
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(CHROMA_COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        # Step 1: Convert the user's question into a vector
        # just like we did with the document chunks during ingestion
        query_embedding = self.model.encode([query]).tolist()

        # Step 2: Ask ChromaDB to find the top_k chunks whose vectors
        # are closest to the query vector — these are the most relevant pieces
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Step 3: Package the results into a clean list of dicts
        # We include the distance so we can see how confident the match is —
        # lower distance means more similar to the query
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": round(dist, 4)
            })

        return retrieved


if __name__ == "__main__":
    # Quick sanity check — run this directly to make sure retrieval is working
    # before wiring it into the full pipeline
    r = Retriever()
    results = r.retrieve("What are the symptoms of diabetes?")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (distance={res['distance']}) from {res['source']} ---")
        print(res["text"][:300])