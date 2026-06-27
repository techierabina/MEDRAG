# pipeline.py
# This is the heart of MedRAG — it connects everything together.
# Think of it as the manager that coordinates the retriever and generator.
# When the app gets a question, it comes here first.
# pipeline.py says: "okay, go find relevant chunks" (retriever),
# then "now turn those chunks into a real answer" (generator).
# Neither the app nor the UI needs to know how retrieval or generation works —
# they just call pipeline.run() and get an answer back. Clean and simple.

from retriever import Retriever
from generator import Generator
from config import TOP_K


class MedRAGPipeline:
    def __init__(self):
        # We load both the retriever and generator once at startup.
        # Loading them is slow (models need to initialize) so we don't
        # want to reload them on every single question — just once is enough.
        print("Loading retriever...")
        self.retriever = Retriever()
        print("Loading generator...")
        self.generator = Generator()
        print("MedRAG pipeline is ready!\n")

    def run(self, question: str, top_k: int = TOP_K) -> dict:
        # Step 1: Search ChromaDB for the most relevant chunks
        # These chunks are the evidence we'll give to the LLM
        context_chunks = self.retriever.retrieve(question, top_k=top_k)

        if not context_chunks:
            # If we found nothing relevant, we stop here.
            # There's no point sending an empty context to the LLM —
            # it would just make something up, which is exactly what we don't want.
            return {
                "answer": "No relevant documents found in the knowledge base. Try adding more medical documents and re-running ingest.py.",
                "sources": [],
                "chunks_used": 0
            }

        # Step 2: Send the retrieved chunks + question to Groq LLaMA
        # The generator will build the prompt and return a grounded answer
        answer = self.generator.generate(question, context_chunks)

        # Step 3: Collect the unique source filenames so we can show
        # the user exactly which documents the answer came from
        sources = list({chunk["source"] for chunk in context_chunks})

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(context_chunks)
        }


if __name__ == "__main__":
    # Quick end-to-end test — runs the full pipeline from question to answer
    pipeline = MedRAGPipeline()
    question = "What are the symptoms of type 2 diabetes?"
    result = pipeline.run(question)

    print(f"Question: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Chunks used: {result['chunks_used']}")