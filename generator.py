# generator.py
# This is where we actually talk to the LLM (Groq LLaMA 3.3).
# By the time we get here, the retriever has already found the most
# relevant chunks from our documents. Now we hand those chunks to the LLM
# and say "using ONLY this information, answer the user's question."
# The LLM's job here is not to think freely — it's to read and summarize
# the evidence we retrieved. That's what keeps answers accurate.

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


class Generator:
    def __init__(self):
        # Connect to Groq using our API key from the .env file
        self.client = Groq(api_key=GROQ_API_KEY)

    def build_prompt(self, question: str, context_chunks: list[dict]) -> str:
        # We format all the retrieved chunks into one block of context.
        # We also include the source filename so the LLM knows
        # which document each piece of information came from.
        context_text = "\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in context_chunks
        )

        # This is the most important part of the whole pipeline —
        # we explicitly tell the LLM to ONLY use the context we provide.
        # If we don't say this clearly, the LLM might mix in its own
        # training knowledge and start hallucinating medical information.
        prompt = f"""You are a clinical medical assistant. Answer the question below using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly — do not guess.

---
CONTEXT:
{context_text}

---
QUESTION:
{question}

---
ANSWER:"""
        return prompt

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        # Build the grounded prompt first
        prompt = self.build_prompt(question, context_chunks)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    # The system message sets the LLM's personality and behavior.
                    # We want it to be careful and cite its reasoning —
                    # not confident and creative like a regular chatbot.
                    "content": "You are a knowledgeable, careful clinical assistant. Always cite your reasoning from the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # Low temperature means less creativity and more factual answers.
            # For medical information, we never want the model to "get creative."
            temperature=0.2,
            max_tokens=1024
        )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick test with a dummy chunk to make sure Groq connection is working
    g = Generator()
    dummy_chunks = [{"source": "test.txt", "text": "Diabetes is characterized by high blood sugar levels."}]
    answer = g.generate("What is diabetes?", dummy_chunks)
    print(answer)