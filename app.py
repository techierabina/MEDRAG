# app.py
# This is the front door of MedRAG — the interface users interact with.
# We use Gradio's Chatbot to give it a real conversational feel, but instead
# of the default Gradio look, we write custom HTML and CSS to give it a
# clean, professional medical feel. The pipeline does all the heavy lifting —
# this file just handles presenting the question, calling the pipeline, and
# displaying the answer with its sources inline in the chat bubble.
#
# Note: each turn is still answered independently by the RAG pipeline
# (we don't feed prior chat history into retrieval) — the "chat" here is a
# conversational display, not multi-turn memory. That keeps every answer
# grounded strictly in retrieved documents, which is what we want for
# clinical QA — no risk of the LLM drifting based on earlier chat turns.

import gradio as gr
from pipeline import MedRAGPipeline

# We initialize the pipeline once when the app starts —
# not on every question, because loading the models takes time.
pipeline = MedRAGPipeline()


def respond(message: str, history: list):
    # Basic validation — no point calling the pipeline with an empty string
    if not message.strip():
        return "Please enter a clinical question."

    result = pipeline.run(message)
    answer = result["answer"]

    # Fold the sources into the same chat bubble as a small footer,
    # so the conversation stays in one clean thread instead of
    # splitting answer/sources into separate boxes.
    if result["sources"]:
        sources_list = "\n".join(f"- {s}" for s in result["sources"])
        answer += f"\n\n---\n**Sources** ({result['chunks_used']} chunks retrieved):\n{sources_list}"
    else:
        answer += "\n\n---\n*No matching documents found in the knowledge base.*"

    return answer


# Custom CSS — Poppins font, deep navy and clean white medical palette,
# MEDRAG title big and bold, chat bubbles styled to match the brand
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background-color: #f0f4f8 !important;
    margin: 0;
    padding: 0;
}

/* Header section */
.medrag-header {
    background: linear-gradient(135deg, #0a2540 0%, #1a4a7a 100%);
    padding: 48px 40px 36px 40px;
    text-align: center;
    border-bottom: 3px solid #2563eb;
}

.medrag-title {
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff;
    margin: 0 0 8px 0;
    line-height: 1;
}

.medrag-title span {
    color: #60a5fa;
}

.medrag-subtitle {
    font-size: 15px;
    font-weight: 400;
    color: #93c5fd;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 16px 0;
}

.medrag-disclaimer {
    display: inline-block;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    font-size: 11px;
    font-weight: 500;
    padding: 6px 16px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}

/* Main content area */
.medrag-body {
    max-width: 900px;
    margin: 24px auto;
    padding: 0 24px;
}

/* Chat window */
.gradio-container .bubble-wrap, .gradio-container [class*="chatbot"] {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
}

/* User bubble */
.message.user {
    background: #0a2540 !important;
    color: #ffffff !important;
}

/* Bot bubble */
.message.bot {
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
}

/* Textboxes */
textarea, input[type="text"] {
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    color: #1e293b !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background: #f8fafc !important;
    padding: 14px !important;
    resize: vertical !important;
    transition: border-color 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #2563eb !important;
    outline: none !important;
    background: #ffffff !important;
}

/* Buttons */
button.primary {
    background: #0a2540 !important;
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
}

button.primary:hover {
    background: #1a4a7a !important;
}

/* Footer */
.medrag-footer {
    text-align: center;
    padding: 32px;
    font-size: 12px;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    margin-top: 24px;
}
"""

HEADER_HTML = """
    <div class="medrag-header">
        <h1 class="medrag-title">MED<span>RAG</span></h1>
        <p class="medrag-subtitle">Clinical Retrieval-Augmented Generation</p>
        <span class="medrag-disclaimer">
            For research purposes only — not a substitute for professional medical advice
        </span>
    </div>
"""

FOOTER_HTML = """
    <div class="medrag-footer">
        MedRAG — Built by Rabina Karki &nbsp;|&nbsp;
        MS Data Science, University of Central Oklahoma &nbsp;|&nbsp;
        ChromaDB · sentence-transformers · Groq LLaMA 3.3
    </div>
"""

EXAMPLES = [
    "What are the symptoms of type 2 diabetes?",
    "How is hypertension diagnosed and classified?",
    "What medications are used to treat asthma?",
    "What is the difference between bacterial and viral infections?",
]

with gr.Blocks(title="MedRAG — Clinical QA") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Column(elem_classes="medrag-body"):
        gr.ChatInterface(
            fn=respond,
            examples=EXAMPLES,
            chatbot=gr.Chatbot(
                height=520,
                placeholder="Ask a clinical question to get started — answers are grounded in the retrieved documents, with sources listed below each response.",
                show_label=False,
            ),
            textbox=gr.Textbox(
                placeholder="e.g. What are the first-line medications for hypertension?",
                show_label=False,
                container=False,
            ),
        )

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(share=False, css=custom_css, server_name="127.0.0.1")