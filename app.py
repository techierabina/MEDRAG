# app.py
# This is the front door of MedRAG — the interface users interact with.
# We use Gradio to build the UI, but instead of the default Gradio look,
# we write custom HTML and CSS to give it a clean, professional medical feel.
# The pipeline does all the heavy lifting — this file just handles
# presenting the question, calling the pipeline, and displaying the answer.

import gradio as gr
from pipeline import MedRAGPipeline

# We initialize the pipeline once when the app starts —
# not on every question, because loading the models takes time.
pipeline = MedRAGPipeline()


def answer_question(question: str):
    # Basic validation — no point calling the pipeline with an empty string
    if not question.strip():
        return "Please enter a clinical question.", ""

    result = pipeline.run(question)

    answer = result["answer"]

    # Format the sources into a clean readable list
    if result["sources"]:
        sources_list = "\n".join(f"- {s}" for s in result["sources"])
        sources_display = f"Retrieved from {result['chunks_used']} chunks across:\n{sources_list}"
    else:
        sources_display = "No sources retrieved."

    return answer, sources_display


# Custom CSS — Poppins font, deep navy and clean white medical palette,
# MEDRAG title big and bold, everything else minimal and professional
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
    margin: 40px auto;
    padding: 0 24px;
}

/* Input and output cards */
.medrag-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.medrag-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 10px;
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

/* Submit button */
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
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
    width: 100% !important;
}

button.primary:hover {
    background: #1a4a7a !important;
}

/* Example questions */
.examples-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
}

/* Footer */
.medrag-footer {
    text-align: center;
    padding: 32px;
    font-size: 12px;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
}
"""

# Build the Gradio interface using Blocks for full layout control
with gr.Blocks(css=custom_css, title="MedRAG — Clinical QA") as demo:

    # Header
    gr.HTML("""
        <div class="medrag-header">
            <h1 class="medrag-title">MED<span>RAG</span></h1>
            <p class="medrag-subtitle">Clinical Retrieval-Augmented Generation</p>
            <span class="medrag-disclaimer">
                For research purposes only — not a substitute for professional medical advice
            </span>
        </div>
    """)

    with gr.Column(elem_classes="medrag-body"):

        # Question input card
        gr.HTML('<div class="medrag-label">Clinical Question</div>')
        question_input = gr.Textbox(
            placeholder="e.g. What are the first-line medications for hypertension?",
            lines=3,
            show_label=False,
            container=False
        )
        submit_btn = gr.Button("Submit Query", variant="primary")

        # Answer output
        gr.HTML('<div class="medrag-label" style="margin-top:24px;">Answer</div>')
        answer_output = gr.Textbox(
            lines=10,
            show_label=False,
            interactive=False,
            container=False,
            placeholder="Answer will appear here after submission..."
        )

        # Sources output
        gr.HTML('<div class="medrag-label" style="margin-top:16px;">Sources</div>')
        sources_output = gr.Textbox(
            lines=3,
            show_label=False,
            interactive=False,
            container=False,
            placeholder="Retrieved source documents will be listed here..."
        )

        # Example questions
        gr.HTML('<div class="examples-header" style="margin-top:28px;">Example Queries</div>')
        gr.Examples(
            examples=[
                ["What are the symptoms of type 2 diabetes?"],
                ["How is hypertension diagnosed and classified?"],
                ["What medications are used to treat asthma?"],
                ["What is the difference between bacterial and viral infections?"],
            ],
            inputs=question_input
        )

    # Footer
    gr.HTML("""
        <div class="medrag-footer">
            MedRAG — Built by Rabina Karki &nbsp;|&nbsp;
            MS Data Science, University of Central Oklahoma &nbsp;|&nbsp;
            ChromaDB · sentence-transformers · Groq LLaMA 3.3
        </div>
    """)

    # Wire up the button and Enter key to the pipeline
    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    question_input.submit(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch(share=False)