import os
import gradio as gr
from app import config, pdf_engine, llm_engine

# Custom CSS for glassmorphism, dark mode, and sleek gradients
CUSTOM_CSS = """
body, .gradio-container {
    background: #0b0f19 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.glass-container {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4) !important;
}
.header-container {
    text-align: center;
    margin-bottom: 20px;
    padding: 15px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.header-title {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 5px;
}
.custom-btn {
    transition: all 0.2s ease !important;
}
.custom-btn:hover {
    transform: translateY(-1px) !important;
}
.primary-btn {
    background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
}
.primary-btn:hover {
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.5) !important;
}
.status-msg {
    color: #38bdf8;
    font-size: 0.95rem;
    font-weight: 500;
}
"""

def create_ui():
    # Setup custom theme
    theme = gr.themes.Default(
        primary_hue="sky",
        secondary_hue="indigo",
        neutral_hue="slate"
    ).set(
        body_background_fill="#0b0f19",
        block_background_fill="#111827",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.05)"
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="PDF Retrieval Chatbot") as demo:
        # State variables
        pdf_path = gr.State("")
        current_page = gr.State(0)
        total_pages = gr.State(0)

        # Header Section
        gr.HTML(
            "<div class='header-container'>"
            "<h1 class='header-title'>Interactive PDF Chatbot</h1>"
            "<p class='header-subtitle'>Upload a PDF and ask questions using OpenAI or a local MiniCPM5-1B model running on your CPU.</p>"
            "</div>"
        )

        # Main Layout
        with gr.Row():
            # Left Column: PDF preview and controls
            with gr.Column(scale=1, min_width=350, elem_classes="glass-container"):
                gr.Markdown("### 📄 PDF Document Viewer")
                pdf_uploader = gr.File(label="Upload PDF File", file_types=[".pdf"])
                
                # Image viewer for the PDF pages
                pdf_image = gr.Image(
                    label="Page Preview", 
                    interactive=False,
                    type="pil"
                )
                
                # Page Navigation Row
                with gr.Row():
                    prev_page_btn = gr.Button("◀ Prev Page", elem_classes="custom-btn")
                    page_label = gr.Markdown("Page 0 of 0", elem_classes="status-msg")
                    next_page_btn = gr.Button("Next Page ▶", elem_classes="custom-btn")
                
                with gr.Row():
                    page_jump_input = gr.Number(
                        label="Jump to Page", 
                        minimum=1, 
                        precision=0, 
                        step=1,
                        value=1
                    )
                    jump_btn = gr.Button("Go", elem_classes="custom-btn")

            # Right Column: Model Settings & Chat
            with gr.Column(scale=1, min_width=450, elem_classes="glass-container"):
                gr.Markdown("### ⚙️ Engine Settings")
                
                with gr.Row():
                    model_type = gr.Radio(
                        choices=["OpenAI", "Local Model (MiniCPM5-1B)"],
                        value="OpenAI",
                        label="Inference Backend"
                    )

                # Configurations depending on model selection
                with gr.Row() as openai_settings:
                    openai_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        value=os.environ.get("OPENAI_API_KEY", "")
                    )

                with gr.Row(visible=False) as local_settings:
                    local_status = gr.Textbox(
                        label="Local Model Status",
                        value="MiniCPM5-1B-SFT model weights will download/load on demand.",
                        interactive=False
                    )

                # Advanced Settings Accordion
                with gr.Accordion("Advanced Parameters", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.0, 
                        maximum=1.2, 
                        value=0.3, 
                        step=0.1, 
                        label="Temperature"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=128, 
                        maximum=4096, 
                        value=1024, 
                        step=128, 
                        label="Max Output Tokens"
                    )
                    enable_thinking_chk = gr.Checkbox(
                        label="Enable Think/Reasoning Mode (Local MiniCPM5 only)",
                        value=True,
                        visible=False
                    )

                gr.Markdown("### 💬 Chat Session")
                chatbot = gr.Chatbot(label="Chat History", type="messages", height=450)
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Type your question about the PDF and press Enter...",
                        show_label=False,
                        scale=8
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="primary-btn")
                
                clear_chat_btn = gr.Button("Clear Chat History", size="sm")

        # Define Reactivity / Toggle settings panels
        def toggle_settings(model_choice):
            if model_choice == "OpenAI":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

        model_type.change(
            fn=toggle_settings,
            inputs=[model_type],
            outputs=[openai_settings, local_settings, enable_thinking_chk]
        )

        # PDF Upload Handler
        def handle_pdf_load(file):
            if file is None:
                return "", 0, 0, gr.update(value=None), "Page 0 of 0", gr.update(value=1)
            
            path = file.name
            img, total = pdf_engine.render_pdf_page(path, 0)
            lbl = f"Page 1 of {total}"
            return path, 0, total, img, lbl, gr.update(maximum=total, value=1)

        pdf_uploader.upload(
            fn=handle_pdf_load,
            inputs=[pdf_uploader],
            outputs=[pdf_path, current_page, total_pages, pdf_image, page_label, page_jump_input]
        )

        # Navigation Handlers
        def navigate_page(direction, path, curr, total):
            if not path:
                return curr, "Page 0 of 0", None, 1
            
            new_page = curr
            if direction == "prev":
                new_page = max(0, curr - 1)
            elif direction == "next":
                new_page = min(total - 1, curr + 1)
                
            img, _ = pdf_engine.render_pdf_page(path, new_page)
            lbl = f"Page {new_page + 1} of {total}"
            return new_page, lbl, img, new_page + 1

        prev_page_btn.click(
            fn=navigate_page,
            inputs=[gr.State("prev"), pdf_path, current_page, total_pages],
            outputs=[current_page, page_label, pdf_image, page_jump_input]
        )

        next_page_btn.click(
            fn=navigate_page,
            inputs=[gr.State("next"), pdf_path, current_page, total_pages],
            outputs=[current_page, page_label, pdf_image, page_jump_input]
        )

        def jump_to_page(page_target, path, total):
            if not path or total == 0:
                return 0, "Page 0 of 0", None
            
            target_idx = int(page_target) - 1
            target_idx = max(0, min(total - 1, target_idx))
            
            img, _ = pdf_engine.render_pdf_page(path, target_idx)
            lbl = f"Page {target_idx + 1} of {total}"
            return target_idx, lbl, img

        jump_btn.click(
            fn=jump_to_page,
            inputs=[page_jump_input, pdf_path, total_pages],
            outputs=[current_page, page_label, pdf_image]
        )

        # Response Generation Handler (Streaming)
        def generate_response(
            history,
            question,
            model_sel,
            openai_key,
            path,
            temp,
            max_toks,
            think_mode,
            tot_pages
        ):
            if not path:
                history.append({"role": "assistant", "content": "❌ Please upload a PDF document first."})
                yield history, "", 0, "Page 0 of 0", None, 1
                return

            if not question.strip():
                yield history, "", 0, "Page 0 of 0", None, 1
                return

            # Append the user's message
            history.append({"role": "user", "content": question})
            # Add an empty placeholder message for the assistant
            history.append({"role": "assistant", "content": ""})
            yield history, "", gr.State(0), "Processing...", None, gr.State(1)

            # Check configuration requirements
            if model_sel == "OpenAI" and not openai_key.strip():
                history[-1]["content"] = "❌ Please provide an OpenAI API Key in the settings panel."
                yield history, "", 0, "Page 0 of 0", None, 1
                return

            # Indexing step
            history[-1]["content"] = "⏳ Indexing PDF document (creating vector embeddings)..."
            yield history, "", 0, "Processing...", None, 1

            try:
                # Store the key in environment variable for the run
                if model_sel == "OpenAI":
                    os.environ["OPENAI_API_KEY"] = openai_key
                db = pdf_engine.index_pdf(path, model_sel, openai_key)
            except Exception as e:
                history[-1]["content"] = f"❌ Error processing PDF: {str(e)}"
                yield history, "", 0, "Error", None, 1
                return

            history[-1]["content"] = "🔍 Retrieving relevant context from document..."
            yield history, "", 0, "Retrieving...", None, 1

            # Retrieve context
            try:
                retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
                docs = retriever.invoke(question)
            except Exception as e:
                history[-1]["content"] = f"❌ Error retrieving context: {str(e)}"
                yield history, "", 0, "Error", None, 1
                return

            if not docs:
                history[-1]["content"] = "❌ No relevant context could be found in the PDF."
                yield history, "", 0, "Finished", None, 1
                return

            # Auto-jump to the page of the most relevant chunk
            best_match_page = docs[0].metadata.get("page", 0)
            img, _ = pdf_engine.render_pdf_page(path, best_match_page)
            lbl = f"Page {best_match_page + 1} of {tot_pages}"

            history[-1]["content"] = "✍️ Generating response..."
            yield history, "", best_match_page, lbl, img, best_match_page + 1

            # Run RAG Prompt & LLM
            prompt = llm_engine.format_rag_prompt(question, docs)
            
            try:
                if model_sel == "OpenAI":
                    stream = llm_engine.generate_openai_stream(
                        prompt, 
                        openai_key, 
                        temp, 
                        max_toks
                    )
                else:
                    def progress_update(msg):
                        history[-1]["content"] = f"⏳ {msg}"
                        # Non-blocking yield of status
                        # Gradio generator handles yields fine
                    
                    stream = llm_engine.generate_local_stream(
                        prompt,
                        think_mode,
                        temp,
                        max_toks,
                        progress_callback=progress_update
                    )
                
                # Clear placeholder
                history[-1]["content"] = ""
                for chunk in stream:
                    history[-1]["content"] += chunk
                    yield history, "", best_match_page, lbl, img, best_match_page + 1
                    
            except Exception as e:
                history[-1]["content"] = f"❌ Error generating response: {str(e)}"
                yield history, "", best_match_page, lbl, img, best_match_page + 1

        # Submit triggers
        chat_input.submit(
            fn=generate_response,
            inputs=[
                chatbot, 
                chat_input, 
                model_type, 
                openai_key_input, 
                pdf_path, 
                temperature_slider, 
                max_tokens_slider, 
                enable_thinking_chk,
                total_pages
            ],
            outputs=[
                chatbot, 
                chat_input, 
                current_page, 
                page_label, 
                pdf_image, 
                page_jump_input
            ]
        )

        send_btn.click(
            fn=generate_response,
            inputs=[
                chatbot, 
                chat_input, 
                model_type, 
                openai_key_input, 
                pdf_path, 
                temperature_slider, 
                max_tokens_slider, 
                enable_thinking_chk,
                total_pages
            ],
            outputs=[
                chatbot, 
                chat_input, 
                current_page, 
                page_label, 
                pdf_image, 
                page_jump_input
            ]
        )

        # Clear Chat trigger
        def clear_chat():
            return []
            
        clear_chat_btn.click(fn=clear_chat, outputs=[chatbot])

    return demo
