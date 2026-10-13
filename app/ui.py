from __future__ import annotations

import logging

import gradio as gr

from app import chat, config, pdf_engine
from app.exceptions import PDFChatbotError

logger = logging.getLogger(__name__)

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
/* Ensure headers, markdown text, and accordion labels are bright and readable */
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container h5, .gradio-container h6,
.gradio-container .prose h3, .gradio-container .prose p,
.gradio-container details summary, .gradio-container details summary span,
.gradio-container button.label-wrap, .gradio-container button.label-wrap span {
    color: #f1f5f9 !important;
}
"""


def _uploaded_path(file: object) -> str:
    """Gradio passes a str subclass with ``.name`` (or a plain str) for uploaded files."""
    return getattr(file, "name", None) or str(file)


def _render(path: str, page: int):
    """Render a page for the viewer; failures become a toast, not a crash."""
    try:
        return pdf_engine.render_pdf_page(path, page)
    except PDFChatbotError as exc:
        gr.Warning(str(exc))
        return None, 0


def create_ui(settings: config.Settings | None = None) -> gr.Blocks:
    settings = settings or config.get_settings()
    server_key_available = settings.use_server_key and bool(settings.server_openai_api_key)

    # Setup custom theme
    theme = gr.themes.Default(
        primary_hue="sky",
        secondary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b0f19",
        block_background_fill="#111827",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.05)",
    )

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="PDF Retrieval Chatbot",
        analytics_enabled=False,
    ) as demo:
        # State variables
        pdf_path = gr.State("")
        current_page = gr.State(0)
        total_pages = gr.State(0)

        # Header Section
        gr.HTML(
            "<div class='header-container'>"
            "<h1 class='header-title'>Interactive PDF Chatbot</h1>"
            "<p class='header-subtitle'>Upload a PDF and ask questions using OpenAI or a local "
            "MiniCPM5-1B model running on your CPU.</p>"
            "</div>"
        )

        # Main Layout
        with gr.Row():
            # Left Column: PDF preview and controls
            with gr.Column(scale=1, min_width=350, elem_classes="glass-container"):
                gr.Markdown("### 📄 PDF Document Viewer")
                pdf_uploader = gr.File(label="Upload PDF File", file_types=[".pdf"])

                # Image viewer for the PDF pages
                pdf_image = gr.Image(label="Page Preview", interactive=False, type="pil")

                # Page Navigation Row
                with gr.Row():
                    prev_page_btn = gr.Button("◀ Prev Page", elem_classes="custom-btn")
                    page_label = gr.Markdown("Page 0 of 0", elem_classes="status-msg")
                    next_page_btn = gr.Button("Next Page ▶", elem_classes="custom-btn")

                with gr.Row():
                    page_jump_input = gr.Number(
                        label="Jump to Page", minimum=1, precision=0, step=1, value=1
                    )
                    jump_btn = gr.Button("Go", elem_classes="custom-btn")

            # Right Column: Model Settings & Chat
            with gr.Column(scale=1, min_width=450, elem_classes="glass-container"):
                gr.Markdown("### ⚙️ Engine Settings")

                with gr.Row():
                    model_type = gr.Radio(
                        choices=list(config.BACKENDS),
                        value=config.BACKEND_OPENAI,
                        label="Inference Backend",
                    )

                # Configurations depending on model selection
                with gr.Row() as openai_settings:
                    # Never pre-filled: a server-side key must not be sent to the browser.
                    openai_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder=(
                            "Leave blank to use the server's key"
                            if server_key_available
                            else "sk-..."
                        ),
                        value="",
                    )

                with gr.Row(visible=False) as local_settings:
                    gr.Textbox(
                        label="Local Model Status",
                        value="Model weights are downloaded and loaded on first use.",
                        interactive=False,
                    )

                # Advanced Settings Accordion
                with gr.Accordion("Advanced Parameters", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.2, value=0.3, step=0.1, label="Temperature"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=128, maximum=4096, value=1024, step=128, label="Max Output Tokens"
                    )
                    enable_thinking_chk = gr.Checkbox(
                        label="Enable Think/Reasoning Mode (Local MiniCPM5 only)",
                        value=True,
                        visible=False,
                    )

                gr.Markdown("### 💬 Chat Session")
                chatbot = gr.Chatbot(label="Chat History", type="messages", height=450)

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Type your question about the PDF and press Enter...",
                        show_label=False,
                        scale=8,
                        max_length=settings.max_question_chars,
                    )
                    send_btn = gr.Button(
                        "Send", variant="primary", scale=1, elem_classes="primary-btn"
                    )

                clear_chat_btn = gr.Button("Clear Chat History", size="sm")

        # Define Reactivity / Toggle settings panels
        def toggle_settings(model_choice):
            is_openai = model_choice == config.BACKEND_OPENAI
            return (
                gr.update(visible=is_openai),
                gr.update(visible=not is_openai),
                gr.update(visible=not is_openai),
            )

        model_type.change(
            fn=toggle_settings,
            inputs=[model_type],
            outputs=[openai_settings, local_settings, enable_thinking_chk],
        )

        # PDF Upload Handler
        def handle_pdf_load(file):
            if file is None:
                return "", 0, 0, None, "Page 0 of 0", gr.update(value=1)

            path = _uploaded_path(file)
            try:
                total = pdf_engine.validate_pdf(path, settings)
                img, _ = pdf_engine.render_pdf_page(path, 0)
            except PDFChatbotError as exc:
                # Shown as a toast; the previous document (if any) stays loaded.
                raise gr.Error(str(exc)) from exc
            return path, 0, total, img, f"Page 1 of {total}", gr.update(maximum=total, value=1)

        upload_outputs = [
            pdf_path,
            current_page,
            total_pages,
            pdf_image,
            page_label,
            page_jump_input,
        ]
        pdf_uploader.upload(fn=handle_pdf_load, inputs=[pdf_uploader], outputs=upload_outputs)
        pdf_uploader.clear(fn=lambda: handle_pdf_load(None), outputs=upload_outputs)

        # Navigation Handlers
        def navigate_page(direction, path, curr, total):
            if not path or not total:
                return curr, "Page 0 of 0", None, 1

            new_page = curr
            if direction == "prev":
                new_page = max(0, curr - 1)
            elif direction == "next":
                new_page = min(total - 1, curr + 1)

            img, _ = _render(path, new_page)
            return new_page, f"Page {new_page + 1} of {total}", img, new_page + 1

        nav_outputs = [current_page, page_label, pdf_image, page_jump_input]
        prev_page_btn.click(
            fn=navigate_page,
            inputs=[gr.State("prev"), pdf_path, current_page, total_pages],
            outputs=nav_outputs,
        )
        next_page_btn.click(
            fn=navigate_page,
            inputs=[gr.State("next"), pdf_path, current_page, total_pages],
            outputs=nav_outputs,
        )

        def jump_to_page(page_target, path, total):
            if not path or not total:
                return 0, "Page 0 of 0", None

            target_idx = max(0, min(total - 1, int(page_target or 1) - 1))
            img, _ = _render(path, target_idx)
            return target_idx, f"Page {target_idx + 1} of {total}", img

        jump_btn.click(
            fn=jump_to_page,
            inputs=[page_jump_input, pdf_path, total_pages],
            outputs=[current_page, page_label, pdf_image],
        )

        # Response Generation Handler (Streaming)
        def generate_response(
            history, question, model_sel, openai_key, path, temp, max_toks, think_mode, tot_pages
        ):
            question = (question or "").strip()
            if not question:
                yield (gr.skip(),) * 6
                return

            history = list(history or [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": ""})
            yield history, "", gr.skip(), gr.skip(), gr.skip(), gr.skip()

            shown_page = None
            updates = chat.run_turn(
                question, path, model_sel, openai_key, temp, int(max_toks), think_mode, settings
            )
            for update in updates:
                history[-1] = {"role": "assistant", "content": update.content}
                if update.page is not None and update.page != shown_page:
                    # Jump the viewer to the best-matching page once, not on every token.
                    img, total = _render(path, update.page)
                    shown_page = update.page
                    yield (
                        history,
                        "",
                        update.page,
                        f"Page {update.page + 1} of {total or tot_pages}",
                        img,
                        update.page + 1,
                    )
                else:
                    yield history, "", gr.skip(), gr.skip(), gr.skip(), gr.skip()

        chat_inputs = [
            chatbot,
            chat_input,
            model_type,
            openai_key_input,
            pdf_path,
            temperature_slider,
            max_tokens_slider,
            enable_thinking_chk,
            total_pages,
        ]
        chat_outputs = [chatbot, chat_input, current_page, page_label, pdf_image, page_jump_input]

        # Submit triggers
        chat_input.submit(fn=generate_response, inputs=chat_inputs, outputs=chat_outputs)
        send_btn.click(fn=generate_response, inputs=chat_inputs, outputs=chat_outputs)

        # Clear Chat trigger
        def clear_chat():
            return []

        clear_chat_btn.click(fn=clear_chat, outputs=[chatbot])

    demo.queue(
        default_concurrency_limit=settings.max_concurrent_requests,
        max_size=settings.queue_max_size,
    )
    return demo
