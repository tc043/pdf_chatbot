"""One question/answer turn: validate -> index -> retrieve -> generate.

The UI layer stays thin: it forwards form values to :func:`run_turn` and renders the
:class:`TurnUpdate` objects it yields. Keeping this logic free of Gradio makes it
unit-testable and keeps secrets out of process-global state.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from app import config, llm_engine, pdf_engine
from app.exceptions import PDFChatbotError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnUpdate:
    """A snapshot of the assistant message while a turn is in progress."""

    content: str
    page: int | None = None  # 0-based page to show in the viewer; None = leave unchanged
    is_error: bool = False


def resolve_openai_key(user_key: str | None, settings: config.Settings) -> str:
    """Pick the API key for this request.

    The user's own key wins. The server's ``OPENAI_API_KEY`` is only used as a
    fallback when enabled, and is never sent to the browser.
    """
    key = (user_key or "").strip()
    if key:
        return key
    if settings.use_server_key and settings.server_openai_api_key:
        return settings.server_openai_api_key
    return ""


def _error(message: str) -> TurnUpdate:
    return TurnUpdate(content=f"❌ {message}", is_error=True)


def run_turn(
    question: str,
    pdf_path: str | None,
    backend: str,
    user_api_key: str | None,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    settings: config.Settings | None = None,
) -> Iterator[TurnUpdate]:
    """Answer ``question`` about the PDF at ``pdf_path``, yielding progress and then text."""
    settings = settings or config.get_settings()
    try:
        yield from _run_turn(
            question,
            pdf_path,
            backend,
            user_api_key,
            temperature,
            max_tokens,
            enable_thinking,
            settings,
        )
    except PDFChatbotError as exc:
        yield _error(str(exc))
    except Exception:
        reference = uuid.uuid4().hex[:8]
        logger.exception("Unexpected error in chat turn (ref %s)", reference)
        yield _error(f"Something went wrong (reference {reference}). Please try again.")


def _run_turn(
    question: str,
    pdf_path: str | None,
    backend: str,
    user_api_key: str | None,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    settings: config.Settings,
) -> Iterator[TurnUpdate]:
    question = (question or "").strip()
    if not question:
        yield _error("Please type a question.")
        return
    if len(question) > settings.max_question_chars:
        yield _error(
            f"The question is {len(question)} characters; the limit is "
            f"{settings.max_question_chars}."
        )
        return
    if not pdf_path:
        yield _error("Please upload a PDF document first.")
        return
    if backend not in config.BACKENDS:
        yield _error("Unknown inference backend selected.")
        return

    api_key = ""
    if backend == config.BACKEND_OPENAI:
        api_key = resolve_openai_key(user_api_key, settings)
        if not api_key:
            yield _error("Please provide an OpenAI API key in the settings panel.")
            return

    logger.info("Chat turn: backend=%s question_chars=%d", backend, len(question))

    yield TurnUpdate("⏳ Indexing the PDF (first time only)...")
    db = pdf_engine.index_pdf(pdf_path, backend, api_key, settings)

    yield TurnUpdate("🔍 Searching the document...")
    docs = pdf_engine.retrieve_context(db, question, settings.retriever_k)
    if not docs:
        yield _error("No relevant context could be found in the PDF.")
        return

    best_page = int(docs[0].metadata.get("page", 0))
    prompt = llm_engine.format_rag_prompt(question, docs)

    if (
        backend == config.BACKEND_LOCAL
        and not llm_engine.LocalModelManager.get_instance().is_loaded
    ):
        yield TurnUpdate(
            "⏳ Loading the local model. The first use downloads about 2 GB and can take "
            "several minutes...",
            page=best_page,
        )
    else:
        yield TurnUpdate("✍️ Generating the answer...", page=best_page)

    if backend == config.BACKEND_OPENAI:
        stream = llm_engine.generate_openai_stream(
            prompt, api_key, temperature, max_tokens, settings=settings
        )
    else:
        stream = llm_engine.generate_local_stream(
            prompt, enable_thinking, temperature, max_tokens, settings=settings
        )

    answer = ""
    for piece in stream:
        answer += piece
        yield TurnUpdate(answer, page=best_page)
    if not answer.strip():
        yield _error("The model returned an empty answer. Please try rephrasing the question.")
