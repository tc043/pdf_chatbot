"""One question/answer turn: validate -> index -> retrieve -> generate.

The UI layer stays thin: it forwards form values to :func:`run_turn` and renders the
:class:`TurnUpdate` objects it yields. Keeping this logic free of Gradio makes it
unit-testable and keeps secrets out of process-global state.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app import config, index_store, llm_engine, pdf_engine, retrieval
from app.exceptions import PDFChatbotError

logger = logging.getLogger(__name__)

ERROR_PREFIX = "❌ "
# The local model has a small context window, so remembered turns are truncated.
HISTORY_QUESTION_CHARS = 500
HISTORY_ANSWER_CHARS = 1000
_REASONING_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


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
    return TurnUpdate(content=f"{ERROR_PREFIX}{message}", is_error=True)


def _message_text(content: Any) -> str:
    """Gradio message content is usually a string but can be a list of blocks or a dict."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(_message_text(part) for part in content)
    return ""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def history_pairs(messages: Sequence[Any] | None, turns: int) -> list[tuple[str, str]]:
    """Turn Gradio chat messages into the last ``turns`` (question, answer) pairs.

    Pairs whose answer is an error, a progress message or empty are dropped, as is a trailing
    question without an answer. The local model's ``<think>`` reasoning is not remembered.
    """
    if turns <= 0:
        return []
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        text = _message_text(message.get("content")).strip()
        if message.get("role") == "user":
            pending = text
        elif message.get("role") == "assistant" and pending is not None:
            answer = _REASONING_RE.sub("", text).strip()
            if pending and answer and not answer.startswith((ERROR_PREFIX, "⏳", "🔍", "✍️")):
                pairs.append(
                    (
                        _truncate(pending, HISTORY_QUESTION_CHARS),
                        _truncate(answer, HISTORY_ANSWER_CHARS),
                    )
                )
            pending = None
    return pairs[-turns:]


def _apply_retention(pdf_path: str, settings: config.Settings) -> None:
    """Enforce the optional index retention limits; a failure must not fail the question."""
    if not (settings.index_max_age_days or settings.index_max_count):
        return
    try:
        index_store.auto_prune(settings, protect=pdf_engine.file_fingerprint(pdf_path))
    except Exception:
        logger.warning("Index retention pass failed", exc_info=True)


def _retrieve(
    db: Any, question: str, history: Sequence[tuple[str, str]], settings: config.Settings
) -> list[Document]:
    """Retrieve context, also searching with the previous question for follow-ups.

    A follow-up such as "what about its cost?" has too little to search on. The question is
    searched alone and together with the previous question, and the two rankings are fused
    (the question alone is ranked first), so a change of topic cannot drown the new question.
    """
    k = settings.retriever_k
    mode = settings.retrieval_mode
    docs = pdf_engine.retrieve_context(db, question, k, mode=mode)
    if not history:
        return docs
    previous = pdf_engine.retrieve_context(db, f"{history[-1][0]} {question}", k, mode=mode)
    return retrieval.reciprocal_rank_fusion([docs, previous], k)


def run_turn(
    question: str,
    pdf_path: str | None,
    backend: str,
    user_api_key: str | None,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    settings: config.Settings | None = None,
    history: Sequence[tuple[str, str]] | None = None,
) -> Iterator[TurnUpdate]:
    """Answer ``question`` about the PDF at ``pdf_path``, yielding progress and then text.

    ``history`` holds earlier (question, answer) pairs from :func:`history_pairs`.
    """
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
            history or (),
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
    history: Sequence[tuple[str, str]],
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
    _apply_retention(pdf_path, settings)

    yield TurnUpdate("🔍 Searching the document...")
    docs = _retrieve(db, question, history, settings)
    if not docs:
        yield _error("No relevant context could be found in the PDF.")
        return

    best_page = int(docs[0].metadata.get("page", 0))
    prompt = llm_engine.format_rag_prompt(question, docs, history)

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
