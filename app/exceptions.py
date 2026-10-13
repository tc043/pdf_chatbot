"""Application exceptions.

Every exception derived from :class:`PDFChatbotError` carries a message that is
safe to show to end users (no stack traces, paths, or secrets). Anything else is
treated as an unexpected bug: it is logged with a traceback and the user only sees
a generic message with a reference id.
"""

from __future__ import annotations


class PDFChatbotError(Exception):
    """Base class for errors whose message is safe to display to users."""


class ConfigError(PDFChatbotError):
    """Invalid or inconsistent configuration (environment variables)."""


class PDFValidationError(PDFChatbotError):
    """The uploaded file is not a usable PDF (missing, too large, encrypted, ...)."""


class IndexingError(PDFChatbotError):
    """Building or loading the vector index failed."""


class RetrievalError(PDFChatbotError):
    """Searching the vector index failed."""


class ModelUnavailableError(PDFChatbotError):
    """A model backend cannot be used (missing key, missing extras, load failure)."""


class GenerationError(PDFChatbotError):
    """The language model failed while producing an answer."""


def wrap_provider_error(exc: BaseException, action: str) -> PDFChatbotError:
    """Translate an exception from a model provider into a user-safe error.

    ``action`` completes the sentence "Could not ...", e.g. ``"generate an answer"``.
    The original exception should be chained by the caller (``raise ... from exc``)
    so the traceback still reaches the logs.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - openai ships with langchain-openai
        openai = None

    if openai is not None:
        if isinstance(exc, openai.AuthenticationError):
            return ModelUnavailableError(
                "The OpenAI API key was rejected. Check that it is correct and still active."
            )
        if isinstance(exc, openai.RateLimitError):
            return ModelUnavailableError(
                "OpenAI rate limit or quota exceeded. Wait a moment or check your plan and billing."
            )
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return ModelUnavailableError(
                "Could not reach the OpenAI API (timeout or network problem). Please try again."
            )
    return PDFChatbotError(f"Could not {action}. Please try again; details are in the server log.")
