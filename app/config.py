"""Application settings.

All tunables come from environment variables prefixed with ``PDFCHAT_`` (plus the
standard ``OPENAI_API_KEY``). Importing this module has no side effects: nothing is
read from the environment and no directories are created until
:func:`get_settings` / :func:`ensure_directories` are called.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.exceptions import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PREFIX = "PDFCHAT_"

# Backend labels shown in the UI. They are also the values passed through the code.
BACKEND_OPENAI = "OpenAI"
BACKEND_LOCAL = "Local Model (MiniCPM5-1B)"
BACKENDS = (BACKEND_OPENAI, BACKEND_LOCAL)

RETRIEVAL_VECTOR = "vector"
RETRIEVAL_HYBRID = "hybrid"
RETRIEVAL_MODES = (RETRIEVAL_VECTOR, RETRIEVAL_HYBRID)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _raw(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value.strip() if value is not None and value.strip() != "" else None


def _str(name: str, default: str) -> str:
    return _raw(name) or default


def _int(name: str, default: int, minimum: int, maximum: int | None = None) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}.") from None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ConfigError(f"{ENV_PREFIX}{name} must be {bound}, got {value}.")
    return value


def _float(name: str, default: float, minimum: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}.") from None
    # float() happily parses "nan" and "inf"; both slip past a ">= minimum" check
    # and would turn a timeout into "wait forever".
    if not math.isfinite(value):
        raise ConfigError(f"{ENV_PREFIX}{name} must be a finite number, got {raw!r}.")
    if value < minimum:
        raise ConfigError(f"{ENV_PREFIX}{name} must be >= {minimum}, got {value}.")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{ENV_PREFIX}{name} must be true or false, got {raw!r}.")


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings."""

    chroma_dir: Path = BASE_DIR / ".chromadb"

    # Models
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    local_model: str = "openbmb/MiniCPM5-1B-SFT"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 3
    retrieval_mode: str = RETRIEVAL_HYBRID

    # Limits
    max_pdf_mb: int = 25
    max_pdf_pages: int = 300
    max_question_chars: int = 2000

    # Provider behaviour
    openai_timeout_s: float = 60.0
    openai_max_retries: int = 2
    local_token_timeout_s: float = 120.0

    # Serving
    host: str = "127.0.0.1"
    port: int = 7860
    log_level: str = "INFO"
    max_concurrent_requests: int = 2
    queue_max_size: int = 20

    # Secrets. ``repr=False`` keeps them out of logs and tracebacks.
    use_server_key: bool = True
    server_openai_api_key: str = field(default="", repr=False)
    auth_user: str = ""
    auth_password: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"{ENV_PREFIX}CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"{ENV_PREFIX}CHUNK_SIZE ({self.chunk_size})."
            )
        if bool(self.auth_user) != bool(self.auth_password):
            raise ConfigError(
                f"Set both {ENV_PREFIX}AUTH_USER and {ENV_PREFIX}AUTH_PASSWORD, or neither."
            )
        for name in ("openai_timeout_s", "local_token_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive, finite number of seconds.")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ConfigError(
                f"{ENV_PREFIX}RETRIEVAL_MODE must be one of: {', '.join(RETRIEVAL_MODES)}."
            )
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"{ENV_PREFIX}LOG_LEVEL is not a valid logging level.")

    @property
    def auth(self) -> tuple[str, str] | None:
        """Credentials for Gradio's basic auth, or ``None`` when auth is disabled."""
        return (self.auth_user, self.auth_password) if self.auth_user else None

    @property
    def is_loopback_host(self) -> bool:
        return self.host in {"127.0.0.1", "localhost", "::1"}

    @classmethod
    def from_env(cls) -> Settings:
        chroma_dir = _raw("CHROMA_DIR")
        return cls(
            chroma_dir=Path(chroma_dir).expanduser() if chroma_dir else BASE_DIR / ".chromadb",
            openai_model=_str("OPENAI_MODEL", cls.openai_model),
            openai_embedding_model=_str("OPENAI_EMBEDDING_MODEL", cls.openai_embedding_model),
            local_model=_str("LOCAL_MODEL", cls.local_model),
            local_embedding_model=_str("LOCAL_EMBEDDING_MODEL", cls.local_embedding_model),
            chunk_size=_int("CHUNK_SIZE", cls.chunk_size, 100, 20000),
            chunk_overlap=_int("CHUNK_OVERLAP", cls.chunk_overlap, 0, 10000),
            retriever_k=_int("RETRIEVER_K", cls.retriever_k, 1, 20),
            retrieval_mode=_str("RETRIEVAL_MODE", cls.retrieval_mode).lower(),
            max_pdf_mb=_int("MAX_PDF_MB", cls.max_pdf_mb, 1, 1024),
            max_pdf_pages=_int("MAX_PDF_PAGES", cls.max_pdf_pages, 1, 10000),
            max_question_chars=_int("MAX_QUESTION_CHARS", cls.max_question_chars, 10, 100000),
            openai_timeout_s=_float("OPENAI_TIMEOUT_S", cls.openai_timeout_s, 1.0),
            openai_max_retries=_int("OPENAI_MAX_RETRIES", cls.openai_max_retries, 0, 10),
            local_token_timeout_s=_float("LOCAL_TOKEN_TIMEOUT_S", cls.local_token_timeout_s, 1.0),
            host=_str("HOST", cls.host),
            port=_int("PORT", cls.port, 1, 65535),
            log_level=_str("LOG_LEVEL", cls.log_level).upper(),
            max_concurrent_requests=_int(
                "MAX_CONCURRENT_REQUESTS", cls.max_concurrent_requests, 1, 64
            ),
            queue_max_size=_int("QUEUE_MAX_SIZE", cls.queue_max_size, 1, 1000),
            use_server_key=_bool("USE_SERVER_KEY", cls.use_server_key),
            server_openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            auth_user=_str("AUTH_USER", ""),
            auth_password=os.environ.get(ENV_PREFIX + "AUTH_PASSWORD", ""),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (read from the environment on first use)."""
    return Settings.from_env()


def ensure_directories(settings: Settings | None = None) -> None:
    """Create the on-disk locations the app writes to."""
    settings = settings or get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
