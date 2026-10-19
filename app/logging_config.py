"""Logging setup with secret redaction."""

from __future__ import annotations

import logging
import re

# Matches OpenAI-style keys (sk-..., sk-proj-...) so they never reach log files.
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_HANDLER_FLAG = "_pdfchat_handler"


class RedactSecretsFilter(logging.Filter):
    """Replace API-key-looking tokens in log messages with ``sk-***``."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _SECRET_RE.sub("sk-***", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once; calling it again only updates the level."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(h, _HANDLER_FLAG, False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    handler.addFilter(RedactSecretsFilter())
    setattr(handler, _HANDLER_FLAG, True)
    root.addHandler(handler)
