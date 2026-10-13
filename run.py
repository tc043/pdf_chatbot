"""Entry point: ``python run.py``. See README.md for the PDFCHAT_* settings."""

from __future__ import annotations

import logging
import signal
import sys

from app import config
from app.exceptions import ConfigError
from app.logging_config import setup_logging


def _terminate(signum: int, frame: object) -> None:
    # Turn SIGTERM (e.g. `docker stop`) into the same clean shutdown path as Ctrl+C.
    raise KeyboardInterrupt


def main() -> int:
    try:
        settings = config.get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(settings.log_level)
    log = logging.getLogger("pdf_chatbot")

    if not settings.is_loopback_host:
        if settings.auth is None:
            log.warning(
                "Listening on %s without authentication. Anyone who can reach this port can "
                "upload files and use the model. Set PDFCHAT_AUTH_USER/PDFCHAT_AUTH_PASSWORD "
                "or put the app behind an authenticating reverse proxy.",
                settings.host,
            )
        if settings.use_server_key and settings.server_openai_api_key:
            log.warning(
                "OPENAI_API_KEY is set and PDFCHAT_USE_SERVER_KEY is enabled, so every visitor "
                "can spend your OpenAI credit. Set PDFCHAT_USE_SERVER_KEY=false to require "
                "users to bring their own key."
            )

    signal.signal(signal.SIGTERM, _terminate)
    try:
        config.ensure_directories(settings)
        from app.ui import create_ui

        demo = create_ui(settings)
        demo.launch(
            server_name=settings.host,
            server_port=settings.port,
            share=False,
            auth=settings.auth,
            show_api=False,
            max_file_size=f"{settings.max_pdf_mb}mb",
        )
    except KeyboardInterrupt:
        log.info("Shutting down PDF Chatbot.")
        return 0
    except Exception:
        log.exception("The server failed to start")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
