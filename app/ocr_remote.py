"""OCR through a remote, OpenAI-compatible vision server (for example Unlimited-OCR on vLLM).

Tesseract is the default OCR engine. This optional engine sends each scanned page, as one PNG
image, to a server you run and configure (``PDFCHAT_OCR_ENGINE=remote``). The page image leaves
this machine, so it is never enabled implicitly. The request mirrors the vLLM recipe for
``baidu/Unlimited-OCR`` and was run against it on vLLM (see docs/benchmark.md); other servers
may need a different prompt or extra parameters.
"""

from __future__ import annotations

import base64
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pymupdf

from app import config

logger = logging.getLogger(__name__)

REMOTE_OCR_DPI = 200
# Requests in flight per document; a vLLM server batches them.
REMOTE_CONCURRENCY = 4
# The model has no chat template: the text must start with a literal "<image>" tag.
PROMPT = "<image>document parsing."
MAX_TOKENS = 8192
# Per-request settings of the server-side no-repeat-ngram processor (single page window).
SERVER_PARAMS = {"skip_special_tokens": False, "vllm_xargs": {"ngram_size": 35, "window_size": 128}}

# Raw output carries layout markers, in one of two forms:
#   <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>   followed by the block text
#   <|det|>label [x1, y1, x2, y2]<|/det|>text
_REF_BLOCK = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*(?:<\|det\|>.*?<\|/det\|>)?", re.DOTALL)
_DET_BLOCK = re.compile(r"<\|det\|>\s*([^<\s\[]*)[^<]*<\|/det\|>", re.DOTALL)
_MARKER = re.compile(r"<\|[^<>|]{1,40}\|>|<｜[^<>｜]{1,40}｜>")
_SKIPPED_LABELS = {"image"}  # figures have no text to index


def clean_output(raw: str) -> str:
    """Turn the server's marked-up output into plain Markdown text, one paragraph per block."""
    blocks: list[tuple[str, list[str]]] = [("text", [])]
    for line in raw.splitlines():
        line = line.rstrip()
        while True:
            match = _REF_BLOCK.search(line) or _DET_BLOCK.search(line)
            if not match:
                break
            blocks.append((match.group(1).strip().lower() or "text", []))
            line = (line[: match.start()] + line[match.end() :]).strip()
        line = _MARKER.sub("", line).rstrip()
        if line:
            blocks[-1][1].append(line)
    text = "\n\n".join(
        "\n".join(lines) for label, lines in blocks if lines and label not in _SKIPPED_LABELS
    )
    return text.strip()


def _client(settings: config.Settings) -> Any:
    from openai import OpenAI

    return OpenAI(
        base_url=settings.ocr_remote_url,
        api_key=settings.ocr_remote_api_key or "EMPTY",
        timeout=settings.ocr_remote_timeout_s,
        max_retries=1,
    )


def _png_data_uri(page: pymupdf.Page) -> str:
    png = page.get_pixmap(dpi=REMOTE_OCR_DPI).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _recognize(image_url: str, page_number: int, settings: config.Settings) -> str:
    try:
        with _client(settings) as client:
            response = client.chat.completions.create(
                model=settings.ocr_remote_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                extra_body=SERVER_PARAMS,
            )
        return clean_output(response.choices[0].message.content or "")
    except Exception:
        # Never log the response or the image: both are the user's document.
        logger.warning("Remote OCR failed for page %d", page_number, exc_info=True)
        return ""


def recognize_page(page: pymupdf.Page, settings: config.Settings) -> str:
    """Read one page through the remote server; failures yield an empty string."""
    try:
        image_url = _png_data_uri(page)
    except Exception:
        logger.warning("Could not render page %d for remote OCR", page.number + 1, exc_info=True)
        return ""
    return _recognize(image_url, page.number + 1, settings)


def recognize_pages(pages: list[pymupdf.Page], settings: config.Settings) -> list[str]:
    """Read several pages with a few requests in flight; results keep the page order.

    Pages are rendered one after another (a PyMuPDF document is not thread-safe), and a page is
    rendered only once a request slot is free, so at most ``REMOTE_CONCURRENCY`` images are held
    at a time however long the document is.
    """
    slots = threading.BoundedSemaphore(REMOTE_CONCURRENCY)

    def read(image_url: str, page_number: int) -> str:
        try:
            return _recognize(image_url, page_number, settings)
        finally:
            slots.release()

    with ThreadPoolExecutor(max_workers=REMOTE_CONCURRENCY) as pool:
        futures: list[Future[str] | None] = []
        for page in pages:
            slots.acquire()
            try:
                image_url = _png_data_uri(page)
            except Exception:
                slots.release()
                logger.warning(
                    "Could not render page %d for remote OCR", page.number + 1, exc_info=True
                )
                futures.append(None)
                continue
            futures.append(pool.submit(read, image_url, page.number + 1))
    return [future.result() if future else "" for future in futures]
