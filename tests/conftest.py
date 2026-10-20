"""Shared fixtures: isolated settings/env, PDF builders, and a fake embedding model."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import config, llm_engine, pdf_engine


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """No test may depend on the developer's real environment or leak state to the next."""
    for name in list(os.environ):
        if name.startswith(config.ENV_PREFIX) or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    config.get_settings.cache_clear()
    pdf_engine._local_embeddings.cache_clear()
    monkeypatch.setattr(llm_engine.LocalModelManager, "_instance", None)
    yield
    config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def hermetic_ocr(request, monkeypatch):
    """Tests do not depend on whether Tesseract happens to be installed on this machine.

    Tests marked ``ocr`` use the real program; others see it as missing unless they fake it.
    """
    if request.node.get_closest_marker("ocr") is None:
        monkeypatch.setattr(pdf_engine, "_tessdata_dir", lambda: None)


@pytest.fixture
def settings(tmp_path: Path) -> config.Settings:
    return config.Settings(
        chroma_dir=tmp_path / "chroma",
        max_pdf_mb=1,
        max_pdf_pages=10,
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., str]:
    """Create a PDF with one page per string ('' = a page with no text)."""
    counter = {"n": 0}

    def _make(pages: list[str], name: str | None = None, **save_kwargs) -> str:
        counter["n"] += 1
        folder = tmp_path / f"pdfs{counter['n']}"
        folder.mkdir()
        path = folder / (name or f"doc{counter['n']}.pdf")
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
        doc.save(path, **save_kwargs)
        doc.close()
        return str(path)

    return _make


@pytest.fixture
def make_scanned_pdf(tmp_path: Path) -> Callable[..., str]:
    """Create a PDF whose pages are pictures of text (no text layer): one list of lines per page."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    counter = {"n": 0}

    def _font():
        try:
            return ImageFont.load_default(size=44)
        except TypeError:  # Pillow < 10.1 has no scalable default font
            return ImageFont.truetype("DejaVuSans.ttf", 44)

    def _make(pages: list[list[str]], name: str | None = None) -> str:
        counter["n"] += 1
        folder = tmp_path / f"scans{counter['n']}"
        folder.mkdir()
        path = folder / (name or f"scan{counter['n']}.pdf")
        doc = pymupdf.open()
        for lines in pages:
            image = Image.new("L", (1240, 700), 255)
            draw = ImageDraw.Draw(image)
            for row, line in enumerate(lines):
                draw.text((60, 80 + row * 100), line, fill=0, font=_font())
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
            page = doc.new_page(width=620, height=350)
            page.insert_image(page.rect, stream=buffer.getvalue())
        doc.save(path, deflate=True, garbage=3)
        doc.close()
        return str(path)

    return _make


@pytest.fixture
def fake_embeddings(monkeypatch) -> DeterministicFakeEmbedding:
    """Deterministic, offline embeddings: identical text gives identical vectors."""
    embeddings = DeterministicFakeEmbedding(size=32)
    monkeypatch.setattr(pdf_engine, "get_embeddings_model", lambda *a, **k: embeddings)
    return embeddings
