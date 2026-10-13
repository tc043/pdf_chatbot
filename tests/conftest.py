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
def fake_embeddings(monkeypatch) -> DeterministicFakeEmbedding:
    """Deterministic, offline embeddings: identical text gives identical vectors."""
    embeddings = DeterministicFakeEmbedding(size=32)
    monkeypatch.setattr(pdf_engine, "get_embeddings_model", lambda *a, **k: embeddings)
    return embeddings
