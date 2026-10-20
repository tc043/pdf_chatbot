import re
import sys

import httpx
import openai
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import config, pdf_engine
from app.exceptions import (
    IndexingError,
    ModelUnavailableError,
    PDFValidationError,
    RetrievalError,
)
from tests.samples import SAMPLE_PAGES

# ----------------------------------------------------------------- validation


def test_validate_returns_page_count(make_pdf, settings):
    assert pdf_engine.validate_pdf(make_pdf(SAMPLE_PAGES), settings) == 3


@pytest.mark.parametrize("value", [None, "", "/definitely/not/here.pdf"])
def test_validate_rejects_missing(value, settings):
    with pytest.raises(PDFValidationError):
        pdf_engine.validate_pdf(value, settings)


def test_validate_rejects_corrupt_pdf(tmp_path, settings):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"this is not a pdf at all")
    with pytest.raises(PDFValidationError):
        pdf_engine.validate_pdf(str(path), settings)


def test_validate_rejects_non_pdf_documents(tmp_path, settings):
    path = tmp_path / "notes.txt"
    path.write_text("plain text that pymupdf can open but is not a PDF")
    with pytest.raises(PDFValidationError, match="not a PDF"):
        pdf_engine.validate_pdf(str(path), settings)


def test_validate_rejects_oversized_file(tmp_path, settings):
    path = tmp_path / "big.pdf"
    path.write_bytes(b"0" * (settings.max_pdf_mb * 1024 * 1024 + 1))
    with pytest.raises(PDFValidationError, match="limit"):
        pdf_engine.validate_pdf(str(path), settings)


def test_validate_rejects_too_many_pages(make_pdf, settings):
    path = make_pdf(["x"] * (settings.max_pdf_pages + 1))
    with pytest.raises(PDFValidationError, match="pages"):
        pdf_engine.validate_pdf(path, settings)


def test_validate_rejects_password_protected(make_pdf, settings):
    import pymupdf

    path = make_pdf(
        ["secret"], encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
    )
    with pytest.raises(PDFValidationError, match="password"):
        pdf_engine.validate_pdf(path, settings)


# ------------------------------------------------------------------ rendering


def test_render_returns_image_and_total(make_pdf):
    image, total = pdf_engine.render_pdf_page(make_pdf(SAMPLE_PAGES), 0)
    assert total == 3
    assert image.mode == "RGB" and image.width > 0


def test_render_clamps_page_numbers(make_pdf):
    path = make_pdf(SAMPLE_PAGES)
    assert pdf_engine.render_pdf_page(path, -5)[1] == 3
    assert pdf_engine.render_pdf_page(path, 999)[0] is not None


@pytest.mark.parametrize("value", [None, "", "/nope.pdf"])
def test_render_without_file_returns_none(value):
    assert pdf_engine.render_pdf_page(value, 0) == (None, 0)


def test_render_corrupt_file_raises_validation_error(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"garbage")
    with pytest.raises(PDFValidationError):
        pdf_engine.render_pdf_page(str(path), 0)


# --------------------------------------------------------------------- naming


def test_fingerprint_depends_on_content_only(make_pdf):
    a = make_pdf(["same"], name="a.pdf")
    b = make_pdf(["same"], name="b.pdf")
    c = make_pdf(["different"], name="a.pdf")
    assert pdf_engine.file_fingerprint(a) == pdf_engine.file_fingerprint(a)
    assert pdf_engine.file_fingerprint(a) != pdf_engine.file_fingerprint(c)
    assert len(pdf_engine.file_fingerprint(b)) == 64


def test_collection_name_is_valid_and_bounded():
    name = pdf_engine.get_collection_name("a" * 64, config.BACKEND_OPENAI, "model", 1000, 200)
    assert re.fullmatch(r"[A-Za-z0-9_]+", name)
    assert 3 <= len(name) <= 60
    assert name == pdf_engine.get_collection_name(
        "a" * 64, config.BACKEND_OPENAI, "model", 1000, 200
    )


def test_collection_name_changes_with_every_index_input():
    base = ("a" * 64, config.BACKEND_OPENAI, "model", 1000, 200)
    names = {
        pdf_engine.get_collection_name(*base),
        pdf_engine.get_collection_name("b" * 64, *base[1:]),
        pdf_engine.get_collection_name(base[0], config.BACKEND_LOCAL, *base[2:]),
        pdf_engine.get_collection_name(*base[:2], "other-model", 1000, 200),
        pdf_engine.get_collection_name(*base[:3], 500, 200),
        pdf_engine.get_collection_name(*base[:4], 100),
    }
    assert len(names) == 6


# ------------------------------------------------------------------ embeddings


def test_openai_embeddings_require_a_key(settings):
    with pytest.raises(ModelUnavailableError, match="API key"):
        pdf_engine.get_embeddings_model(config.BACKEND_OPENAI, "", settings)


def test_openai_embeddings_use_configured_model(settings):
    emb = pdf_engine.get_embeddings_model(config.BACKEND_OPENAI, "sk-test-123456789", settings)
    assert emb.model == settings.openai_embedding_model


def test_local_embeddings_explain_missing_extras(monkeypatch, settings):
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)  # makes the import fail
    with pytest.raises(ModelUnavailableError, match=r"requirements-local\.txt"):
        pdf_engine.get_embeddings_model(config.BACKEND_LOCAL, None, settings)


# ------------------------------------------------------------------ extraction


def test_extract_skips_blank_pages_and_keeps_zero_based_page_numbers(make_pdf):
    docs = pdf_engine.extract_documents(make_pdf(["first page", "", "third page"], name="x.pdf"))
    assert [d.metadata["page"] for d in docs] == [0, 2]
    assert docs[0].metadata["source"] == "x.pdf"
    assert docs[0].metadata["okf_concept"] == "/pages/page_0001"
    assert docs[1].metadata["okf_concept"] == "/pages/page_0003"


def test_extract_rejects_pdfs_without_text(make_pdf):
    with pytest.raises(PDFValidationError, match="OCR"):
        pdf_engine.extract_documents(make_pdf(["", ""]))


def test_extract_documents_preserves_pdf_tables_as_markdown(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "table_doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 40), "Quarterly Performance Summary", fontsize=12)

    x0, y0, col_w, row_h = 50, 70, 120, 25
    for r in range(3):
        page.draw_line((x0, y0 + r * row_h), (x0 + 2 * col_w, y0 + r * row_h))
    for c in range(3):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + 2 * row_h))
    page.insert_text((x0 + 10, y0 + 17), "Metric", fontsize=10)
    page.insert_text((x0 + col_w + 10, y0 + 17), "Value", fontsize=10)
    page.insert_text((x0 + 10, y0 + row_h + 17), "Accuracy", fontsize=10)
    page.insert_text((x0 + col_w + 10, y0 + row_h + 17), "98.4%", fontsize=10)
    doc.save(pdf_path)
    doc.close()

    extracted = pdf_engine.extract_documents(str(pdf_path))
    assert len(extracted) == 1
    assert extracted[0].metadata["table_count"] >= 1
    compact = extracted[0].page_content.replace(" ", "")
    assert "|Metric|Value|" in compact
    assert "|Accuracy|98.4%|" in compact
    assert "QuarterlyPerformanceSummary" in compact


# -------------------------------------------------------------------- indexing


def _count(settings, name):
    db = Chroma(
        collection_name=name,
        embedding_function=DeterministicFakeEmbedding(size=32),
        persist_directory=str(settings.chroma_dir),
    )
    return len(db.get()["ids"])


def test_index_pdf_builds_a_searchable_index(make_pdf, settings, fake_embeddings):
    path = make_pdf(SAMPLE_PAGES)
    db = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    stored = db.get()
    assert len(stored["ids"]) >= 3
    assert {m["page"] for m in stored["metadatas"]} == {0, 1, 2}

    # Querying with a stored chunk's exact text must return that chunk first.
    target = next(
        d for d, m in zip(stored["documents"], stored["metadatas"], strict=True) if m["page"] == 1
    )
    hits = pdf_engine.retrieve_context(db, target, k=2)
    assert hits[0].page_content == target
    assert hits[0].metadata["page"] == 1


def test_index_pdf_writes_conformant_okf_v02_bundle(make_pdf, settings, fake_embeddings):
    path = make_pdf(SAMPLE_PAGES, name="fruits.pdf")
    pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)

    fp = pdf_engine.file_fingerprint(path)
    bundle_dir = pdf_engine.get_bundle_dir(fp, settings)

    assert (bundle_dir / "index.md").is_file()
    assert (bundle_dir / "document.md").is_file()
    assert (bundle_dir / "pages" / "index.md").is_file()

    root_index = (bundle_dir / "index.md").read_text(encoding="utf-8")
    assert 'okf_version: "0.2"' in root_index

    page1 = (bundle_dir / "pages" / "page_0001.md").read_text(encoding="utf-8")
    assert page1.startswith("---\n")
    assert 'type: "PDF Page"' in page1
    assert 'resource: "fruits.pdf#page=1"' in page1
    assert "status: stable" in page1
    assert "[Next Page](/pages/page_0002.md)" in page1
    assert "[Document Overview](/document.md)" in page1


def test_chroma_telemetry_is_disabled(make_pdf, settings, fake_embeddings):
    db = pdf_engine.index_pdf(make_pdf(SAMPLE_PAGES), config.BACKEND_LOCAL, settings=settings)
    assert db._client.get_settings().anonymized_telemetry is False
    assert db._client.get_settings().is_persistent is True
    assert str(settings.chroma_dir) in str(db._client.get_settings().persist_directory)


def test_existing_index_is_reused(make_pdf, settings, fake_embeddings, monkeypatch):
    path = make_pdf(SAMPLE_PAGES)
    pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)

    def boom(_):
        raise AssertionError("PDF was re-extracted although an index already exists")

    monkeypatch.setattr(pdf_engine, "extract_documents", boom)
    db = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert db.get(limit=1)["ids"]


def test_same_filename_different_content_gets_separate_indexes(make_pdf, settings, fake_embeddings):
    first = make_pdf(["alpha content about apples " * 10], name="report.pdf")
    second = make_pdf(["omega content about oceans " * 10], name="report.pdf")
    db1 = pdf_engine.index_pdf(first, config.BACKEND_LOCAL, settings=settings)
    db2 = pdf_engine.index_pdf(second, config.BACKEND_LOCAL, settings=settings)
    assert db1._collection.name != db2._collection.name
    assert all("alpha" in d for d in db1.get()["documents"])
    assert all("omega" in d for d in db2.get()["documents"])


def test_scanned_pdf_is_rejected_with_a_clear_message(make_pdf, settings, fake_embeddings):
    with pytest.raises(PDFValidationError, match="OCR"):
        pdf_engine.index_pdf(make_pdf(["", ""]), config.BACKEND_LOCAL, settings=settings)


def test_failed_indexing_leaves_no_partial_index(make_pdf, settings, fake_embeddings, monkeypatch):
    path = make_pdf(SAMPLE_PAGES)

    def fail(self, *args, **kwargs):
        raise RuntimeError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(Chroma, "add_documents", fail)
        with pytest.raises(IndexingError) as info:
            pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert "disk full" not in str(info.value)  # internals stay in the log

    # With the failure gone, a retry must rebuild the index from scratch.
    db = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert db.get(limit=1)["ids"]


def test_provider_errors_during_indexing_become_friendly(make_pdf, settings, monkeypatch):
    class RejectingEmbeddings(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com"))
            raise openai.AuthenticationError("nope", response=response, body=None)

    monkeypatch.setattr(
        pdf_engine, "get_embeddings_model", lambda *a, **k: RejectingEmbeddings(size=8)
    )
    with pytest.raises(ModelUnavailableError, match="key was rejected"):
        pdf_engine.index_pdf(make_pdf(SAMPLE_PAGES), config.BACKEND_OPENAI, "sk-x", settings)


def test_index_rejects_invalid_pdf_before_touching_models(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(
        pdf_engine, "get_embeddings_model", lambda *a, **k: pytest.fail("models must not load")
    )
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"nope")
    with pytest.raises(PDFValidationError):
        pdf_engine.index_pdf(str(bad), config.BACKEND_LOCAL, settings=settings)


# ------------------------------------------------------------------- retrieval


class _BrokenDB:
    def __init__(self, exc):
        self.exc = exc

    def similarity_search(self, *args, **kwargs):
        raise self.exc


def test_retrieval_failure_is_wrapped():
    with pytest.raises(RetrievalError):
        pdf_engine.retrieve_context(_BrokenDB(RuntimeError("chroma exploded")), "q", 3)


def test_retrieval_rate_limit_is_friendly():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
    exc = openai.RateLimitError("slow down", response=response, body=None)
    with pytest.raises(ModelUnavailableError, match="rate limit"):
        pdf_engine.retrieve_context(_BrokenDB(exc), "q", 3)
