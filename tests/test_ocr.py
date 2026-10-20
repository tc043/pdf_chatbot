import shutil

import pymupdf
import pytest

from app import chat, config, pdf_engine
from app.exceptions import PDFValidationError

SCAN_LINES = ["The quick brown fox jumps", "Invoice total 4271 dollars"]


@pytest.fixture
def fake_tesseract(tmp_path, monkeypatch):
    """Pretend Tesseract is installed with English data; OCR returns the text of each page."""
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"")
    monkeypatch.setattr(pdf_engine, "_tessdata_dir", lambda: tessdata)
    calls = []

    def fake_ocr(page, language):
        calls.append((page.number, language))
        return f"scanned text of page {page.number + 1}"

    monkeypatch.setattr(pdf_engine, "_ocr_page", fake_ocr)
    return calls


# ---------------------------------------------------------------- availability


def test_ocr_is_unavailable_without_tesseract(settings):
    assert pdf_engine.ocr_available(settings) is False


def test_ocr_is_available_with_the_requested_language_only(fake_tesseract, settings):
    assert pdf_engine.ocr_available(settings) is True
    german = config.Settings(**{**settings.__dict__, "ocr_language": "eng+deu"})
    assert pdf_engine.ocr_available(german) is False  # deu.traineddata is missing


def test_ocr_can_be_switched_off(fake_tesseract, settings):
    off = config.Settings(**{**settings.__dict__, "ocr": config.OCR_OFF})
    assert pdf_engine.ocr_available(off) is False


# ------------------------------------------------------------------ extraction


def test_scanned_pages_are_read_with_ocr_and_tagged(fake_tesseract, make_scanned_pdf, settings):
    docs = pdf_engine.extract_documents(make_scanned_pdf([SCAN_LINES, SCAN_LINES]), settings)
    assert [d.page_content for d in docs] == [
        "scanned text of page 1",
        "scanned text of page 2",
    ]
    assert all(d.metadata["ocr"] is True for d in docs)
    assert fake_tesseract == [(0, "eng"), (1, "eng")]


def test_pages_with_a_text_layer_are_never_ocred(fake_tesseract, make_pdf, settings):
    docs = pdf_engine.extract_documents(make_pdf(["real text", ""]), settings)
    assert [d.metadata["page"] for d in docs] == [0]
    assert "ocr" not in docs[0].metadata
    assert fake_tesseract == []  # the blank page has no image, so there is nothing to read


def test_mixed_documents_keep_page_order(fake_tesseract, make_pdf, make_scanned_pdf, settings):
    text_pdf = make_pdf(["typed page"])
    scan_pdf = make_scanned_pdf([SCAN_LINES])
    with pymupdf.open() as merged:
        merged.insert_pdf(pymupdf.open(scan_pdf))
        merged.insert_pdf(pymupdf.open(text_pdf))
        path = str(scan_pdf).replace(".pdf", "-mixed.pdf")
        merged.save(path)
    docs = pdf_engine.extract_documents(path, settings)
    assert [(d.metadata["page"], bool(d.metadata.get("ocr"))) for d in docs] == [
        (0, True),
        (1, False),
    ]


def test_ocr_page_limit_is_enforced_with_a_clear_message(
    fake_tesseract, make_scanned_pdf, settings
):
    small = config.Settings(**{**settings.__dict__, "ocr_max_pages": 1})
    with pytest.raises(PDFValidationError, match=r"2 pages.*limit is 1.*OCR_MAX_PAGES"):
        pdf_engine.extract_documents(make_scanned_pdf([SCAN_LINES, SCAN_LINES]), small)
    assert fake_tesseract == []  # fails before spending any time on OCR


def test_a_scan_that_ocr_cannot_read_is_reported_as_such(
    fake_tesseract, make_scanned_pdf, settings, monkeypatch
):
    monkeypatch.setattr(pdf_engine, "_ocr_page", lambda page, language: "")
    with pytest.raises(PDFValidationError, match="OCR could not read any text"):
        pdf_engine.extract_documents(make_scanned_pdf([SCAN_LINES]), settings)


@pytest.mark.parametrize(
    ("ocr", "expected"),
    [
        (config.OCR_AUTO, "install Tesseract"),
        (config.OCR_OFF, "PDFCHAT_OCR=auto"),
    ],
)
def test_error_names_the_real_reason_when_ocr_is_not_possible(
    make_scanned_pdf, settings, ocr, expected
):
    custom = config.Settings(**{**settings.__dict__, "ocr": ocr})
    with pytest.raises(PDFValidationError, match=expected):
        pdf_engine.extract_documents(make_scanned_pdf([SCAN_LINES]), custom)


def test_pdf_without_text_or_images_says_there_is_nothing_to_read(make_pdf, settings):
    with pytest.raises(PDFValidationError, match="no scanned page images"):
        pdf_engine.extract_documents(make_pdf(["", ""]), settings)


def test_ocr_failure_on_one_page_does_not_lose_the_others(
    fake_tesseract, make_scanned_pdf, settings, monkeypatch
):
    def flaky(page, language):
        return "" if page.number == 0 else "second page text"

    monkeypatch.setattr(pdf_engine, "_ocr_page", flaky)
    docs = pdf_engine.extract_documents(make_scanned_pdf([SCAN_LINES, SCAN_LINES]), settings)
    assert [d.metadata["page"] for d in docs] == [1]


def test_ocr_page_swallows_tesseract_errors():
    class Page:
        number = 3

        def get_textpage_ocr(self, **kwargs):
            raise RuntimeError("tesseract crashed")

    assert pdf_engine._ocr_page(Page(), "eng") == ""


# -------------------------------------------------------------------- indexing


def test_collection_name_changes_when_ocr_becomes_active():
    args = ("f" * 64, config.BACKEND_LOCAL, "m", 1000, 200)
    plain = pdf_engine.get_collection_name(*args)
    assert pdf_engine.get_collection_name(*args, ocr="") == plain  # OCR off: names unchanged
    assert pdf_engine.get_collection_name(*args, ocr="eng") != plain
    assert pdf_engine.get_collection_name(*args, ocr="eng") != pdf_engine.get_collection_name(
        *args, ocr="deu"
    )


def test_scanned_pdf_is_indexed_and_marked_in_its_okf_bundle(
    fake_tesseract, make_scanned_pdf, settings, fake_embeddings
):
    path = make_scanned_pdf([SCAN_LINES])
    db = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert db.get()["metadatas"][0]["ocr"] is True
    bundle = pdf_engine.get_bundle_dir(pdf_engine.file_fingerprint(path), settings)
    page_file = next((bundle / "pages").glob("*.md"))
    assert '"ocr"' in page_file.read_text(encoding="utf-8")


def test_turning_ocr_on_rebuilds_instead_of_reusing_an_index_without_it(
    make_pdf, make_scanned_pdf, settings, fake_embeddings, monkeypatch, tmp_path
):
    # A document with one typed page and one scanned page, indexed while OCR is unavailable.
    scan = make_scanned_pdf([SCAN_LINES])
    typed = make_pdf(["typed page text"])
    with pymupdf.open() as merged:
        merged.insert_pdf(pymupdf.open(typed))
        merged.insert_pdf(pymupdf.open(scan))
        path = str(tmp_path / "mixed.pdf")
        merged.save(path)
    before = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert {m["page"] for m in before.get()["metadatas"]} == {0}

    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"")
    monkeypatch.setattr(pdf_engine, "_tessdata_dir", lambda: tessdata)
    monkeypatch.setattr(pdf_engine, "_ocr_page", lambda page, language: "recognised words")
    after = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert {m["page"] for m in after.get()["metadatas"]} == {0, 1}


def test_chat_warns_that_indexing_may_be_slow_when_ocr_is_active(
    fake_tesseract, settings, monkeypatch
):
    monkeypatch.setattr(
        pdf_engine, "index_pdf", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    first = next(chat.run_turn("q?", "/x.pdf", config.BACKEND_LOCAL, "", 0.3, 256, True, settings))
    assert "OCR" in first.content


# ------------------------------------------------------------------- config


@pytest.mark.parametrize("value", ["../etc", "eng+", "+eng", "eng deu", "e;ng"])
def test_ocr_language_must_look_like_tesseract_codes(monkeypatch, value):
    monkeypatch.setenv("PDFCHAT_OCR_LANGUAGE", value)
    with pytest.raises(config.ConfigError):
        config.Settings.from_env()


def test_ocr_settings_have_defaults_and_validate(monkeypatch):
    settings = config.Settings.from_env()
    assert (settings.ocr, settings.ocr_language, settings.ocr_max_pages) == ("auto", "eng", 50)
    monkeypatch.setenv("PDFCHAT_OCR", "OFF")
    monkeypatch.setenv("PDFCHAT_OCR_LANGUAGE", "eng+deu")
    assert config.Settings.from_env().ocr == "off"
    monkeypatch.setenv("PDFCHAT_OCR", "maybe")
    with pytest.raises(config.ConfigError, match="OCR"):
        config.Settings.from_env()


# ------------------------------------------------------- the real Tesseract


@pytest.mark.ocr
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
def test_real_tesseract_reads_a_scanned_page(make_scanned_pdf, settings):
    docs = pdf_engine.extract_documents(make_scanned_pdf([SCAN_LINES]), settings)
    text = " ".join(d.page_content.lower() for d in docs)
    assert "quick brown fox" in text and "4271" in text
    assert docs[0].metadata["ocr"] is True
