import pytest

from app import chat, config, ocr_remote, pdf_engine
from app.exceptions import ConfigError, PDFValidationError

URL = "http://ocr.example:8000/v1"


@pytest.fixture
def remote(settings):
    return config.Settings(
        **{**settings.__dict__, "ocr_engine": config.OCR_ENGINE_REMOTE, "ocr_remote_url": URL}
    )


class FakeClient:
    """Stands in for ``openai.OpenAI``; records the request and returns canned text."""

    def __init__(self, reply="page text", fail=False):
        self.reply, self.fail, self.requests = reply, fail, []
        self.chat = self
        self.completions = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise ConnectionError("server down")
        message = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


@pytest.fixture
def fake_server(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ocr_remote, "_client", lambda settings: client)
    return client


# ------------------------------------------------------------------ cleaning


def test_clean_output_handles_the_ref_det_form():
    raw = (
        "<|ref|>title<|/ref|><|det|>[[10, 10, 500, 60]]<|/det|>\n# Annual report\n"
        "<|ref|>text<|/ref|><|det|>[[10, 80, 500, 200]]<|/det|>\nRevenue grew 12%.\nCosts fell.\n"
        "<|ref|>image<|/ref|><|det|>[[0, 0, 9, 9]]<|/det|>\n<|ref|>text<|/ref|>"
        "<|det|>[[1, 2, 3, 4]]<|/det|>\nSecond paragraph.<｜end▁of▁sentence｜>"
    )
    assert ocr_remote.clean_output(raw) == (
        "# Annual report\n\nRevenue grew 12%.\nCosts fell.\n\nSecond paragraph."
    )


def test_clean_output_handles_the_det_only_form_and_drops_images():
    raw = (
        "<|det|>text [10, 10, 20, 20]<|/det|>First block\ncontinues here\n"
        "<|det|>image [0, 0, 5, 5]<|/det|>\n"
        "<|det|>table [1, 1, 2, 2]<|/det|>| a | b |"
    )
    assert ocr_remote.clean_output(raw) == "First block\ncontinues here\n\n| a | b |"


def test_clean_output_keeps_plain_text_and_handles_empty():
    assert ocr_remote.clean_output("just plain text\nsecond line") == "just plain text\nsecond line"
    assert ocr_remote.clean_output("") == ""


# ------------------------------------------------------------------- request


def test_the_request_follows_the_server_recipe(fake_server, make_scanned_pdf, remote):
    import pymupdf

    with pymupdf.open(make_scanned_pdf([["hello"]])) as doc:
        text = ocr_remote.recognize_page(doc[0], remote)
    assert text == "page text"
    request = fake_server.requests[0]
    text_part, image_part = request["messages"][0]["content"]
    assert text_part["text"].startswith("<image>")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert request["model"] == "baidu/Unlimited-OCR" and request["temperature"] == 0.0
    assert request["extra_body"]["skip_special_tokens"] is False


def test_a_failing_server_yields_empty_text(monkeypatch, make_scanned_pdf, remote):
    import pymupdf

    monkeypatch.setattr(ocr_remote, "_client", lambda settings: FakeClient(fail=True))
    with pymupdf.open(make_scanned_pdf([["hello"]])) as doc:
        assert ocr_remote.recognize_page(doc[0], remote) == ""


# ------------------------------------------------------------------- engine


def test_remote_ocr_is_available_once_a_url_is_set(remote, settings):
    assert pdf_engine.ocr_available(remote) is True
    assert pdf_engine.ocr_available(settings) is False  # tesseract is hidden in tests
    off = config.Settings(**{**remote.__dict__, "ocr": config.OCR_OFF})
    assert pdf_engine.ocr_available(off) is False


def test_scanned_pages_go_to_the_remote_server(fake_server, make_scanned_pdf, remote):
    docs = pdf_engine.extract_documents(make_scanned_pdf([["a"], ["b"]]), remote)
    assert [d.page_content for d in docs] == ["page text", "page text"]
    assert all(d.metadata["ocr"] is True for d in docs)
    assert len(fake_server.requests) == 2  # one request per page


def test_remote_failure_explains_what_to_check(monkeypatch, make_scanned_pdf, remote):
    monkeypatch.setattr(ocr_remote, "_client", lambda settings: FakeClient(fail=True))
    with pytest.raises(PDFValidationError, match="OCR server is reachable"):
        pdf_engine.extract_documents(make_scanned_pdf([["a"]]), remote)


def test_remote_and_tesseract_indexes_never_share_a_name(remote, settings, monkeypatch):
    tesseract = config.Settings(**{**settings.__dict__})
    monkeypatch.setattr(pdf_engine, "ocr_available", lambda s=None: True)
    names = {
        pdf_engine.get_collection_name("f" * 64, config.BACKEND_LOCAL, "m", 1000, 200, ocr=tag)
        for tag in (
            "",
            pdf_engine.ocr_tag(tesseract),
            pdf_engine.ocr_tag(remote),
            pdf_engine.ocr_tag(config.Settings(**{**remote.__dict__, "ocr_remote_model": "other"})),
        )
    }
    assert len(names) == 4
    assert pdf_engine.ocr_tag(tesseract) == "eng"  # unchanged, so existing indexes stay valid


def test_the_indexing_message_says_pages_are_sent_to_the_server(remote):
    assert chat.OCR_REMOTE_NOTE in chat.indexing_message(remote)


# -------------------------------------------------------------------- config


def test_remote_engine_needs_an_http_url():
    for url in ("", "ocr.example", "ftp://ocr.example/v1", "file:///etc/passwd"):
        with pytest.raises(ConfigError, match="OCR_REMOTE_URL"):
            config.Settings(ocr_engine=config.OCR_ENGINE_REMOTE, ocr_remote_url=url)
    config.Settings(ocr_engine=config.OCR_ENGINE_REMOTE, ocr_remote_url=URL)


def test_unknown_engine_is_rejected_and_key_is_not_in_repr():
    with pytest.raises(ConfigError, match="OCR_ENGINE"):
        config.Settings(ocr_engine="cloud")
    assert "sekret" not in repr(config.Settings(ocr_remote_api_key="sekret"))


def test_engine_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("PDFCHAT_OCR_ENGINE", "Remote")
    monkeypatch.setenv("PDFCHAT_OCR_REMOTE_URL", URL)
    monkeypatch.setenv("PDFCHAT_OCR_REMOTE_API_KEY", " k ")
    monkeypatch.setenv("PDFCHAT_OCR_REMOTE_TIMEOUT_S", "30")
    loaded = config.Settings.from_env()
    assert (loaded.ocr_engine, loaded.ocr_remote_url) == ("remote", URL)
    assert (loaded.ocr_remote_api_key, loaded.ocr_remote_timeout_s) == ("k", 30.0)


def test_output_seen_from_the_live_server_is_cleaned():
    raw = (
        "<|det|>aside_text [23, 268, 61, 702]<|/det|>arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n"
        "<|det|>title [343, 185, 655, 208]<|/det|>Attention Is All You Need\n"
        "<|det|>image [10, 10, 90, 90]<|/det|>\n"
        "<|det|>text [189, 294, 355, 338]<|/det|>Ashish Vaswani*\nGoogle Brain"
    )
    assert ocr_remote.clean_output(raw) == (
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n\nAttention Is All You Need\n\n"
        "Ashish Vaswani*\nGoogle Brain"
    )


def test_several_pages_are_read_in_order_and_a_failure_blanks_only_its_page(
    monkeypatch, make_scanned_pdf, remote
):
    import pymupdf

    def fake(image_url, page_number, settings):
        if page_number == 2:
            return ""
        return f"text of page {page_number}"

    monkeypatch.setattr(ocr_remote, "_recognize", fake)
    with pymupdf.open(make_scanned_pdf([["a"], ["b"], ["c"]])) as doc:
        texts = ocr_remote.recognize_pages(list(doc), remote)
    assert texts == ["text of page 1", "", "text of page 3"]


def test_only_a_few_page_images_are_held_at_once(monkeypatch, make_scanned_pdf, remote):
    import threading

    import pymupdf

    lock = threading.Lock()
    held = peak = 0
    real_render = ocr_remote._png_data_uri

    def render(page):
        nonlocal held, peak
        with lock:
            held += 1
            peak = max(peak, held)
        return real_render(page)

    def recognize(image_url, page_number, settings):
        nonlocal held
        with lock:
            held -= 1
        return f"page {page_number}"

    monkeypatch.setattr(ocr_remote, "_png_data_uri", render)
    monkeypatch.setattr(ocr_remote, "_recognize", recognize)
    with pymupdf.open(make_scanned_pdf([[str(i)] for i in range(12)])) as doc:
        texts = ocr_remote.recognize_pages(list(doc), remote)
    assert texts == [f"page {i}" for i in range(1, 13)]
    assert peak <= ocr_remote.REMOTE_CONCURRENCY


def test_another_server_gets_its_own_index(remote):
    elsewhere = config.Settings(**{**remote.__dict__, "ocr_remote_url": "http://other:8000/v1"})
    assert pdf_engine.ocr_tag(remote) != pdf_engine.ocr_tag(elsewhere)
    assert "other" not in pdf_engine.ocr_tag(elsewhere)
