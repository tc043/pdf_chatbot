import pymupdf

from app import config
from evals import qasper_benchmark as qb
from evals import qasper_scanned as qs


def text_pdf(path, pages):
    with pymupdf.open() as pdf:
        for text in pages:
            pdf.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=12)
        pdf.save(path)
    return path


SENTENCE = "The retrieval benchmark measures how often the right page is ranked near the top. " * 6


def test_rasterise_keeps_the_page_count_and_drops_the_text_layer(tmp_path):
    original = text_pdf(tmp_path / "a.pdf", [SENTENCE, SENTENCE])
    assert qs.has_text_layer(original)

    scan = qs.rasterise(original, tmp_path / "scan" / "a.pdf")

    with pymupdf.open(scan) as pdf:
        assert len(pdf) == 2
        assert all(not page.get_text().strip() for page in pdf)
        assert all(page.get_images() for page in pdf)
    assert not qs.has_text_layer(scan)


def test_rasterise_reuses_an_existing_copy(tmp_path):
    original = text_pdf(tmp_path / "a.pdf", [SENTENCE])
    target = tmp_path / "scan.pdf"
    target.write_bytes(b"kept")
    assert qs.rasterise(original, target).read_bytes() == b"kept"


def test_fidelity_is_the_mean_share_of_text_grams_found_in_the_ocr_page():
    grams = qb.shingles(qb.tokens("one two three four five six"))
    half = set(sorted(grams)[: len(grams) // 2 + 1])
    text = {0: grams, 1: grams, 2: set()}
    assert qs.fidelity(text, {0: grams, 1: set()}) == 0.5
    assert qs.fidelity(text, {0: grams, 1: grams}) == 1.0
    assert 0 < qs.fidelity({0: grams}, {0: half}) < 1
    assert qs.fidelity({0: set()}, {}) is None


class FakeDb:
    def get(self, include):
        assert "documents" in include and "metadatas" in include
        return {
            "documents": [
                "alpha beta gamma delta",
                "gamma delta epsilon zeta",
                "one two three four",
            ],
            "metadatas": [{"page": 0}, {"page": 0}, {"page": 1}],
        }


def test_indexed_page_shingles_union_the_chunks_of_each_page():
    pages = qs.indexed_page_shingles(FakeDb())
    assert set(pages) == {0, 1}
    assert ("alpha", "beta", "gamma", "delta") in pages[0]
    assert ("gamma", "delta", "epsilon", "zeta") in pages[0]
    assert ("one", "two", "three", "four") in pages[1]


def records():
    def row(paper, text, ocr):
        return {
            "paper": paper,
            "question_id": paper,
            "scores": {"text-hybrid": {"page@3": text}, "tesseract-hybrid": {"page@3": ocr}},
        }

    return [row("p1", 1.0, 0.0), row("p2", 1.0, 1.0), row("p3", 0.0, 0.0)]


def test_paired_difference_compares_arms_question_by_question():
    mean, low, high = qs.paired_difference(records(), "tesseract-hybrid", "text-hybrid", "page@3")
    assert abs(mean - (-1 / 3)) < 1e-9
    assert low <= mean <= high


def test_ocr_settings_use_the_app_chunk_defaults_and_a_separate_index(tmp_path):
    settings = qs.ocr_settings(config.OCR_ENGINE_REMOTE, "http://ocr.local/v1", "m")
    defaults = config.Settings()
    assert (settings.chunk_size, settings.chunk_overlap) == (
        defaults.chunk_size,
        defaults.chunk_overlap,
    )
    assert settings.ocr == config.OCR_AUTO
    assert settings.ocr_engine == config.OCR_ENGINE_REMOTE
    assert settings.ocr_remote_url == "http://ocr.local/v1"
    assert settings.chroma_dir != qb.eval_settings(settings.chunk_size).chroma_dir


def test_format_report_lists_every_arm_and_the_paired_differences():
    cell = {"mean": 0.5, "low": 0.4, "high": 0.6}
    arms = ["text-vector", "text-hybrid", "tesseract-vector", "tesseract-hybrid"]
    result = {
        "engine": "tesseract",
        "scan_dpi": 200,
        "questions": 10,
        "papers": 4,
        "chunk_size": 500,
        "ocr_fidelity_mean": 0.8,
        "scores": {arm: dict.fromkeys(qs.METRICS, cell) for arm in arms},
        "ocr_minus_text": {m: dict.fromkeys(qs.METRICS, cell) for m in qs.METHODS},
    }
    report = qs.format_report(result)
    assert "80.0%" in report
    assert all(arm in report for arm in arms)
    assert "tesseract - text, hybrid page@6" in report
