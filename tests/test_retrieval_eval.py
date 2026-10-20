import pytest

from evals import retrieval_eval as ev


def test_normalize_folds_ligatures_case_and_line_breaks():
    assert ev.normalize("Speciﬁcally,\n  the   Model") == "specifically, the model"


def test_gold_pages_finds_phrases_split_across_lines():
    pages = ["nothing here", "the beam size\nof 4 was used", "beam size of 4 again"]
    assert ev.gold_pages(pages, "beam size of 4") == {1, 2}


def test_gold_pages_rejects_a_phrase_that_is_not_in_the_document():
    with pytest.raises(ValueError, match="not found"):
        ev.gold_pages(["alpha"], "beta")


def test_first_hit_rank_is_one_based_and_none_when_missed():
    assert ev.first_hit_rank([3, 1, 2], {1, 2}) == 2
    assert ev.first_hit_rank([3, 4], {1}) is None


def test_summarize_computes_hit_rate_and_mrr():
    score = ev.summarize([1, 2, None, 4])
    assert score.questions == 4
    assert score.hit_rate == 0.75
    assert score.mrr == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert ev.summarize([]) == ev.Score(0, 0.0, 0.0)


def test_every_question_has_a_known_kind_and_a_nonempty_evidence_phrase():
    import json

    entries = json.loads(ev.QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = [q for entry in entries for q in entry["questions"]]
    assert questions
    assert {q["kind"] for q in questions} <= {"exact", "paraphrase"}
    assert all(q["question"].strip() and q["evidence"].strip() for q in questions)
    assert all((ev.ROOT / entry["pdf"]).exists() for entry in entries)


def test_evidence_phrases_exist_in_the_demo_pdfs():
    """Guards the labels: a re-exported PDF that drops a phrase fails here, not silently."""
    import json

    import pymupdf

    for entry in json.loads(ev.QUESTIONS_FILE.read_text(encoding="utf-8")):
        with pymupdf.open(ev.ROOT / entry["pdf"]) as pdf:
            texts = [page.get_text() for page in pdf]
        for q in entry["questions"]:
            assert ev.gold_pages(texts, q["evidence"])


def test_format_table_lists_every_method():
    score = ev.Score(2, 0.5, 0.25)
    table = ev.format_table(
        {m: {"exact": score, "paraphrase": score, "all": score} for m in ev.METHODS}, 3
    )
    assert all(method in table for method in ev.METHODS)
    assert "50%" in table
