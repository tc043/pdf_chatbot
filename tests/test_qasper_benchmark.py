from langchain_core.documents import Document

from evals import qasper_benchmark as qb


def doc(text, page):
    return Document(page_content=text, metadata={"page": page})


PAGES = {
    0: qb.shingles(qb.tokens("alpha beta gamma delta epsilon zeta eta theta iota kappa")),
    1: qb.shingles(qb.tokens("one two three four five six seven eight nine ten eleven twelve")),
}


def test_tokens_undo_hyphenation_and_drop_qasper_placeholders():
    assert qb.tokens("seg-\nmentation of BIBREF12 text INLINEFORM0 here") == [
        "segmentation",
        "of",
        "text",
        "here",
    ]


def test_match_evidence_finds_the_page_that_holds_the_paragraph():
    found = qb.match_evidence("two three four five six seven eight", PAGES)
    assert found is not None and found.pages == frozenset({1})


def test_match_evidence_rejects_text_that_is_not_in_the_pdf():
    assert qb.match_evidence("completely different words appear in this sentence", PAGES) is None
    assert qb.match_evidence("too short", PAGES) is None


def test_match_evidence_can_span_two_pages():
    text = "alpha beta gamma delta epsilon zeta one two three four five six"
    found = qb.match_evidence(text, PAGES)
    assert found is not None and found.pages == frozenset({0, 1})


def paper(answers):
    return {"qas": [{"question_id": "q", "question": " Why? ", "answers": answers}]}


def answer(evidence, unanswerable=False):
    return {"answer": {"unanswerable": unanswerable, "evidence": evidence}}


def test_build_questions_counts_every_drop_reason():
    dropped = qb.Dropped()
    kept = qb.build_questions(
        "p", paper([answer(["two three four five six seven eight"])]), PAGES, dropped
    )
    assert [(q.text, q.gold_pages) for q in kept] == [("Why?", frozenset({1}))]
    for answers in (
        [answer([], unanswerable=True)],
        [answer(["FLOAT SELECTED: Table 1: a b c d e f"])],
        [answer(["nothing here matches the document at all"])],
    ):
        assert qb.build_questions("p", paper(answers), PAGES, dropped) == []
    assert (dropped.kept, dropped.unanswerable, dropped.no_usable_evidence) == (1, 1, 1)
    assert (dropped.evidence_not_found, dropped.total) == (1, 4)


def test_scores_use_the_first_gold_page_and_evidence_chunks():
    evidence = qb.shingles(qb.tokens("two three four five six seven"))
    docs = [doc("unrelated words everywhere in this chunk", 0), doc("two three four five six", 1)]
    scores = qb.question_scores(docs, frozenset({1}), evidence)
    assert scores["page@1"] == 0 and scores["page@3"] == 1
    assert scores["chunk@1"] == 0 and scores["chunk@3"] == 1
    assert scores["mrr10"] == 0.5


def test_bootstrap_ci_brackets_the_mean_and_handles_empty():
    mean, low, high = qb.bootstrap_ci({"a": (3.0, 4), "b": (1.0, 4), "c": (2.0, 4)})
    assert mean == 0.5 and low <= mean <= high
    assert qb.bootstrap_ci({}) == (0.0, 0.0, 0.0)


def test_summarize_reports_paired_difference_against_vector():
    def rec(paper, vector, hybrid):
        return {"paper": paper, "scores": {"vector": {"m": vector}, "hybrid": {"m": hybrid}}}

    out = qb.summarize(
        [rec("a", 0, 1), rec("b", 0, 1), rec("c", 1, 1)], ("vector", "hybrid"), ["m"]
    )
    assert out["hybrid_minus_vector"]["m"]["mean"] == 2 / 3
    assert out["questions"] == 3 and out["papers"] == 3
