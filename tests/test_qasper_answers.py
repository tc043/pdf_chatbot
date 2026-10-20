from evals import qasper_answers as qa


def test_token_f1_and_recall_use_squad_normalisation():
    assert qa.token_f1("The cat, sat!", "a cat sat") == 1.0
    assert qa.token_f1("dog", "cat") == 0.0
    assert round(qa.token_f1("cat sat down", "cat sat"), 3) == 0.8
    assert qa.token_recall("cat sat down now", "cat sat") == 1.0


def test_clean_answer_drops_reasoning_and_page_citations():
    raw = "<think>hmm</think>It uses BERT (p. 3) and LSTM (pp. 4, 5)."
    assert qa.normalize(qa.clean_answer(raw)) == "it uses bert and lstm"
    assert qa.clean_answer("<think>never closed") == ""


def test_leading_yes_no():
    assert qa.leading_yes_no("Yes, they do (p. 2).") == "yes"
    assert qa.leading_yes_no("No.") == "no"
    assert qa.leading_yes_no("Nobody knows") is None


def annotation(**kw):
    base = {"unanswerable": False, "yes_no": None, "extractive_spans": [], "free_form_answer": ""}
    return {"answer": {**base, **kw}}


def test_gold_answers_by_kind_skip_unanswerable():
    item = {
        "answers": [
            annotation(unanswerable=True),
            annotation(yes_no=False),
            annotation(extractive_spans=["BERT", "LSTM"]),
            annotation(free_form_answer=" it works "),
        ]
    }
    assert qa.gold_answers(item) == [
        ("boolean", "No"),
        ("extractive", "BERT, LSTM"),
        ("abstractive", "it works"),
    ]


def test_score_answer_takes_the_best_annotator():
    golds = [("extractive", "BERT"), ("abstractive", "a transformer encoder model")]
    best = qa.score_answer("BERT (p. 1)", golds)
    assert best["score"] == 1.0 and best["kind"] == "extractive" and best["words"] == 1
    yes = qa.score_answer("Yes, because of the results (p. 2)", [("boolean", "Yes")])
    assert yes["score"] == 1.0
    assert qa.score_answer("No.", [("boolean", "Yes")])["score"] == 0.0


def test_evidence_documents_respect_the_budget_and_skip_tables():
    item = {
        "answers": [
            annotation(unanswerable=True),
            {
                "answer": {
                    "unanswerable": False,
                    "evidence": ["x" * 2500, "FLOAT SELECTED: Table 1", "y" * 2500, "x" * 2500],
                }
            },
        ]
    }
    docs = qa.evidence_documents(item, {4, 2})
    assert [len(d.page_content) for d in docs] == [2500, 500]
    assert docs[0].metadata["page"] == 2


def rec(paper, default, small, words=(10, 10)):
    def arm(score, w):
        return {"kind": "extractive", "score": score, "recall": score, "words": w, "seconds": 1.0}

    return {
        "paper": paper,
        "arms": {"default": arm(default, words[0]), "small": arm(small, words[1])},
    }


def test_verdict_follows_the_pre_set_rule():
    better = [rec(f"p{i}", 0.2, 0.4) for i in range(10)]
    assert "clear improvement" in qa.verdict(better)
    same = [rec(f"p{i}", 0.3, 0.3) for i in range(10)]
    assert "not meaningfully worse" in qa.verdict(same)
    worse = [rec(f"p{i}", 0.4, 0.2) for i in range(10)]
    assert qa.verdict(worse).startswith("keep the default")
    longer = [rec(f"p{i}", 0.3, 0.3, words=(10, 20)) for i in range(10)]
    assert qa.verdict(longer).startswith("keep the default")
