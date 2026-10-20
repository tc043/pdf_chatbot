import pytest
from langchain_core.documents import Document

from app import config, pdf_engine, retrieval
from app.exceptions import RetrievalError


def doc(text: str, page: int = 0) -> Document:
    return Document(page_content=text, metadata={"page": page})


# ------------------------------------------------------------------------ BM25


def test_tokenize_lowercases_and_drops_markdown_punctuation():
    assert retrieval.tokenize("| BLEU | 28.4 |\n## Table-2") == ["bleu", "28", "4", "table", "2"]


def test_bm25_ranks_the_chunk_with_the_rare_term_first():
    index = retrieval.BM25Index(
        [
            doc("the model uses attention over the input"),
            doc("the optimizer was Adam with warmup"),
            doc("the model and the input and the output"),
        ]
    )
    hits = index.search("which optimizer", 3)
    assert [h.page_content for h in hits] == ["the optimizer was Adam with warmup"]


def test_bm25_prefers_shorter_chunks_for_equal_term_counts():
    index = retrieval.BM25Index([doc("alpha " + "filler " * 50), doc("alpha beta")])
    assert index.search("alpha", 2)[0].page_content == "alpha beta"


@pytest.mark.parametrize("query", ["", "!!!", "nothing matches here"])
def test_bm25_returns_nothing_without_matching_terms(query):
    assert retrieval.BM25Index([doc("alpha beta")]).search(query, 3) == []


def test_bm25_on_an_empty_index():
    assert retrieval.BM25Index([]).search("alpha", 3) == []


def test_bm25_respects_k_and_is_deterministic_on_ties():
    index = retrieval.BM25Index([doc("alpha", 0), doc("alpha", 1), doc("alpha", 2)])
    assert [h.metadata["page"] for h in index.search("alpha", 2)] == [0, 1]


# ------------------------------------------------------------------------- RRF


def test_rrf_promotes_chunks_found_by_both_rankers_and_dedupes():
    a, b, c = doc("a", 0), doc("b", 1), doc("c", 2)
    fused = retrieval.reciprocal_rank_fusion([[a, b], [c, b]], k=3)
    assert [d.page_content for d in fused] == ["b", "a", "c"]


def test_rrf_truncates_to_k_and_keeps_vector_order_on_ties():
    a, b, c = doc("a"), doc("b", 1), doc("c", 2)
    fused = retrieval.reciprocal_rank_fusion([[a], [b], [c]], k=2)
    assert [d.page_content for d in fused] == ["a", "b"]


def test_identical_text_on_different_pages_is_not_collapsed():
    fused = retrieval.reciprocal_rank_fusion([[doc("same", 0), doc("same", 1)]], k=5)
    assert [d.metadata["page"] for d in fused] == [0, 1]


# ------------------------------------------------------- against a real Chroma

PAGES = [
    "Introduction to fruit. Apples and pears grow on trees in many orchards.",
    "Bananas are tropical. The identifier QZX4471 appears only in this paragraph.",
    "Vegetables grow in the soil: carrots, potatoes and onions are common roots.",
]


@pytest.fixture
def indexed(make_pdf, settings, fake_embeddings):
    return pdf_engine.index_pdf(make_pdf(PAGES), config.BACKEND_LOCAL, settings=settings)


def test_hybrid_finds_an_exact_identifier(indexed):
    hits = pdf_engine.retrieve_context(
        indexed, "what is QZX4471?", k=1, mode=config.RETRIEVAL_HYBRID
    )
    assert hits[0].metadata["page"] == 1


def test_hybrid_returns_k_distinct_chunks(indexed):
    hits = pdf_engine.retrieve_context(indexed, "fruit trees", k=3, mode=config.RETRIEVAL_HYBRID)
    keys = {(h.metadata["page"], h.page_content) for h in hits}
    assert len(hits) == len(keys) == 3


def test_vector_mode_is_plain_similarity_search(indexed):
    question = "fruit trees"
    expected = indexed.similarity_search(question, k=2)
    assert pdf_engine.retrieve_context(indexed, question, k=2) == expected
    assert (
        pdf_engine.retrieve_context(indexed, question, k=2, mode=config.RETRIEVAL_VECTOR)
        == expected
    )


def test_hybrid_without_keyword_matches_falls_back_to_vector_results(indexed):
    question = "zzzz"
    hits = pdf_engine.retrieve_context(indexed, question, k=2, mode=config.RETRIEVAL_HYBRID)
    assert hits == indexed.similarity_search(question, k=2)


def test_keyword_index_is_cached_per_collection_content(indexed):
    assert retrieval._keyword_index(indexed) is retrieval._keyword_index(indexed)


def test_keyword_index_cache_is_bounded(indexed, monkeypatch):
    monkeypatch.setattr(retrieval, "_cache", {})
    for i in range(retrieval._CACHE_SIZE + 3):
        monkeypatch.setattr(
            indexed, "get", lambda include=None, i=i: {"documents": [f"doc {i}"], "metadatas": [{}]}
        )
        retrieval._keyword_index(indexed)
    assert len(retrieval._cache) == retrieval._CACHE_SIZE


def test_hybrid_failure_building_the_keyword_index_is_wrapped(indexed, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sqlite exploded")

    monkeypatch.setattr(indexed, "get", boom)
    with pytest.raises(RetrievalError):
        pdf_engine.retrieve_context(indexed, "fruit", k=2, mode=config.RETRIEVAL_HYBRID)


def test_keyword_cache_does_not_mix_up_pdfs_with_identical_text_but_different_pages(
    indexed, monkeypatch
):
    monkeypatch.setattr(retrieval, "_cache", {})
    texts = ["alpha text", "beta text"]

    def stored(pages):
        return lambda include=None: {
            "documents": texts,
            "metadatas": [{"page": page} for page in pages],
        }

    monkeypatch.setattr(indexed, "get", stored([0, 1]))
    first = retrieval._keyword_index(indexed)
    # The same chunks after an extra cover page was added: every page number shifts by one.
    monkeypatch.setattr(indexed, "get", stored([1, 2]))
    second = retrieval._keyword_index(indexed)
    assert second is not first
    assert [d.metadata["page"] for d in second.search("alpha", 1)] == [1]
    assert [d.metadata["page"] for d in first.search("alpha", 1)] == [0]
