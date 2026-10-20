"""Offline retrieval-quality evaluation: vector vs BM25 vs hybrid on the demo PDFs.

For each question in ``questions.json`` the *gold pages* are the pages whose extracted text
contains the question's ``evidence`` phrase (so labels come from the PDF, not from memory).
A retriever scores a hit when one of its top-k chunks comes from a gold page. This measures
retrieval only, not answer quality.

Questions are tagged ``exact`` (numbers, symbols and terms copied from the document, which
favour keyword matching) or ``paraphrase`` (reworded, which favour embeddings). They are
reported separately: a hybrid retriever should help the first without hurting the second.

Uses the real local embedding model, so it needs ``requirements-local.txt`` and the first run
downloads ``PDFCHAT_LOCAL_EMBEDDING_MODEL``::

    python -m evals.retrieval_eval [--k 3] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from app import config, pdf_engine, retrieval

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = Path(__file__).with_name("questions.json")
METHODS = ("vector", "bm25", "hybrid")


def normalize(text: str) -> str:
    """Fold ligatures, case and whitespace so line breaks do not hide a phrase."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def gold_pages(page_texts: Sequence[str], evidence: str) -> set[int]:
    """0-based pages whose text contains ``evidence``; raises if none does."""
    needle = normalize(evidence)
    pages = {i for i, text in enumerate(page_texts) if needle in normalize(text)}
    if not pages:
        raise ValueError(f"Evidence phrase not found in the PDF: {evidence!r}")
    return pages


def first_hit_rank(retrieved_pages: Sequence[int], gold: set[int]) -> int | None:
    """1-based rank of the first retrieved chunk from a gold page, else ``None``."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in gold:
            return rank
    return None


@dataclass(frozen=True)
class Score:
    questions: int
    hit_rate: float  # share of questions with a gold page in the top k
    mrr: float  # mean reciprocal rank of the first gold page (0 when missed)


def summarize(ranks: Iterable[int | None]) -> Score:
    ranks = list(ranks)
    if not ranks:
        return Score(0, 0.0, 0.0)
    hits = [r for r in ranks if r is not None]
    return Score(len(ranks), len(hits) / len(ranks), sum(1 / r for r in hits) / len(ranks))


def _pages(docs: Iterable[Document]) -> list[int]:
    return [int(doc.metadata.get("page", -1)) for doc in docs]


def evaluate_pdf(
    pdf_path: Path, questions: Sequence[dict], k: int, settings: config.Settings
) -> dict[str, dict[str, list[int | None]]]:
    """Return ``{method: {kind: [first-hit rank per question]}}`` for one PDF."""
    with pymupdf.open(pdf_path) as pdf:
        page_texts = [page.get_text() for page in pdf]
    db = pdf_engine.index_pdf(str(pdf_path), config.BACKEND_LOCAL, settings=settings)
    keyword = retrieval._keyword_index(db)
    retrievers: dict[str, Callable[[str], list[Document]]] = {
        "vector": lambda q: db.similarity_search(q, k=k),
        "bm25": lambda q: keyword.search(q, k),
        "hybrid": lambda q: retrieval.search(db, q, k, config.RETRIEVAL_HYBRID),
    }
    ranks: dict[str, dict[str, list[int | None]]] = {m: defaultdict(list) for m in METHODS}
    for item in questions:
        gold = gold_pages(page_texts, item["evidence"])
        for method, retrieve in retrievers.items():
            ranks[method][item["kind"]].append(
                first_hit_rank(_pages(retrieve(item["question"])), gold)
            )
    return ranks


def run(k: int, pdf_filter: str | None = None) -> dict[str, dict[str, Score]]:
    """Evaluate every PDF in ``questions.json``; scores are keyed ``method -> subset``."""
    pooled: dict[str, dict[str, list[int | None]]] = {m: defaultdict(list) for m in METHODS}
    for entry in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")):
        if pdf_filter and pdf_filter not in entry["pdf"]:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            settings = config.Settings.from_env()
            settings = config.Settings(**{**settings.__dict__, "chroma_dir": Path(tmp)})
            ranks = evaluate_pdf(ROOT / entry["pdf"], entry["questions"], k, settings)
        for method in METHODS:
            for kind, values in ranks[method].items():
                pooled[method][kind].extend(values)
    scores: dict[str, dict[str, Score]] = {}
    for method in METHODS:
        by_kind = pooled[method]
        scores[method] = {kind: summarize(by_kind[kind]) for kind in sorted(by_kind)}
        scores[method]["all"] = summarize(r for values in by_kind.values() for r in values)
    return scores


def format_table(scores: dict[str, dict[str, Score]], k: int) -> str:
    subsets = [s for s in ("exact", "paraphrase", "all") if s in scores[METHODS[0]]]
    lines = [f"Retrieval eval (top {k} chunks); hit rate / MRR, higher is better", ""]
    header = f"{'method':<8}" + "".join(
        f"{s + ' (n=' + str(scores[METHODS[0]][s].questions) + ')':>22}" for s in subsets
    )
    lines.append(header)
    for method in METHODS:
        cells = "".join(
            f"{scores[method][s].hit_rate:>14.0%} /{scores[method][s].mrr:>5.2f}" for s in subsets
        )
        lines.append(f"{method:<8}{cells}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved per question")
    parser.add_argument("--pdf", help="only evaluate PDFs whose path contains this text")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args(argv)
    scores = run(args.k, args.pdf)
    if args.json:
        print(
            json.dumps({m: {s: vars(v) for s, v in d.items()} for m, d in scores.items()}, indent=2)
        )
    else:
        print(format_table(scores, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
