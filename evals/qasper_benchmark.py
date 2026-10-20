"""Retrieval benchmark on Qasper: real arXiv papers, questions written by other people.

Qasper (Dasigi et al., 2021, CC BY 4.0) has questions about NLP papers that were written by
readers who saw only the title and abstract, and annotators marked the paragraphs of the paper
that answer each one. This script downloads the papers as PDFs, runs them through this app's
own extraction, chunking, embedding and retrieval code, and measures whether the retrieved
chunks come from the pages that hold the evidence. It measures retrieval only, not answers.

Gold labels. Each evidence paragraph is matched to the PDF pages that contain its text (word
4-gram containment, see :func:`match_evidence`). A question is kept when at least one answerable
annotation has an evidence paragraph that can be located. Table and figure evidence
("FLOAT SELECTED"), unanswerable questions, papers that cannot be downloaded and evidence that
cannot be located are dropped and counted in the output.

Metrics (k = 1, 2, 3, 5, 6, 10; all with 95% bootstrap intervals that resample *papers*,
because questions about the same paper are not independent):
  * page hit@k: one of the top-k chunks comes from a gold page (what the viewer uses);
  * chunk hit@k: one of the top-k chunks is mostly evidence text (at least half of its word
    4-grams occur in the evidence paragraphs), which does not reward a lucky page;
  * MRR@10 of the first gold page.

Decision rule (written before the test split was scored):
  1. Matcher thresholds, the arXiv version and the report format were chosen on the validation
     split only. The test split is scored once.
  2. Hybrid stays the default retrieval mode unless the paired difference (hybrid - vector) in
     page hit@3 has a 95% interval entirely below zero. In that case the default goes back to
     vector. BM25 alone is never the default, whatever its score, because it cannot match
     paraphrases without shared words; it is reported for reference.
  3. The default chunk size (1000) changes only if another size, compared at an equal context
     budget (k x chunk size = 3000 characters), beats it in page hit with an interval for the
     difference that excludes zero.

Usage (needs requirements-local.txt; downloads papers from arXiv, about 3 s apart)::

    python -m evals.qasper_benchmark fetch --split validation --limit 40
    python -m evals.qasper_benchmark inspect --split validation --limit 40
    python -m evals.qasper_benchmark run --split validation --limit 40 --out results.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import config, pdf_engine, retrieval

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".eval_cache"
MANIFEST = Path(__file__).with_name("qasper_manifest.json")
USER_AGENT = "pdf-chatbot-benchmark/1.0 (+https://github.com/tc043/pdf_chatbot)"
ARXIV_DELAY_S = 3.0

DATA_BASE = "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
SPLITS = {
    "validation": ("qasper-train-dev-v0.3.tgz", "qasper-dev-v0.3.json"),
    "test": ("qasper-test-and-evaluator-v0.3.tgz", "qasper-test-v0.3.json"),
}

# Qasper replaces citations, formulas and table/figure references with placeholder tokens.
_PLACEHOLDER = re.compile(r"\b(?:BIBREF|INLINEFORM|TABREF|FIGREF|SECREF|DISPLAYFORM)\d*\b")

SHINGLE = 4  # words per shingle
PAGE_MIN = 0.2  # a page "holds" an evidence paragraph if it contains this share of its shingles
MIN_COVER = 0.6  # ...and the pages together must contain at least this share to count as found
CHUNK_EVIDENCE_MIN = 0.5  # share of a chunk's shingles that must be evidence to be a chunk hit
KS = (1, 2, 3, 5, 6, 10)
CONTEXT_BUDGET = 3000  # characters of context: k x chunk size, for comparing chunk sizes
METHODS = ("vector", "bm25", "hybrid")
BOOTSTRAP_ROUNDS = 2000
FLOAT_PREFIX = "FLOAT SELECTED"


# --------------------------------------------------------------------------- #
# Text matching (pure functions)
# --------------------------------------------------------------------------- #
def tokens(text: str) -> list[str]:
    """Lowercase word tokens, with hyphenation at line ends undone and ligatures folded."""
    text = unicodedata.normalize("NFKC", text)
    text = _PLACEHOLDER.sub(" ", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.findall(r"\w+", text.lower())


def shingles(words: Sequence[str], size: int = SHINGLE) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


@dataclass(frozen=True)
class EvidenceMatch:
    pages: frozenset[int]  # 0-based pages that hold the paragraph
    shingles: frozenset[tuple[str, ...]]  # the paragraph's shingles (for the chunk metric)


def match_evidence(
    evidence: str, page_shingles: dict[int, set[tuple[str, ...]]]
) -> EvidenceMatch | None:
    """Locate an evidence paragraph in the PDF, or return ``None`` if it cannot be found."""
    wanted = shingles(tokens(evidence))
    if not wanted:
        return None
    pages = {
        page for page, have in page_shingles.items() if len(wanted & have) / len(wanted) >= PAGE_MIN
    }
    covered = set().union(*(wanted & page_shingles[p] for p in pages)) if pages else set()
    if len(covered) / len(wanted) < MIN_COVER:
        return None
    return EvidenceMatch(frozenset(pages), frozenset(wanted))


@dataclass
class Question:
    paper: str
    question_id: str
    text: str
    gold_pages: frozenset[int]
    evidence_shingles: frozenset[tuple[str, ...]]


@dataclass
class Dropped:
    """Counts of what was left out, so the denominator is never hidden."""

    papers_without_pdf: int = 0
    unanswerable: int = 0
    no_usable_evidence: int = 0
    evidence_not_found: int = 0
    evidence_paragraphs: int = 0
    evidence_paragraphs_matched: int = 0
    kept: int = 0
    total: int = 0
    extra: dict[str, int] = field(default_factory=dict)


def build_questions(
    paper_id: str,
    paper: dict[str, Any],
    page_shingles: dict[int, set[tuple[str, ...]]],
    dropped: Dropped,
) -> list[Question]:
    """Turn one Qasper paper into questions with gold pages, updating the drop counts."""
    questions = []
    for qa in paper["qas"]:
        dropped.total += 1
        answers = [a["answer"] for a in qa["answers"] if not a["answer"]["unanswerable"]]
        if not answers:
            dropped.unanswerable += 1
            continue
        paragraphs = [
            e
            for answer in answers
            for e in answer["evidence"]
            if not e.startswith(FLOAT_PREFIX) and len(tokens(e)) >= SHINGLE  # skips headings
        ]
        if not paragraphs:
            dropped.no_usable_evidence += 1
            continue
        matches = []
        for paragraph in dict.fromkeys(paragraphs):  # de-duplicate, keep order
            dropped.evidence_paragraphs += 1
            found = match_evidence(paragraph, page_shingles)
            if found:
                dropped.evidence_paragraphs_matched += 1
                matches.append(found)
        if not matches:
            dropped.evidence_not_found += 1
            continue
        dropped.kept += 1
        questions.append(
            Question(
                paper=paper_id,
                question_id=qa["question_id"],
                text=qa["question"].strip(),
                gold_pages=frozenset().union(*(m.pages for m in matches)),
                evidence_shingles=frozenset().union(*(m.shingles for m in matches)),
            )
        )
    return questions


# --------------------------------------------------------------------------- #
# Metrics (pure functions)
# --------------------------------------------------------------------------- #
def first_gold_rank(pages: Sequence[int], gold: Iterable[int]) -> int | None:
    gold = set(gold)
    for rank, page in enumerate(pages, start=1):
        if page in gold:
            return rank
    return None


def is_evidence_chunk(chunk_text: str, evidence: frozenset[tuple[str, ...]]) -> bool:
    chunk = shingles(tokens(chunk_text))
    return bool(chunk) and len(chunk & evidence) / len(chunk) >= CHUNK_EVIDENCE_MIN


def question_scores(
    docs: Sequence[Any], gold_pages: frozenset[int], evidence: frozenset[tuple[str, ...]]
) -> dict[str, float]:
    """Per-question scores for one ranked list of chunks (best first, up to 10)."""
    page_rank = first_gold_rank([int(d.metadata.get("page", -1)) for d in docs], gold_pages)
    chunk_rank = next(
        (i for i, d in enumerate(docs, start=1) if is_evidence_chunk(d.page_content, evidence)),
        None,
    )
    scores = {"mrr10": 1 / page_rank if page_rank else 0.0}
    for k in KS:
        scores[f"page@{k}"] = float(page_rank is not None and page_rank <= k)
        scores[f"chunk@{k}"] = float(chunk_rank is not None and chunk_rank <= k)
    return scores


def bootstrap_ci(
    per_paper: dict[str, tuple[float, int]],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and 95% interval of a per-question metric, resampling whole papers.

    ``per_paper`` maps a paper to ``(sum of the metric over its questions, number of questions)``.
    """
    import numpy as np

    sums = np.array([s for s, _ in per_paper.values()], dtype=float)
    counts = np.array([n for _, n in per_paper.values()], dtype=float)
    if counts.sum() == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(rounds, len(sums)))
    boots = sums[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def summarize(
    records: list[dict[str, Any]], methods: Sequence[str], metrics: Sequence[str]
) -> dict[str, Any]:
    """Aggregate per-question records into means, CIs and paired differences vs vector."""

    def per_paper(method: str, metric: str, baseline: str | None = None):
        agg: dict[str, list[float]] = {}
        for rec in records:
            value = rec["scores"][method][metric]
            if baseline:
                value -= rec["scores"][baseline][metric]
            entry = agg.setdefault(rec["paper"], [0.0, 0])
            entry[0] += value
            entry[1] += 1
        return {paper: (s, n) for paper, (s, n) in agg.items()}

    out: dict[str, Any] = {"questions": len(records), "papers": len({r["paper"] for r in records})}
    for method in methods:
        out[method] = {
            m: dict(zip(("mean", "low", "high"), bootstrap_ci(per_paper(method, m)), strict=True))
            for m in metrics
        }
    if "vector" in methods:
        for method in methods:
            if method == "vector":
                continue
            out[f"{method}_minus_vector"] = {
                m: dict(
                    zip(
                        ("mean", "low", "high"),
                        bootstrap_ci(per_paper(method, m, "vector")),
                        strict=True,
                    )
                )
                for m in metrics
            }
    return out


# --------------------------------------------------------------------------- #
# Data and PDFs (network)
# --------------------------------------------------------------------------- #
def load_split(split: str) -> dict[str, Any]:
    """Download (once) and parse a Qasper split: ``{arxiv_id: paper}``."""
    archive, member = SPLITS[split]
    cached = CACHE_DIR / "qasper" / member
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(  # noqa: S310
            DATA_BASE + archive, headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            blob = response.read()
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"{member} not found in {archive}")
            cached.write_bytes(extracted.read())
    return json.loads(cached.read_text(encoding="utf-8"))


def select_papers(papers: dict[str, Any], limit: int | None, seed: int = 0) -> list[str]:
    ids = sorted(papers)
    if limit:
        ids = sorted(random.Random(seed).sample(ids, min(limit, len(ids))))  # noqa: S311
    return ids


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def pdf_path(arxiv_id: str, version: str) -> Path:
    return CACHE_DIR / "pdfs" / f"{arxiv_id}{version}.pdf"


def fetch_pdf(arxiv_id: str, version: str) -> Path | None:
    """Download one paper politely (>= 3 s between requests, retry on 429/503)."""
    target = pdf_path(arxiv_id, version)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://export.arxiv.org/pdf/{arxiv_id}{version}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        time.sleep(ARXIV_DELAY_S * (attempt + 1))
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                data = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                continue
            return None
        except (urllib.error.URLError, TimeoutError):
            continue
        if data[:5] == b"%PDF-":
            target.write_bytes(data)
            return target
        return None
    return None


def fetch_all(ids: Sequence[str], version: str) -> dict[str, Any]:
    manifest = load_manifest()
    for n, arxiv_id in enumerate(ids, start=1):
        key = f"{arxiv_id}{version}"
        if key in manifest and (
            manifest[key].get("failed") or pdf_path(arxiv_id, version).exists()
        ):
            continue
        path = fetch_pdf(arxiv_id, version)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path else None
        if digest and manifest.get(key, {}).get("sha256", digest) != digest:
            print(f"WARNING {key}: PDF differs from the pinned sha256", file=sys.stderr)
        manifest[key] = {"sha256": digest} if digest else {"failed": True}
        save_manifest(manifest)
        print(f"[{n}/{len(ids)}] {key}: {'ok' if path else 'FAILED'}", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# Running the benchmark
# --------------------------------------------------------------------------- #
def eval_settings(chunk_size: int) -> config.Settings:
    return config.Settings(
        chroma_dir=CACHE_DIR / f"chroma_{chunk_size}",
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,
        ocr=config.OCR_OFF,
        max_pdf_mb=100,
        max_pdf_pages=2000,
    )


def page_shingle_map(path: Path) -> dict[int, set[tuple[str, ...]]]:
    """Shingles of each page's text exactly as the app extracts it."""
    documents = pdf_engine.extract_documents(str(path), eval_settings(1000))
    return {int(d.metadata["page"]): shingles(tokens(d.page_content)) for d in documents}


def prepare_questions(
    split: str, ids: Sequence[str], version: str, papers: dict[str, Any]
) -> tuple[list[Question], Dropped]:
    dropped = Dropped()
    questions: list[Question] = []
    for arxiv_id in ids:
        path = pdf_path(arxiv_id, version)
        if not path.exists():
            dropped.papers_without_pdf += 1
            dropped.total += len(papers[arxiv_id]["qas"])
            continue
        try:
            page_shingles = page_shingle_map(path)
        except Exception as exc:
            print(f"{arxiv_id}: unreadable PDF ({exc})", file=sys.stderr)
            dropped.papers_without_pdf += 1
            dropped.total += len(papers[arxiv_id]["qas"])
            continue
        questions.extend(build_questions(arxiv_id, papers[arxiv_id], page_shingles, dropped))
    return questions, dropped


def ranked(db: Any, method: str, question: str, k: int) -> list[Any]:
    if method == "vector":
        return db.similarity_search(question, k=k)
    if method == "bm25":
        return retrieval._keyword_index(db).search(question, k)
    return retrieval.search(db, question, k, config.RETRIEVAL_HYBRID)


def score_method(db: Any, method: str, q: Question) -> dict[str, float]:
    """Scores for one question. Hybrid draws its candidates per k, so it is run once per k."""
    if method != "hybrid":
        docs = ranked(db, method, q.text, max(KS))
        return question_scores(docs, q.gold_pages, q.evidence_shingles)
    scores: dict[str, float] = {}
    for k in KS:
        at_k = question_scores(ranked(db, method, q.text, k), q.gold_pages, q.evidence_shingles)
        scores.update({name: value for name, value in at_k.items() if name.endswith(f"@{k}")})
    top = question_scores(ranked(db, method, q.text, max(KS)), q.gold_pages, q.evidence_shingles)
    scores["mrr10"] = top["mrr10"]
    return scores


def run_size(chunk_size: int, questions: Sequence[Question], version: str) -> list[dict[str, Any]]:
    """Score every question with every method for one chunk size."""
    settings = eval_settings(chunk_size)
    by_paper: dict[str, list[Question]] = {}
    for q in questions:
        by_paper.setdefault(q.paper, []).append(q)
    records = []
    for n, (paper, items) in enumerate(sorted(by_paper.items()), start=1):
        db = pdf_engine.index_pdf(
            str(pdf_path(paper, version)), config.BACKEND_LOCAL, None, settings
        )
        for q in items:
            scores = {method: score_method(db, method, q) for method in METHODS}
            records.append({"paper": paper, "question_id": q.question_id, "scores": scores})
        if n % 20 == 0:
            print(f"  size {chunk_size}: {n}/{len(by_paper)} papers", flush=True)
    return records


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run(args: argparse.Namespace) -> int:
    papers = load_split(args.split)
    ids = select_papers(papers, args.limit)
    questions, dropped = prepare_questions(args.split, ids, args.version, papers)
    print(f"{len(questions)} questions kept of {dropped.total} ({dropped})", flush=True)
    metrics = ["mrr10"] + [f"{kind}@{k}" for kind in ("page", "chunk") for k in KS]
    result: dict[str, Any] = {
        "split": args.split,
        "papers_requested": len(ids),
        "arxiv_version": args.version,
        "commit": git_sha(),
        "embedding_model": config.Settings().local_embedding_model,
        "dropped": dropped.__dict__,
        "sizes": {},
    }
    for size in args.chunk_sizes:
        records = run_size(size, questions, args.version)
        result["sizes"][str(size)] = summarize(records, METHODS, metrics)
        result["sizes"][str(size)]["chunk_size"] = size
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(format_report(result))
    return 0


def format_report(result: dict[str, Any]) -> str:
    lines = [
        f"Qasper {result['split']}: {result['papers_requested']} papers requested, "
        f"{result['dropped']['kept']} of {result['dropped']['total']} questions kept",
        "",
    ]
    for size, data in result["sizes"].items():
        lines.append(f"chunk size {size}, {data['questions']} questions, {data['papers']} papers")
        lines.append(f"{'':<22}" + "".join(f"{m:>22}" for m in METHODS))
        for metric in ("page@1", "page@3", "page@5", "page@10", "chunk@3", "mrr10"):
            cells = ""
            for method in METHODS:
                c = data[method][metric]
                cells += f"{c['mean']:>9.1%} [{c['low']:.1%}-{c['high']:.1%}]".rjust(22)
            lines.append(f"{metric:<22}{cells}")
        for method in ("hybrid", "bm25"):
            c = data[f"{method}_minus_vector"]["page@3"]
            lines.append(
                f"{method} - vector, page@3: {c['mean']:+.1%} [{c['low']:+.1%}, {c['high']:+.1%}]"
            )
        lines.append("")
    lines.append(f"Equal context budget (k x chunk size = {CONTEXT_BUDGET}), page hit:")
    for size, data in result["sizes"].items():
        k = CONTEXT_BUDGET // int(size)
        for method in METHODS:
            c = data[method][f"page@{k}"]
            lines.append(
                f"  chunk {size:>5} k={k}  {method:<7}"
                f" {c['mean']:.1%} [{c['low']:.1%}-{c['high']:.1%}]"
            )
    return "\n".join(lines)


def inspect(args: argparse.Namespace) -> int:
    """Print random question/evidence matches so the labels can be checked by eye."""
    papers = load_split(args.split)
    ids = [i for i in select_papers(papers, args.limit) if pdf_path(i, args.version).exists()]
    rng = random.Random(1)  # noqa: S311
    rng.shuffle(ids)
    shown = 0
    for arxiv_id in ids:
        page_shingles = page_shingle_map(pdf_path(arxiv_id, args.version))
        texts = {
            int(d.metadata["page"]): d.page_content
            for d in pdf_engine.extract_documents(
                str(pdf_path(arxiv_id, args.version)), eval_settings(1000)
            )
        }
        for qa in papers[arxiv_id]["qas"][:1]:
            for answer in qa["answers"][:1]:
                answer = answer["answer"]
                if answer["unanswerable"]:
                    continue
                for paragraph in answer["evidence"][:1]:
                    if paragraph.startswith(FLOAT_PREFIX) or not paragraph.strip():
                        continue
                    found = match_evidence(paragraph, page_shingles)
                    print(f"--- {arxiv_id}: {qa['question'].strip()}")
                    print(f"evidence: {paragraph[:200]!r}")
                    if found:
                        page = min(found.pages)
                        print(
                            f"matched pages {sorted(found.pages)}; page {page} starts: "
                            f"{texts[page][:120]!r}"
                        )
                    else:
                        print("NOT FOUND")
                    shown += 1
        if shown >= args.n:
            break
    return 0


def version_check(args: argparse.Namespace) -> int:
    """Compare how many evidence paragraphs are found in arXiv v1 vs the latest version."""
    papers = load_split(args.split)
    ids = select_papers(papers, args.limit)
    for version in ("v1", ""):
        fetch_all(ids, version)
        questions, dropped = prepare_questions(args.split, ids, version, papers)
        rate = dropped.evidence_paragraphs_matched / max(dropped.evidence_paragraphs, 1)
        print(
            f"version {version or 'latest'}: {dropped.evidence_paragraphs_matched}/"
            f"{dropped.evidence_paragraphs} evidence paragraphs found ({rate:.1%}), "
            f"{len(questions)} questions kept, {dropped.papers_without_pdf} papers missing"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "inspect", "run", "version-check"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--split", choices=sorted(SPLITS), default="validation")
        cmd.add_argument("--limit", type=int, help="random subset of this many papers")
        cmd.add_argument("--version", default="", help="arXiv version; '' (default) is the latest")
        if name == "inspect":
            cmd.add_argument("--n", type=int, default=20)
        if name == "run":
            cmd.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000, 1500])
            cmd.add_argument("--out", default="qasper_results.json")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_all(select_papers(load_split(args.split), args.limit), args.version)
        return 0
    return {"inspect": inspect, "run": run, "version-check": version_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
