"""Scanned-pages benchmark: how much retrieval is lost when a paper has to be read with OCR?

Each Qasper paper is turned into an image-only PDF (every page rasterised, no text layer) and
indexed through the app's own OCR path. The questions and their gold pages come from the
ORIGINAL text PDF (see ``qasper_benchmark.py``), so the two conditions answer exactly the same
questions and the differences come from OCR quality alone:

  * ``text``      : the original PDF with its text layer;
  * ``tesseract`` : the image-only copy, read with Tesseract (the default OCR engine);
  * ``remote``    : the image-only copy, read through an OpenAI-compatible server
                    (``--ocr-engine remote --ocr-url ... --ocr-api-key ...``; needs a GPU server,
                    so it is not run in CI).

Besides page hit and chunk hit (as in the retrieval benchmark) it reports **OCR fidelity**: the
share of each text page's word 4-grams that the OCR text of the same page contains.

This experiment is descriptive. No default depends on it, and the chunk settings are the app's
defaults (500 characters, overlap 100, k=6).

Usage (needs requirements-local.txt, Tesseract for the default engine, and the PDFs from
``qasper_benchmark fetch --split test``)::

    python -m evals.qasper_scanned run --papers 60 --out evals/results/qasper_scanned.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymupdf

from app import config, pdf_engine
from app.exceptions import PDFValidationError
from evals import qasper_benchmark as qb

SCAN_DPI = 200
METHODS = ("vector", "hybrid")
METRICS = ("page@1", "page@3", "page@6", "chunk@6", "mrr10")


def rasterise(source: Path, target: Path, dpi: int = SCAN_DPI) -> Path:
    """Write an image-only copy of ``source`` (greyscale page images, no text layer)."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as original, pymupdf.open() as scan:
        for page in original:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            new_page = scan.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, pixmap=pixmap)
        scan.save(target, deflate=True, garbage=3)
    return target


def has_text_layer(path: Path) -> bool:
    """True when the app finds selectable text with OCR switched off."""
    off = dataclasses.replace(qb.eval_settings(500), ocr=config.OCR_OFF)
    try:
        pdf_engine.extract_documents(str(path), off)
    except PDFValidationError:
        return False
    return True


def ocr_settings(engine: str, url: str, model: str, api_key: str = "") -> config.Settings:
    """The app's default chunking, with OCR on and a page limit no benchmark paper reaches."""
    defaults = config.Settings()
    return dataclasses.replace(
        qb.eval_settings(defaults.chunk_size),
        chunk_overlap=defaults.chunk_overlap,
        chroma_dir=qb.CACHE_DIR / f"chroma_scanned_{engine}",
        ocr=config.OCR_AUTO,
        ocr_max_pages=1000,
        ocr_engine=engine,
        ocr_remote_url=url,
        ocr_remote_model=model,
        ocr_remote_api_key=api_key,
    )


def indexed_page_shingles(db: Any) -> dict[int, set[tuple[str, ...]]]:
    """Shingles of each page as indexed (the union over its chunks).

    Reading them back from the index means each scanned paper is OCR'd once. Chunk overlap keeps
    the shingles that straddle a chunk boundary, so the union is a close view of the page text.
    """
    data = db.get(include=["documents", "metadatas"])
    pages: dict[int, set[tuple[str, ...]]] = {}
    for text, meta in zip(data["documents"], data["metadatas"], strict=True):
        pages.setdefault(int(meta["page"]), set()).update(qb.shingles(qb.tokens(text)))
    return pages


def fidelity(text_pages: dict[int, set], ocr_pages: dict[int, set]) -> float | None:
    """Mean share of each text page's 4-grams found on the same OCR page."""
    shares = [
        len(grams & ocr_pages.get(page, set())) / len(grams)
        for page, grams in text_pages.items()
        if grams
    ]
    return sum(shares) / len(shares) if shares else None


def paired_difference(
    records: list[dict[str, Any]], arm: str, base: str, metric: str
) -> tuple[float, float, float]:
    per_paper: dict[str, list[float]] = {}
    for rec in records:
        entry = per_paper.setdefault(rec["paper"], [0.0, 0])
        entry[0] += rec["scores"][arm][metric] - rec["scores"][base][metric]
        entry[1] += 1
    return qb.bootstrap_ci({p: (s, n) for p, (s, n) in per_paper.items()})


def run(args: argparse.Namespace) -> int:
    papers = qb.load_split("test")
    ids = [
        i
        for i in qb.select_papers(papers, args.papers, seed=args.seed)
        if qb.pdf_path(i, "").exists()
    ]
    questions, dropped = qb.prepare_questions("test", ids, "", papers)
    print(f"{len(questions)} questions from {len(ids)} papers", flush=True)

    engine_settings = ocr_settings(args.ocr_engine, args.ocr_url, args.ocr_model, args.ocr_api_key)
    text_settings = qb.eval_settings(engine_settings.chunk_size)
    by_paper: dict[str, list[qb.Question]] = {}
    for q in questions:
        by_paper.setdefault(q.paper, []).append(q)

    records: list[dict[str, Any]] = []
    fidelities: list[float] = []
    for n, (paper, items) in enumerate(sorted(by_paper.items()), start=1):
        original = qb.pdf_path(paper, "")
        scan = rasterise(original, qb.CACHE_DIR / "scanned" / f"{paper}.pdf")
        if has_text_layer(scan):
            raise RuntimeError(f"{scan} still has a text layer; the scan copy is invalid")
        dbs = {
            "text": pdf_engine.index_pdf(str(original), config.BACKEND_LOCAL, None, text_settings),
            args.ocr_engine: pdf_engine.index_pdf(
                str(scan), config.BACKEND_LOCAL, None, engine_settings
            ),
        }
        score = fidelity(
            indexed_page_shingles(dbs["text"]), indexed_page_shingles(dbs[args.ocr_engine])
        )
        if score is not None:
            fidelities.append(score)
        for q in items:
            scores = {
                f"{condition}-{method}": qb.score_method(db, method, q)
                for condition, db in dbs.items()
                for method in METHODS
            }
            records.append({"paper": paper, "question_id": q.question_id, "scores": scores})
        if n % 10 == 0:
            print(f"  {n}/{len(by_paper)} papers", flush=True)

    arms = [f"{c}-{m}" for c in ("text", args.ocr_engine) for m in METHODS]
    result: dict[str, Any] = {
        "engine": args.ocr_engine,
        "ocr_model": args.ocr_model if args.ocr_engine == "remote" else "tesseract",
        "scan_dpi": SCAN_DPI,
        "commit": qb.git_sha(),
        "chunk_size": engine_settings.chunk_size,
        "papers": len(by_paper),
        "questions": len(records),
        "dropped": dropped.__dict__,
        "ocr_fidelity_mean": sum(fidelities) / len(fidelities) if fidelities else None,
        "scores": {
            arm: {
                m: dict(zip(("mean", "low", "high"), _arm_ci(records, arm, m), strict=True))
                for m in METRICS
            }
            for arm in arms
        },
        "ocr_minus_text": {
            f"{method}": {
                m: dict(
                    zip(
                        ("mean", "low", "high"),
                        paired_difference(
                            records, f"{args.ocr_engine}-{method}", f"text-{method}", m
                        ),
                        strict=True,
                    )
                )
                for m in METRICS
            }
            for method in METHODS
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(format_report(result))
    return 0


def _arm_ci(records: list[dict[str, Any]], arm: str, metric: str) -> tuple[float, float, float]:
    per_paper: dict[str, list[float]] = {}
    for rec in records:
        entry = per_paper.setdefault(rec["paper"], [0.0, 0])
        entry[0] += rec["scores"][arm][metric]
        entry[1] += 1
    return qb.bootstrap_ci({p: (s, n) for p, (s, n) in per_paper.items()})


def format_report(result: dict[str, Any]) -> str:
    engine = result["engine"]
    fidelity_text = (
        f"{result['ocr_fidelity_mean']:.1%}" if result["ocr_fidelity_mean"] is not None else "n/a"
    )
    lines = [
        f"Scanned pages, {engine} OCR at {result['scan_dpi']} dpi: "
        f"{result['questions']} questions, {result['papers']} papers, "
        f"chunk size {result['chunk_size']}",
        f"OCR fidelity (text-page 4-grams found in the indexed OCR of that page): {fidelity_text}",
        "",
        f"{'':<10}" + "".join(f"{a:>26}" for a in result["scores"]),
    ]
    for metric in METRICS:
        cells = ""
        for arm in result["scores"]:
            c = result["scores"][arm][metric]
            cells += f"{c['mean']:>9.1%} [{c['low']:.1%}-{c['high']:.1%}]".rjust(26)
        lines.append(f"{metric:<10}{cells}")
    lines.append("")
    for method, metrics in result["ocr_minus_text"].items():
        for metric in ("page@3", "page@6"):
            c = metrics[metric]
            interval = f"[{c['low']:+.1%}, {c['high']:+.1%}]"
            lines.append(f"{engine} - text, {method} {metric}: {c['mean']:+.1%} {interval}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("run")
    cmd.add_argument("--papers", type=int, default=60, help="random subset of test papers")
    cmd.add_argument("--seed", type=int, default=2)
    cmd.add_argument(
        "--ocr-engine",
        choices=[config.OCR_ENGINE_TESSERACT, config.OCR_ENGINE_REMOTE],
        default=config.OCR_ENGINE_TESSERACT,
    )
    cmd.add_argument("--ocr-url", default="")
    cmd.add_argument("--ocr-model", default="baidu/Unlimited-OCR")
    cmd.add_argument(
        "--ocr-api-key",
        default=os.environ.get(config.ENV_PREFIX + "OCR_REMOTE_API_KEY", ""),
        help="bearer token for the OCR server (default: $PDFCHAT_OCR_REMOTE_API_KEY)",
    )
    cmd.add_argument("--out", default="evals/results/qasper_scanned.json")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
