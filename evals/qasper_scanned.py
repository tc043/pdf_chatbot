"""Scanned-pages benchmark: how much retrieval is lost when a paper has to be read with OCR?

Each Qasper paper is turned into an image-only PDF (every page rasterised, no text layer) and
indexed through the app's own OCR path. The questions and their gold pages come from the
ORIGINAL text PDF (see ``qasper_benchmark.py``), so the conditions answer exactly the same
questions and the differences come from OCR quality alone:

  * ``text``      : the original PDF with its text layer;
  * ``tesseract`` : the image-only copy, read with Tesseract (the default OCR engine);
  * ``remote``    : the image-only copy, read through an OpenAI-compatible server
                    (``--ocr-url ... --ocr-api-key ...``; needs a GPU server, so it is not run
                    in CI).

Several engines can be given in one run (``--ocr-engine tesseract remote``) so that the
engines are also compared with each other, question by question.

Besides page hit and chunk hit (as in the retrieval benchmark) it reports **OCR fidelity**: the
share of each text page's word 4-grams that the OCR text of the same page contains, and the
number of pages for which an engine produced no indexed text at all.

Degraded scans (``--degrade moderate|severe``). The clean 200 dpi rasterisation is the easy
case; both engines were within noise of the text layer on it. The degraded levels are
SYNTHETIC "bad scan" copies, not real scans. Each page is re-rendered at a lower resolution,
rotated by a small random angle, given Gaussian noise and JPEG-compressed (``LEVELS``). The
randomness is seeded per paper and page, so every engine and every rerun sees the same images.
Levels, written down before any degraded result was produced (they are only changed after
a pilot on the validation split, which is stated in docs/benchmark.md if it happens):

  * moderate: 150 dpi, rotation 0.3-0.8 degrees, noise sigma 6, JPEG quality 60;
  * severe  : 100 dpi, rotation 1.0-2.0 degrees, noise sigma 14, JPEG quality 35.

Reading rule: a level shows a difference between two conditions only when the paired 95%
interval (resampling papers) of their page-hit difference excludes zero; otherwise it is
reported as not separable. Page hit (hybrid and vector, top 3 and top 6) is the primary measure.

This experiment is descriptive. No default depends on it, and the chunk settings are the app's
defaults (500 characters, overlap 100, k=6).

Usage (needs requirements-local.txt, Tesseract for the default engine, and the PDFs from
``qasper_benchmark fetch --split test``)::

    python -m evals.qasper_scanned run --papers 100 --ocr-engine tesseract \\
        --degrade moderate --out evals/results/qasper_scanned_moderate.json
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image

from app import config, pdf_engine
from app.exceptions import PDFValidationError
from evals import qasper_benchmark as qb

SCAN_DPI = 200
REMOTE_TIMEOUT_S = 300.0
METHODS = ("vector", "hybrid")
METRICS = ("page@1", "page@3", "page@6", "chunk@6", "mrr10")


@dataclasses.dataclass(frozen=True)
class Level:
    dpi: int
    rotation_deg: tuple[float, float]  # magnitude range; the sign is random
    noise_sigma: float
    jpeg_quality: int


LEVELS = {
    "moderate": Level(dpi=150, rotation_deg=(0.3, 0.8), noise_sigma=6.0, jpeg_quality=60),
    "severe": Level(dpi=100, rotation_deg=(1.0, 2.0), noise_sigma=14.0, jpeg_quality=35),
}
DEGRADE_CHOICES = ["none", *LEVELS]


def degrade_page(page: pymupdf.Page, level: Level, seed: int) -> bytes:
    """The page as a degraded greyscale JPEG: low resolution, slight skew, noise, compression."""
    rng = np.random.default_rng(seed)
    pixmap = page.get_pixmap(dpi=level.dpi, colorspace=pymupdf.csGRAY)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    low, high = level.rotation_deg
    angle = float(rng.uniform(low, high)) * (1 if rng.random() < 0.5 else -1)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    noisy = np.asarray(image, dtype=np.float64) + rng.normal(
        0.0, level.noise_sigma, image.size[::-1]
    )
    image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=level.jpeg_quality)
    return buffer.getvalue()


def rasterise(
    source: Path, target: Path, dpi: int = SCAN_DPI, degrade: str = "none", seed_key: str = ""
) -> Path:
    """Write an image-only copy of ``source`` (greyscale page images, no text layer).

    ``degrade`` names a level in ``LEVELS``; pages are then degraded with a seed that depends
    only on ``seed_key`` (the paper id) and the page number, never on the process.
    """
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    level = LEVELS.get(degrade)
    with pymupdf.open(source) as original, pymupdf.open() as scan:
        for number, page in enumerate(original):
            new_page = scan.new_page(width=page.rect.width, height=page.rect.height)
            if level is None:
                pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
                new_page.insert_image(new_page.rect, pixmap=pixmap)
            else:
                seed = zlib.crc32(f"{seed_key}:{number}".encode())
                new_page.insert_image(new_page.rect, stream=degrade_page(page, level, seed))
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


def ocr_settings(
    engine: str, url: str, model: str, api_key: str = "", degrade: str = "none"
) -> config.Settings:
    """The app's default chunking, with OCR on and a page limit no benchmark paper reaches."""
    defaults = config.Settings()
    return dataclasses.replace(
        qb.eval_settings(defaults.chunk_size),
        chunk_overlap=defaults.chunk_overlap,
        chroma_dir=qb.CACHE_DIR / f"chroma_scanned_{engine}" / degrade,
        ocr=config.OCR_AUTO,
        ocr_max_pages=1000,
        ocr_engine=engine,
        ocr_remote_url=url,
        ocr_remote_model=model,
        ocr_remote_api_key=api_key,
        # Degraded pages can take longer to read; a timeout would silently blank a page.
        ocr_remote_timeout_s=REMOTE_TIMEOUT_S,
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


def _mean_ci(records: list[dict[str, Any]], arm: str, metric: str) -> dict[str, float]:
    return dict(zip(("mean", "low", "high"), _arm_ci(records, arm, metric), strict=True))


def _difference(records: list[dict[str, Any]], arm: str, base: str) -> dict[str, Any]:
    return {
        method: {
            m: dict(
                zip(
                    ("mean", "low", "high"),
                    paired_difference(records, f"{arm}-{method}", f"{base}-{method}", m),
                    strict=True,
                )
            )
            for m in METRICS
        }
        for method in METHODS
    }


def run(args: argparse.Namespace) -> int:
    papers = qb.load_split(args.split)
    ids = [
        i
        for i in qb.select_papers(papers, args.papers, seed=args.seed)
        if qb.pdf_path(i, "").exists()
    ]
    questions, dropped = qb.prepare_questions(args.split, ids, "", papers)
    print(f"{len(questions)} questions from {len(ids)} papers", flush=True)

    engines = list(dict.fromkeys(args.ocr_engine))
    settings = {
        engine: ocr_settings(engine, args.ocr_url, args.ocr_model, args.ocr_api_key, args.degrade)
        for engine in engines
    }
    text_settings = qb.eval_settings(next(iter(settings.values())).chunk_size)
    by_paper: dict[str, list[qb.Question]] = {}
    for q in questions:
        by_paper.setdefault(q.paper, []).append(q)

    scan_dir = "scanned" if args.degrade == "none" else f"scanned_{args.degrade}"
    records: list[dict[str, Any]] = []
    fidelities: dict[str, list[float]] = {engine: [] for engine in engines}
    pages = dict.fromkeys(engines, 0)
    blank = dict.fromkeys(engines, 0)
    for n, (paper, items) in enumerate(sorted(by_paper.items()), start=1):
        original = qb.pdf_path(paper, "")
        scan = rasterise(
            original, qb.CACHE_DIR / scan_dir / f"{paper}.pdf", degrade=args.degrade, seed_key=paper
        )
        if has_text_layer(scan):
            raise RuntimeError(f"{scan} still has a text layer; the scan copy is invalid")
        dbs = {
            "text": pdf_engine.index_pdf(str(original), config.BACKEND_LOCAL, None, text_settings)
        }
        text_pages = indexed_page_shingles(dbs["text"])
        with pymupdf.open(scan) as pdf:
            scan_pages = len(pdf)
        for engine in engines:
            dbs[engine] = pdf_engine.index_pdf(
                str(scan), config.BACKEND_LOCAL, None, settings[engine]
            )
            read = indexed_page_shingles(dbs[engine])
            pages[engine] += scan_pages
            blank[engine] += scan_pages - len(read)
            score = fidelity(text_pages, read)
            if score is not None:
                fidelities[engine].append(score)
        for q in items:
            scores = {
                f"{condition}-{method}": qb.score_method(db, method, q)
                for condition, db in dbs.items()
                for method in METHODS
            }
            records.append({"paper": paper, "question_id": q.question_id, "scores": scores})
        if n % 10 == 0:
            print(f"  {n}/{len(by_paper)} papers", flush=True)

    conditions = ["text", *engines]
    arms = [f"{c}-{m}" for c in conditions for m in METHODS]
    pairs = [(e, "text") for e in engines] + [
        (b, a) for i, a in enumerate(engines) for b in engines[i + 1 :]
    ]
    level = LEVELS.get(args.degrade)
    result: dict[str, Any] = {
        "engines": engines,
        "degrade": args.degrade,
        "degrade_params": dataclasses.asdict(level) if level else None,
        "ocr_model": args.ocr_model if config.OCR_ENGINE_REMOTE in engines else None,
        "scan_dpi": SCAN_DPI if level is None else level.dpi,
        "split": args.split,
        "commit": qb.git_sha(),
        "chunk_size": next(iter(settings.values())).chunk_size,
        "papers": len(by_paper),
        "questions": len(records),
        "dropped": dropped.__dict__,
        "ocr_fidelity_mean": {e: (sum(f) / len(f) if f else None) for e, f in fidelities.items()},
        "pages": pages,
        "pages_without_indexed_text": blank,
        "scores": {arm: {m: _mean_ci(records, arm, m) for m in METRICS} for arm in arms},
        "differences": {f"{a} - {b}": _difference(records, a, b) for a, b in pairs},
        "records": records,
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
    def pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    lines = [
        f"Scanned pages, degrade={result['degrade']}, {result['scan_dpi']} dpi: "
        f"{result['questions']} questions, {result['papers']} papers, "
        f"chunk size {result['chunk_size']}",
    ]
    for engine, value in result["ocr_fidelity_mean"].items():
        blank = result.get("pages_without_indexed_text", {}).get(engine)
        total = result.get("pages", {}).get(engine)
        extra = f", {blank}/{total} pages without indexed text" if total else ""
        lines.append(f"OCR fidelity, {engine}: {pct(value)}{extra}")
    lines += ["", f"{'':<10}" + "".join(f"{a:>26}" for a in result["scores"])]
    for metric in METRICS:
        cells = ""
        for arm in result["scores"]:
            c = result["scores"][arm][metric]
            cells += f"{c['mean']:>9.1%} [{c['low']:.1%}-{c['high']:.1%}]".rjust(26)
        lines.append(f"{metric:<10}{cells}")
    lines.append("")
    for pair, methods in result["differences"].items():
        for method, metrics in methods.items():
            for metric in ("page@3", "page@6"):
                c = metrics[metric]
                interval = f"[{c['low']:+.1%}, {c['high']:+.1%}]"
                verdict = "" if c["low"] > 0 or c["high"] < 0 else "  (includes zero)"
                lines.append(f"{pair}, {method} {metric}: {c['mean']:+.1%} {interval}{verdict}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("run")
    cmd.add_argument("--papers", type=int, default=60, help="random subset of test papers")
    cmd.add_argument("--seed", type=int, default=2)
    cmd.add_argument("--split", choices=["test", "validation"], default="test")
    cmd.add_argument(
        "--ocr-engine",
        nargs="+",
        choices=[config.OCR_ENGINE_TESSERACT, config.OCR_ENGINE_REMOTE],
        default=[config.OCR_ENGINE_TESSERACT],
        help="one or more engines; with several, they are also compared with each other",
    )
    cmd.add_argument("--degrade", choices=DEGRADE_CHOICES, default="none")
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
