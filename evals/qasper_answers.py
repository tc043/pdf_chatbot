"""Answer-quality benchmark on Qasper: does the retrieval setting change the answers?

The retrieval benchmark (``qasper_benchmark.py``) showed that 500-character chunks with k=6
find the evidence page more often than 1000-character chunks with k=3 at the same amount of
text. That is not the same as better answers, so this script generates answers with the app's
own local model and prompt and scores them against Qasper's human answers.

Arms (all use hybrid retrieval and the app's real prompt; the same questions in every arm):
  * ``default``: 1000-character chunks, k=3 (the current default);
  * ``small``  : 500-character chunks, k=6 (the candidate);
  * ``oracle`` : the annotators' evidence paragraphs as context (at most the same 3000
    characters). It shows how much of the remaining error is retrieval and how much is the model.

Scoring needs no judge model: for yes/no questions the leading "yes"/"no" of the answer is
compared with the gold label; for the others it is Qasper's token-level Answer-F1 (SQuAD-style
normalisation), maximised over annotators. Page citations such as "(p. 3)", which the app's
prompt asks for, are removed first. The mean answer length and the share of gold-answer tokens
that appear in the answer (recall) are reported next to F1, because F1 punishes long answers.

Decision rule (written before any answers were generated):
  * ``small`` replaces ``default`` only if the paired mean difference in the answer score
    (small - default, resampling papers, 95% interval) has a LOWER bound above -0.02 (it is
    not meaningfully worse) AND the answers are not longer on average by more than 25%.
  * If the lower bound is above 0 the change is a clear improvement; otherwise the default
    stays and the retrieval gain is reported as not carrying over to answers.
  * The result holds for the local 1B model only. The OpenAI backend, which a different
    default would also affect, was not tested (no API key was available).

Usage (needs requirements-local.txt and the indexes from the retrieval benchmark)::

    python -m evals.qasper_answers generate --papers 40 --out evals/results/answers.jsonl
    python -m evals.qasper_answers score --answers evals/results/answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app import config, llm_engine, pdf_engine, retrieval
from evals import qasper_benchmark as qb

MAX_NEW_TOKENS = 200
ARMS = {
    "default": {"chunk_size": 1000, "k": 3},
    "small": {"chunk_size": 500, "k": 6},
    "oracle": {"chunk_size": 1000, "k": 3},  # context comes from the evidence, not retrieval
}
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_CITATION = re.compile(r"\(\s*pp?\.\s*\d+(?:\s*[,–-]\s*\d+)*\s*\)", re.IGNORECASE)
_MARGIN = -0.02


# --------------------------------------------------------------------------- #
# Scoring (pure functions)
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """SQuAD-style normalisation: lower case, no punctuation, no articles, single spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, truth: str) -> float:
    pred, gold = normalize(prediction).split(), normalize(truth).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, truth: str) -> float:
    pred, gold = normalize(prediction).split(), normalize(truth).split()
    if not gold:
        return 0.0
    return sum((Counter(pred) & Counter(gold)).values()) / len(gold)


def clean_answer(raw: str) -> str:
    """Drop reasoning blocks and page citations before scoring."""
    return " ".join(_CITATION.sub("", _THINK.sub("", raw)).split())


def leading_yes_no(answer: str) -> str | None:
    words = normalize(answer).split()
    return words[0] if words and words[0] in {"yes", "no"} else None


def gold_answers(qa: dict[str, Any]) -> list[tuple[str, str]]:
    """``[(kind, text)]`` per answerable annotation; kind: boolean, extractive or abstractive."""
    out = []
    for entry in qa["answers"]:
        answer = entry["answer"]
        if answer["unanswerable"]:
            continue
        if answer["yes_no"] is not None:
            out.append(("boolean", "Yes" if answer["yes_no"] else "No"))
        elif answer["extractive_spans"]:
            out.append(("extractive", ", ".join(answer["extractive_spans"])))
        elif answer["free_form_answer"].strip():
            out.append(("abstractive", answer["free_form_answer"].strip()))
    return out


def score_answer(raw_answer: str, golds: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Best score over annotators: accuracy for yes/no gold, token F1 otherwise."""
    answer = clean_answer(raw_answer)
    best = {"kind": golds[0][0], "score": 0.0, "recall": 0.0}
    for kind, text in golds:
        if kind == "boolean":
            score = float(leading_yes_no(answer) == text.lower())
            recall = score
        else:
            score, recall = token_f1(answer, text), token_recall(answer, text)
        if score > best["score"] or (score == best["score"] and recall > best["recall"]):
            best = {"kind": kind, "score": score, "recall": recall}
    best["words"] = len(normalize(answer).split())
    return best


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def evidence_documents(qa: dict[str, Any], gold_pages: Iterable[int]) -> list[Document]:
    """The annotators' evidence paragraphs as context, within the same 3000-character budget."""
    pages = sorted(gold_pages)
    docs, used = [], 0
    for answer in (a["answer"] for a in qa["answers"] if not a["answer"]["unanswerable"]):
        for paragraph in answer["evidence"]:
            paragraph = paragraph.strip()
            if not paragraph or paragraph.startswith(qb.FLOAT_PREFIX):
                continue
            if any(paragraph == d.page_content for d in docs) or used >= qb.CONTEXT_BUDGET:
                continue
            paragraph = paragraph[: qb.CONTEXT_BUDGET - used]
            docs.append(Document(page_content=paragraph, metadata={"page": pages[0]}))
            used += len(paragraph)
    return docs


def generate_answer(docs: Sequence[Document], question: str, settings: config.Settings) -> str:
    prompt = llm_engine.format_rag_prompt(question, docs)
    return "".join(
        llm_engine.generate_local_stream(
            prompt,
            enable_thinking=False,
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
            settings=settings,
        )
    )


def finished(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {(row["question_id"], row["arm"]) for row in rows}


def generate(args: argparse.Namespace) -> int:
    papers = qb.load_split("test")
    ids = [
        i
        for i in qb.select_papers(papers, args.papers, seed=args.seed)
        if qb.pdf_path(i, "").exists()
    ]
    # Random order, so stopping early still leaves a random sample of papers.
    random.Random(args.seed).shuffle(ids)  # noqa: S311
    questions, dropped = qb.prepare_questions("test", ids, "", papers)
    print(f"{len(questions)} questions from {len(ids)} papers ({dropped.kept} kept)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = finished(out)
    qa_by_id = {
        qa["question_id"]: qa for paper in ids for qa in papers[paper]["qas"]
    }  # for evidence text
    local = config.Settings(local_model=config.Settings().local_model)
    dbs: dict[tuple[str, int], Any] = {}
    current_paper = ""
    with out.open("a", encoding="utf-8") as handle:
        for n, q in enumerate(questions, start=1):
            for arm in args.arms:
                if (q.question_id, arm) in done:
                    continue
                spec = ARMS[arm]
                started = time.time()
                if arm == "oracle":
                    docs = evidence_documents(qa_by_id[q.question_id], q.gold_pages)
                else:
                    if q.paper != current_paper:  # questions are grouped by paper
                        dbs.clear()
                        current_paper = q.paper
                    key = (q.paper, spec["chunk_size"])
                    if key not in dbs:
                        dbs[key] = pdf_engine.index_pdf(
                            str(qb.pdf_path(q.paper, "")),
                            config.BACKEND_LOCAL,
                            None,
                            qb.eval_settings(spec["chunk_size"]),
                        )
                    docs = retrieval.search(dbs[key], q.text, spec["k"], config.RETRIEVAL_HYBRID)
                answer = generate_answer(docs, q.text, local)
                row = {
                    "paper": q.paper,
                    "question_id": q.question_id,
                    "arm": arm,
                    "pages": [d.metadata.get("page") for d in docs],
                    "context_chars": sum(len(d.page_content) for d in docs),
                    "answer": answer,
                    "seconds": round(time.time() - started, 1),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(questions)} questions", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def load_scores(answers_path: Path) -> list[dict[str, Any]]:
    """Per-question records ``{paper, question_id, arms: {arm: scores}}`` for complete questions."""
    papers = qb.load_split("test")
    qas = {qa["question_id"]: (pid, qa) for pid, p in papers.items() for qa in p["qas"]}
    by_question: dict[str, dict[str, Any]] = {}
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        paper, qa = qas[row["question_id"]]
        golds = gold_answers(qa)
        if not golds:
            continue
        record = by_question.setdefault(
            row["question_id"], {"paper": paper, "question_id": row["question_id"], "arms": {}}
        )
        record["arms"][row["arm"]] = score_answer(row["answer"], golds) | {
            "seconds": row["seconds"],
            "context_chars": row["context_chars"],
        }
    arms = {a for r in by_question.values() for a in r["arms"]}
    return [r for r in by_question.values() if set(r["arms"]) == arms]


def paired(records: list[dict[str, Any]], arm: str, base: str, field: str = "score"):
    agg: dict[str, list[float]] = {}
    for rec in records:
        entry = agg.setdefault(rec["paper"], [0.0, 0])
        entry[0] += rec["arms"][arm][field] - rec["arms"][base][field]
        entry[1] += 1
    return qb.bootstrap_ci({p: (s, n) for p, (s, n) in agg.items()})


def mean(records: list[dict[str, Any]], arm: str, field: str) -> float:
    return sum(r["arms"][arm][field] for r in records) / max(len(records), 1)


def verdict(records: list[dict[str, Any]]) -> str:
    if not {"default", "small"} <= set(records[0]["arms"]):
        return "no verdict (needs the default and small arms)"
    delta, low, high = paired(records, "small", "default")
    longer = mean(records, "small", "words") / max(mean(records, "default", "words"), 1e-9)
    if low > 0 and longer <= 1.25:
        return f"small is a clear improvement ({delta:+.3f} [{low:+.3f}, {high:+.3f}])"
    if low > _MARGIN and longer <= 1.25:
        return f"small is not meaningfully worse ({delta:+.3f} [{low:+.3f}, {high:+.3f}]): switch"
    return f"keep the default ({delta:+.3f} [{low:+.3f}, {high:+.3f}], answers x{longer:.2f} long)"


def report(records: list[dict[str, Any]]) -> str:
    arms = list(records[0]["arms"])
    lines = [f"{len(records)} questions from {len({r['paper'] for r in records})} papers", ""]
    lines.append(f"{'':<12}" + "".join(f"{a:>12}" for a in arms))
    for label, field in (("score", "score"), ("recall", "recall"), ("words", "words")):
        lines.append(f"{label:<12}" + "".join(f"{mean(records, a, field):>12.3f}" for a in arms))
    lines.append(
        f"{'sec/answer':<12}" + "".join(f"{mean(records, a, 'seconds'):>12.1f}" for a in arms)
    )
    lines.append("")
    for arm in arms:
        if arm == "default" or "default" not in arms:
            continue
        d, lo, hi = paired(records, arm, "default")
        lines.append(f"{arm} - default, score: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines.append("")
    kinds = sorted({r["arms"][arms[0]]["kind"] for r in records})
    for kind in kinds:
        part = [r for r in records if r["arms"][arms[0]]["kind"] == kind]
        cells = "  ".join(f"{a} {mean(part, a, 'score'):.3f}" for a in arms)
        lines.append(f"{kind:<12} n={len(part):<4} {cells}")
    lines += ["", "Verdict: " + verdict(records)]
    return "\n".join(lines)


def score(args: argparse.Namespace) -> int:
    records = load_scores(Path(args.answers))
    if not records:
        print("no complete questions yet")
        return 1
    print(report(records))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--papers", type=int, default=40, help="random subset of test papers")
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=["default", "small"])
    gen.add_argument("--out", default="evals/results/answers.jsonl")
    sc = sub.add_parser("score")
    sc.add_argument("--answers", default="evals/results/answers.jsonl")
    args = parser.parse_args(argv)
    return {"generate": generate, "score": score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
