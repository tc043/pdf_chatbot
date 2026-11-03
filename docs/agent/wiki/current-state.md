---
type: status
title: Current state
description: What is on main, how it is checked and what is waiting, each with its source.
sources:
  - id: main
    resource: "git history of main up to 7989ccf"
  - id: readme
    resource: ../../../README.md
    title: README
  - id: settings
    resource: ../../../app/config.py
    title: Settings and their defaults
  - id: benchmark
    resource: ../../benchmark.md
    title: Retrieval, answer-quality and scanned-page benchmarks
  - id: ci
    resource: ../../../.github/workflows/ci.yml
    title: CI workflow
  - id: ci-run
    resource: https://github.com/tc043/pdf_chatbot/actions/runs/41690752692
    title: CI run 99 on main at 7989ccf
  - id: pr26
    resource: https://github.com/tc043/pdf_chatbot/pull/26
    title: "PR #26: Report and retry unread OCR pages; index PDFs in parallel"
  - id: pr20
    resource: https://github.com/tc043/pdf_chatbot/pull/20
    title: "PR #20: Measure Unlimited-OCR on clean and degraded scans, and on answer quality"
  - id: open-prs
    resource: https://github.com/tc043/pdf_chatbot/pulls
    title: Open pull requests
  - id: local-run
    resource: "pytest --cov on 7989ccf with Python 3.13, torch and Tesseract installed, run on 2026-11-03"
generated:
  by: "claude-code/2.1.317"
  at: "2026-11-03T18:50:00Z"
verified: []
status: draft
stale_after: "2026-12-03T00:00:00Z"
---

# Current state

> **This wiki is not the source of truth.** It is an agent's reading of the code, the tests, the README and the pull requests. If it disagrees with them, they win and this page is fixed.

Checked on 2026-11-03 against `main` at `7989ccf`.

## What the app is

- A Gradio app: upload a PDF and ask questions about it. The answer is written from retrieved, page-labelled excerpts and cites page numbers, and the page viewer jumps to the best match.[^readme]
- Two backends, chosen in the UI: OpenAI (`gpt-4o-mini`, embeddings `text-embedding-3-small`), or a local `openbmb/MiniCPM5-1B-SFT` with `sentence-transformers/all-MiniLM-L6-v2` embeddings (`app/config.py::Settings`).[^settings]
- One server process for a small number of users, with no per-user isolation: every user shares the stored indexes.[^readme]

## Defaults worth knowing

All settings are in the README's Configuration table.[^readme] These change behaviour the most:

| Setting | Default | Why, or since when |
|---|---|---|
| Chunking and retrieval | 500 characters with overlap 100, `k=6`, hybrid (vector and BM25) | Since PR #20. On the Qasper test split, 500/`k=6` finds the evidence page in 86.6% of questions against 73.6% for 1000/`k=3`, at the same amount of text.[^benchmark] |
| Chat memory | 3 earlier turns; `0` turns it off | PR #19 |
| OCR | `auto`: Tesseract reads pages that have only an image, when it is installed. A PDF with more than 50 such pages is rejected | PR #19. The remote engine (`PDFCHAT_OCR_ENGINE=remote`) is optional and experimental (PR #20). |
| Retention | off: nothing is deleted unless asked | PR #19 |
| Listen address | `127.0.0.1`. The Docker image listens on `0.0.0.0` and sets `PDFCHAT_USE_SERVER_KEY=false` | PR #1; [Dockerfile](../../../Dockerfile) |

## Latest changes

- **PR #26** (merged 2026-11-03, `7989ccf`): a scanned page that OCR failed on is no longer dropped silently. The answer names the unread pages, later questions retry them (at most 3 attempts), and the retry state is a small file under `PDFCHAT_CHROMA_DIR/incomplete/`. Indexing no longer holds the global lock while it extracts and embeds; Tesseract and the table finder still work one page at a time.[^pr26] The locks are described in [architecture](architecture.md).
- **PR #20** (merged 2026-10-21, `a4e188d`): the Qasper retrieval and answer-quality benchmarks ([docs/benchmark.md](../../benchmark.md)) and, from them, the new chunking default (500/100 with `k=6`, was 1000/200 with `k=3`); the optional remote OCR engine (`app/ocr_remote.py`), a Modal script that serves `baidu/Unlimited-OCR`, and the scanned-page benchmarks.[^pr20]
- Every pull request is in the [change index](change-index.md).

## How main is checked

CI ([ci.yml](../../../.github/workflows/ci.yml)) runs on every push to `main` and on every pull request.[^ci] All of its jobs passed on `7989ccf` (run 99).[^ci-run]

| Job | What it runs |
|---|---|
| `test` (Python 3.10, 3.11, 3.12, 3.13) | `ruff check`, `ruff format --check`, then `pytest --cov --cov-fail-under=85`. The `local_model` and `ocr` tests skip themselves here, because neither torch nor Tesseract is installed. |
| `local-model` | Installs CPU PyTorch and `requirements-local.txt`, then `pytest -m local_model`. |
| `ocr` | Installs Tesseract with English data, then `pytest -m ocr`. |
| `docker` | Builds the image with the default build arguments (no OCR, no local model) and checks that it serves HTTP. |

A local run of the whole suite on `7989ccf` with Python 3.13, torch and Tesseract installed: 283 passed, 95% coverage.[^local-run] CI's jobs skip some of these tests, so their counts differ.

## Waiting

Five Dependabot pull requests, all opened on 2026-10-27; none is merged.[^open-prs]

| PR | Bump | Note |
|---|---|---|
| #21 | `sentence-transformers>=6.2.0` (`requirements-local.txt`) | The local embeddings. The `local-model` job installs and imports it. |
| #22 | `pymupdf>=2.0,<3` | A new major version. Check the Tesseract and table-finder locks against it ([gotchas](gotchas.md)). |
| #23 | `ruff>=0.17.2` | Lint only. |
| #24 | `gradio>=6.30,<7` | The UI uses APIs that Gradio 6 removed (README Limitations). `.github/dependabot.yml` ignores Gradio major updates, yet this was opened ([open items](open-items.md)). |
| #25 | `pillow>=11.3.0,<13` | Raises the lower bound only. Gradio 5 still keeps Pillow below 12. |

What else is open is in [open items](open-items.md).

[^readme]: README
[^settings]: Settings and their defaults
[^benchmark]: Retrieval, answer-quality and scanned-page benchmarks
[^ci]: CI workflow
[^ci-run]: CI run 99 on main at 7989ccf
[^pr26]: PR #26: Report and retry unread OCR pages; index PDFs in parallel
[^pr20]: PR #20: Measure Unlimited-OCR on clean and degraded scans, and on answer quality
[^open-prs]: Open pull requests
[^local-run]: pytest --cov on 7989ccf with Python 3.13, torch and Tesseract installed
