---
type: index
title: Change index
description: One line per pull request, #1 to #26, with its date, topic and whether it reached main.
sources:
  - id: prs
    resource: https://github.com/tc043/pdf_chatbot/pulls?q=is%3Apr
    title: All pull requests, with their merged_at and closed_at times
  - id: main
    resource: "git log --first-parent of main up to 7989ccf"
  - id: ancestry
    resource: "git merge-base --is-ancestor: the heads of #2, #3, #4, #6, #9 and #10, and commit 9d75d7f, are in 9aad64c and not in 9d39153"
  - id: pr7
    resource: https://github.com/tc043/pdf_chatbot/pull/7
    title: "PR #7, closed by Dependabot: Superseded by #17"
  - id: pr8
    resource: https://github.com/tc043/pdf_chatbot/pull/8
    title: "PR #8, closed by Dependabot: Superseded by #13"
  - id: pr13
    resource: https://github.com/tc043/pdf_chatbot/pull/13
    title: "PR #13, closed without merging"
generated:
  by: "claude-code/2.1.317"
  at: "2026-11-03T18:50:00Z"
verified: []
status: draft
stale_after: "2026-12-03T00:00:00Z"
---

# Change index

> **This wiki is not the source of truth.** It is an agent's reading of the code, the tests, the README and the pull requests. If it disagrees with them, they win and this page is fixed.

Dates are UTC, from GitHub's `merged_at` or `closed_at`.[^prs] "Merged" and "Squashed" name the commit on `main` that brought the change in.[^main] Before PR #1, `main` had five commits, from `0c42d23` (2025-04-04) to `7f6d8e3` (2026-06-15).[^main]

GitHub shows #2, #3, #4, #6, #9 and #10 as merged because #11 merged their commits; none of them has a merge commit of its own on `main`.[^ancestry]

| PR | Closed | Topic | Reached main |
|---|---|---|---|
| #1 | 2026-10-19 | Production hardening: settings and error handling, the test suite, CI, Docker and a new README | Merged, `9d39153` |
| #2 | 2026-10-20 | `actions/setup-python` 5 to 7 (Dependabot) | Through #11 |
| #3 | 2026-10-20 | `pytest-cov>=7.1.0` (Dependabot) | Through #11 |
| #4 | 2026-10-20 | `pillow>=12.3.0,<13` (Dependabot) | Through #11, which also reverts it (`9d75d7f`): Gradio 5 needs Pillow below 12 |
| #5 | 2026-10-20 | `ruff>=0.17.1` (Dependabot) | Merged, `3e66bf6` |
| #6 | 2026-10-20 | `chromadb>=1.5.9,<2` (Dependabot) | Through #11 |
| #7 | 2026-10-20 | `transformers>=5.20.0` (Dependabot) | No: superseded by #17[^pr7] |
| #8 | 2026-10-20 | Docker base image `python:3.14-slim` (Dependabot) | No: superseded by #13[^pr8] |
| #9 | 2026-10-20 | `actions/checkout` 4 to 7 (Dependabot) | Through #11 |
| #10 | 2026-10-20 | Per-PDF OKF v0.2 Markdown bundles, tables kept as Markdown | Through #11 |
| #11 | 2026-10-20 | Merges #10 and the bumps #2, #3, #4, #6 and #9; reverts #4; puts the extraction format in the index name (`e422a69`) | Merged, `9aad64c` |
| #12 | 2026-10-20 | Dependabot ignores `pillow>=12` until the Gradio 6 migration | Merged, `30af754` |
| #13 | 2026-10-20 | Docker base image `python:3.15-slim` (Dependabot) | No: closed without merging; no reason recorded[^pr13] |
| #14 | 2026-10-20 | `langchain-core>=1.6.12,<2` (Dependabot) | Merged, `e6fe699` |
| #15 | 2026-10-20 | `torch>=2.14.1` (Dependabot) | Merged, `a371059` |
| #16 | 2026-10-20 | `langchain-text-splitters>=1.1.3,<2` (Dependabot) | Merged, `439e7cf` |
| #17 | 2026-10-20 | `transformers>=5.20.1` (Dependabot) | Merged, `4ad0630` |
| #18 | 2026-10-20 | CI installs `requirements-local.txt` and runs the `local_model` tests | Merged, `0d3f1c5` |
| #19 | 2026-10-20 | Hybrid retrieval, chat memory, stored-index management and Tesseract OCR | Merged, `73869de` |
| #20 | 2026-10-21 | Qasper retrieval and answer benchmarks, the 500/`k=6` chunking default, the optional remote OCR engine and the scanned-page benchmarks | Squashed, `a4e188d` |
| #21 | open | `sentence-transformers>=6.2.0` (Dependabot) | Not yet |
| #22 | open | `pymupdf>=2.0,<3` (Dependabot) | Not yet |
| #23 | open | `ruff>=0.17.2` (Dependabot) | Not yet |
| #24 | open | `gradio>=6.30.0,<7` (Dependabot) | Not yet |
| #25 | open | `pillow>=11.3.0,<13` (Dependabot) | Not yet |
| #26 | 2026-11-03 | Unread OCR pages are reported and retried; PDFs are indexed in parallel | Squashed, `7989ccf` |

What the open ones would change is in [current state](current-state.md#waiting).

[^prs]: All pull requests, with their merged_at and closed_at times
[^main]: git log --first-parent of main up to 7989ccf
[^ancestry]: git merge-base --is-ancestor: the heads of #2, #3, #4, #6, #9 and #10, and commit 9d75d7f, are in 9aad64c and not in 9d39153
[^pr7]: PR #7, closed by Dependabot: Superseded by #17
[^pr8]: PR #8, closed by Dependabot: Superseded by #13
[^pr13]: PR #13, closed without merging
