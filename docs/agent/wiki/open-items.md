---
type: register
title: Open items
description: What the records leave open or contradict, each with its source, and what closed earlier items.
sources:
  - id: readme
    resource: ../../../README.md
    title: README
  - id: env-example
    resource: ../../../.env.example
    title: Example settings
  - id: benchmark
    resource: ../../benchmark.md
    title: Retrieval, answer-quality and scanned-page benchmarks
  - id: requirements
    resource: ../../../requirements.txt
    title: Runtime requirements
  - id: dependabot
    resource: ../../../.github/dependabot.yml
    title: Dependabot settings
  - id: ci
    resource: ../../../.github/workflows/ci.yml
    title: CI workflow
  - id: dockerfile
    resource: ../../../Dockerfile
    title: Dockerfile
  - id: open-prs
    resource: https://github.com/tc043/pdf_chatbot/pulls
    title: Open pull requests
  - id: pr12
    resource: https://github.com/tc043/pdf_chatbot/pull/12
    title: "PR #12: Ignore pillow>=12 in Dependabot until Gradio 6 migration"
  - id: pr19
    resource: https://github.com/tc043/pdf_chatbot/pull/19
    title: "PR #19: Address README limitations: hybrid retrieval, chat memory, index management, OCR"
  - id: pr26
    resource: https://github.com/tc043/pdf_chatbot/pull/26
    title: "PR #26: Report and retry unread OCR pages; index PDFs in parallel"
  - id: readme-before-19
    resource: "README.md at a371059, the main commit before PR #19"
  - id: readme-before-26
    resource: "README.md at a4e188d, the main commit before PR #26"
  - id: engine-before-26
    resource: "app/pdf_engine.py at a4e188d: EXTRACTION_VERSION and ocr_tag are the same as on 7989ccf"
  - id: gradio
    resource: "gradio 5.50.0: the /api/{api_name} route in gradio/routes.py, and the api_open argument of Blocks.queue in gradio/blocks.py"
  - id: api-check
    resource: "create_ui on 7989ccf with gradio 5.50.0, run on 2026-11-03: demo.api_open was True, and a POST to /gradio_api/api/toggle_settings answered 200 without joining the queue"
generated:
  by: "claude-code/2.1.317"
  at: "2026-11-03T18:50:00Z"
verified: []
status: draft
stale_after: "2026-12-03T00:00:00Z"
---

# Open items

> **This wiki is not the source of truth.** It is an agent's reading of the code, the tests, the README and the pull requests. If it disagrees with them, they win and this page is fixed.

Checked against `main` at `7989ccf`. OPEN and UNCLEAR items are open; an item is CLOSED only when a record closes it ([conventions](conventions.md)). This page records items and fixes none of them: a fix is its own change.

## Open

### O1. Indexes built before PR #26 can miss pages for good

Before PR #26, a scanned page that OCR failed on was left out of the index without a word.[^pr26] PR #26 kept the index names (`app/pdf_engine.py::EXTRACTION_VERSION` and `app/pdf_engine.py::ocr_tag` did not change[^engine-before-26]), and such an index has no incomplete marker, so `app/pdf_engine.py::build_index` reuses it as complete and never retries its pages. Deleting the index (the UI button or `python -m app.index_store`) makes the next question build it again.[^readme]

### O2. Tesseract reads one page at a time for the whole server

A large scanned PDF makes the OCR of every other PDF wait (README Limitations).[^readme] The lock is `app/pdf_engine.py::_TESSERACT_LOCK`; see [gotchas](gotchas.md) G1.

### O3. Indexing runs inside the first question

The first question about a PDF waits for its extraction, OCR and embedding (README Limitations).[^readme]

### O4. No per-user isolation

"There is no per-user isolation" (README Security and privacy), and the app is designed for one server process and a small number of users (README Limitations).[^readme] Indexes are keyed by the PDF's content, so users who upload the same PDF share its index, and the delete button removes it for all of them (`app/index_store.py::delete_for_pdf`).

### O5. The request queue does not cover direct API calls

Gradio's `api_open` is true by default outside Hugging Face Spaces, and then a `POST` to an event's REST route runs the event without joining the queue.[^gradio] `app/ui.py::create_ui` calls `demo.queue(...)` without `api_open=False`, and every event has an API name, among them `generate_response`, `handle_pdf_load` and `delete_stored_index`. A direct `POST` to `/gradio_api/api/toggle_settings` answered 200 outside the queue.[^api-check] So `PDFCHAT_MAX_CONCURRENT_REQUESTS` and `PDFCHAT_QUEUE_MAX_SIZE`, described as the "Gradio concurrency and queue bound" (README Configuration),[^readme] do not limit such calls. Not checked: how far a direct call gets into a streaming event such as `generate_response`. Authentication, when set, still applies to these routes.[^gradio]

### O6. Gradio is held at 5.x

The UI uses APIs that Gradio 6 removed (README Limitations),[^readme] so `requirements.txt` pins `gradio>=5.5,<6`,[^requirements] which also keeps Pillow below 12 ([gotchas](gotchas.md) G3). PR #24 proposes Gradio 6.[^open-prs]

### O7. The optional Docker images are not built in CI

The `docker` job builds the image with the default build arguments only.[^ci] Images built with the `WITH_OCR` or `WITH_LOCAL_MODEL` arguments of the Dockerfile[^dockerfile] are never built there, as README Development says.[^readme]

### O8. Five Dependabot pull requests wait

#21 to #25, opened on 2026-10-27.[^open-prs] What each would change is in [current state](current-state.md#waiting).

## Records that disagree

All OPEN. Each is fixed in the record that is wrong, in its own change.

### D1. `.env.example` calls remote OCR untested against a live server

`.env.example` says "Untested against a live server".[^env-example] docs/benchmark.md (Scanned pages (OCR)) reports runs against a live Unlimited-OCR server through [evals/modal_unlimited_ocr.py](../../../evals/modal_unlimited_ocr.py), with the results in [evals/results/qasper_scanned_remote.json](../../../evals/results/qasper_scanned_remote.json).[^benchmark]

### D2. The README says only retrieval is evaluated

README Limitations says the repository "only evaluates retrieval (see Development), not the quality of generated answers".[^readme] README Development reports an Answer-F1 comparison, and docs/benchmark.md has the sections "Answer quality (local model)" and "Answer quality on severe scans", both added by PR #20.[^benchmark]

### D3. The README says OCR needs Tesseract

README Limitations says scanned pages are read with OCR "only when Tesseract is installed".[^readme] With `PDFCHAT_OCR_ENGINE=remote` and `PDFCHAT_OCR_REMOTE_URL` set, `app/pdf_engine.py::ocr_available` is true without Tesseract, and the README's own OCR section describes that engine.

### D4. The README says the local model runs on CPU or GPU

The README's introduction says "a local MiniCPM5-1B-SFT model on your CPU/GPU".[^readme] `app/llm_engine.py::LocalModelManager.load_model` loads it with `device_map="cpu"` in float32.

## Unclear

### U1. Why Dependabot opened PR #24

`.github/dependabot.yml` has ignored major Gradio updates since `e86b14f`,[^dependabot] yet Dependabot opened #24 (`gradio>=6.30.0,<7`) on 2026-10-27.[^open-prs] No record explains it.

## Closed

### C1. Scanned pages that OCR failed on were dropped silently

Closed by PR #26 (`7989ccf`): the answer names the unread pages, and later questions retry them, up to `app/pdf_engine.py::MAX_INDEX_ATTEMPTS` builds.[^pr26] Tests: [tests/test_ocr.py](../../../tests/test_ocr.py).

### C2. One lock held every index build

Before PR #26, the README said "Index builds are serialised in one process".[^readme-before-26] Closed by PR #26 (`7989ccf`): extraction and embedding run outside the global lock, and only Tesseract and the table finder go one page at a time ([architecture](architecture.md#concurrency)).[^pr26]

### C3. The README limitations before PR #19

The README listed no conversation memory, no OCR, no hybrid keyword search, and indexes kept until the Chroma directory is deleted.[^readme-before-19] Closed by PR #19 (`73869de`): chat memory, Tesseract OCR, hybrid retrieval, and a delete button, a command-line tool and optional retention for stored indexes.[^pr19]

### C4. Dependabot proposed Pillow 12, which Gradio 5 cannot install

PR #4's bump reached `main` through #11 together with its revert, `9d75d7f` ([change index](change-index.md)). Closed by PR #12 (`30af754`): Dependabot ignores `pillow>=12`.[^pr12][^dependabot]

[^readme]: README
[^env-example]: Example settings
[^benchmark]: Retrieval, answer-quality and scanned-page benchmarks
[^requirements]: Runtime requirements
[^dependabot]: Dependabot settings
[^ci]: CI workflow
[^dockerfile]: Dockerfile
[^open-prs]: Open pull requests
[^pr12]: PR #12: Ignore pillow>=12 in Dependabot until Gradio 6 migration
[^pr19]: PR #19: Address README limitations: hybrid retrieval, chat memory, index management, OCR
[^pr26]: PR #26: Report and retry unread OCR pages; index PDFs in parallel
[^readme-before-19]: README.md at a371059, the main commit before PR #19
[^readme-before-26]: README.md at a4e188d, the main commit before PR #26
[^engine-before-26]: app/pdf_engine.py at a4e188d: EXTRACTION_VERSION and ocr_tag are the same as on 7989ccf
[^gradio]: gradio 5.50.0: the /api/{api_name} route in gradio/routes.py, and the api_open argument of Blocks.queue in gradio/blocks.py
[^api-check]: create_ui on 7989ccf with gradio 5.50.0, run on 2026-11-03: demo.api_open was True, and a POST to /gradio_api/api/toggle_settings answered 200 without joining the queue
