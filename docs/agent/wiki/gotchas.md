---
type: register
title: Gotchas
description: Known traps in the code and tests, each with an anchor that shows it.
sources:
  - id: pdf-engine
    resource: ../../../app/pdf_engine.py
    title: PDF extraction, OCR and indexing
  - id: requirements
    resource: ../../../requirements.txt
    title: Runtime requirements
  - id: dependabot
    resource: ../../../.github/dependabot.yml
    title: Dependabot settings
  - id: readme
    resource: ../../../README.md
    title: README
  - id: revert
    resource: "commit 9d75d7f: Revert pillow>=12.3.0 bump: gradio 5.x requires pillow<12"
  - id: conftest
    resource: ../../../tests/conftest.py
    title: Shared test fixtures
  - id: pyproject
    resource: ../../../pyproject.toml
    title: Test and lint settings
  - id: fake-patch
    resource: "DeterministicFakeEmbedding with langchain-core 1.6.12, run on 2026-11-03: setting embed_documents on an instance raises ValueError"
  - id: retrieval
    resource: ../../../app/retrieval.py
    title: BM25 and rank fusion
  - id: llm-engine
    resource: ../../../app/llm_engine.py
    title: Prompt and generation
  - id: c369e8f
    resource: "commit c369e8f: Address review: keep model reserved until a stalled worker exits; reject nan/inf timeouts"
  - id: run
    resource: ../../../run.py
    title: Entry point
  - id: gradio-docs
    resource: "gradio 5.50.0, gradio/blocks.py: the show_api arguments of Blocks.launch and of event listeners"
  - id: ocr-remote
    resource: ../../../app/ocr_remote.py
    title: Remote OCR client
  - id: benchmark
    resource: ../../benchmark.md
    title: Retrieval, answer-quality and scanned-page benchmarks
generated:
  by: "claude-code/2.1.317"
  at: "2026-11-03T18:50:00Z"
verified: []
status: draft
stale_after: "2026-12-03T00:00:00Z"
---

# Gotchas

> **This wiki is not the source of truth.** It is an agent's reading of the code, the tests, the README and the pull requests. If it disagrees with them, they win and this page is fixed.

Checked against `main` at `7989ccf`. Each trap names the code that shows it; read that code before relying on the description here.

## G1. Tesseract cannot run in two threads at once

Two Tesseract runs at the same time fail with "Attempt to use Leptonica from 2 threads at once!", even for different PDFs (the comment above the lock).[^pdf-engine] `app/pdf_engine.py::_ocr_pages` reads each page under `app/pdf_engine.py::_TESSERACT_LOCK`; keep every Tesseract call inside it. Remote OCR does not take this lock.

## G2. The table finder is serialised too

`app/pdf_engine.py::_extract_page_markdown` calls PyMuPDF's `find_tables` under `app/pdf_engine.py::_TABLES_LOCK`, because older PyMuPDF releases keep the table finder's state in module globals.[^pdf-engine] `requirements.txt` allows `pymupdf>=1.24.3,<2`,[^requirements] and PR #22 proposes 2.x: check both locks against it before merging.

## G3. Gradio 5 keeps Pillow below 12

Every Gradio 5.x release requires `pillow<12`,[^dependabot] and Gradio 6 removed APIs the UI uses.[^readme] `requirements.txt` pins `gradio>=5.5,<6` and allows `pillow>=10,<13`,[^requirements] so pip installs a Pillow below 12. A floor of Pillow 12 cannot be installed: PR #4 tried it and was reverted.[^revert] Gradio 6 and Pillow 12 move together.

## G4. A stored index is reused without reading the PDF again

Only the collection name decides reuse (`app/pdf_engine.py::get_collection_name`): the PDF's content hash, the backend, the embedding model, the chunk size and overlap, `app/pdf_engine.py::EXTRACTION_VERSION` and `app/pdf_engine.py::ocr_tag`.[^pdf-engine] A change to how pages are extracted, chunked or labelled is invisible to the stored indexes unless one of these changes, so bump `EXTRACTION_VERSION` in the same change. Tests: `tests/test_pdf_engine.py::test_collection_name_changes_with_extraction_version` and `tests/test_pdf_engine.py::test_collection_name_changes_with_every_index_input`.

## G5. The tests are hermetic on purpose

`tests/conftest.py::isolated_environment` runs for every test: it removes every `PDFCHAT_*` variable and `OPENAI_API_KEY`, and clears the cached settings, local embeddings, OCR text and local model. `tests/conftest.py::hermetic_ocr` makes Tesseract look missing unless a test is marked `ocr`.[^conftest] A test that needs a setting passes it in (the `settings` fixture) rather than reading the environment. Tests that need torch or Tesseract are marked `local_model` or `ocr`, and `--strict-markers` rejects any other marker.[^pyproject]

## G6. Patch the fake embeddings on the class

`DeterministicFakeEmbedding` is a pydantic model, so setting a method on an instance raises `ValueError`.[^fake-patch] Tests patch the class with `monkeypatch`, which undoes it after the test, as `tests/test_ocr.py::test_retries_that_read_nothing_new_keep_the_index_and_stop_at_the_limit` does.

## G7. The BM25 cache key covers the metadata

`app/retrieval.py::_keyword_index` caches one BM25 index per collection content, keyed by a hash of every chunk's text and metadata.[^retrieval] Two PDFs can share every chunk's text and still differ in page numbers (an extra cover page); a key of the text alone would give one of them the other's pages. Test: `tests/test_retrieval.py::test_keyword_cache_does_not_mix_up_pdfs_with_identical_text_but_different_pages`. The cache saves building the index, not reading the chunks: every hybrid search reads all chunks of the collection from Chroma.

## G8. The worker thread releases the local model

`app/llm_engine.py::generate_local_stream` takes the model's generation lock and hands its release to the worker thread (`on_finished` in `app/llm_engine.py::_stream_in_thread`), so the model stays reserved until generation has really stopped, even after the client has gone.[^llm-engine] Releasing it when the consumer stops would let a second request use the model while a stalled worker still runs.[^c369e8f]

## G9. Hiding the API page does not close the API

`run.py::main` launches with `show_api=False`,[^run] which only hides the API documentation; the events still answer API clients.[^gradio-docs] What protects a server that others can reach is authentication (`PDFCHAT_AUTH_USER` and `PDFCHAT_AUTH_PASSWORD`, or an authenticating proxy) and `PDFCHAT_USE_SERVER_KEY=false`, so that visitors bring their own OpenAI key. `run.py::main` warns about both when it listens on an address other than loopback. Direct API calls also skip the request queue ([open items](open-items.md) O5).

## G10. The remote OCR request is tuned to one model

`app/ocr_remote.py` sends each page as a 200 dpi PNG with the literal prompt `app/ocr_remote.py::PROMPT` (the model has no chat template, so the text must start with `<image>`) and the server-side settings in `app/ocr_remote.py::SERVER_PARAMS`. `app/ocr_remote.py::clean_output` strips the layout markers from the reply and drops figure blocks.[^ocr-remote] The scanned-page benchmarks used the same prompt and settings,[^benchmark] so changing them makes those numbers out of date.

[^pdf-engine]: PDF extraction, OCR and indexing
[^requirements]: Runtime requirements
[^dependabot]: Dependabot settings
[^readme]: README
[^revert]: commit 9d75d7f: Revert pillow>=12.3.0 bump: gradio 5.x requires pillow<12
[^conftest]: Shared test fixtures
[^pyproject]: Test and lint settings
[^fake-patch]: DeterministicFakeEmbedding with langchain-core 1.6.12, run on 2026-11-03: setting embed_documents on an instance raises ValueError
[^retrieval]: BM25 and rank fusion
[^llm-engine]: Prompt and generation
[^c369e8f]: commit c369e8f: Address review: keep model reserved until a stalled worker exits; reject nan/inf timeouts
[^run]: Entry point
[^gradio-docs]: gradio 5.50.0, gradio/blocks.py: the show_api arguments of Blocks.launch and of event listeners
[^ocr-remote]: Remote OCR client
[^benchmark]: Retrieval, answer-quality and scanned-page benchmarks
