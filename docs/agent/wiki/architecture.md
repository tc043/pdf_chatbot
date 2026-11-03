---
type: architecture
title: Architecture
description: How one question flows through the modules, what is stored where, and which locks keep indexing safe.
sources:
  - id: readme
    resource: ../../../README.md
    title: README (How it works, Managing stored indexes)
  - id: app
    resource: ../../../app
    title: The application package
  - id: run
    resource: ../../../run.py
    title: Entry point
  - id: tests
    resource: ../../../tests
    title: The test suite
  - id: pr26
    resource: https://github.com/tc043/pdf_chatbot/pull/26
    title: "PR #26: Report and retry unread OCR pages; index PDFs in parallel"
  - id: pr10
    resource: https://github.com/tc043/pdf_chatbot/pull/10
    title: "PR #10: extract PDFs into OKF v0.2 Markdown bundles with table preservation"
generated:
  by: "claude-code/2.1.317"
  at: "2026-11-03T18:50:00Z"
verified: []
status: draft
stale_after: "2026-12-03T00:00:00Z"
---

# Architecture

> **This wiki is not the source of truth.** It is an agent's reading of the code, the tests, the README and the pull requests. If it disagrees with them, they win and this page is fixed.

Checked against `main` at `7989ccf`. Names are code anchors (`path::name`); read the code before relying on them. The README has the same flow as a diagram.[^readme]

## One question, end to end

1. **Upload.** `handle_pdf_load`, inside `app/ui.py::create_ui`, runs `app/pdf_engine.py::validate_pdf` (the file exists and is a PDF, not password-protected, within the size and page limits) and shows page 1 with `app/pdf_engine.py::render_pdf_page`. A new or removed PDF clears the chat. Nothing is indexed yet.
2. **Ask.** `generate_response`, also inside `create_ui`, turns the chat into earlier question and answer pairs with `app/chat.py::history_pairs`, then streams `app/chat.py::run_turn`.
3. **Check.** `app/chat.py::_run_turn` rejects an empty or too long question, a missing PDF and an unknown backend. For OpenAI, `app/chat.py::resolve_openai_key` takes the key typed in the UI, or else the server's key if `PDFCHAT_USE_SERVER_KEY` allows it.
4. **Index**, on the first question about a PDF: `app/pdf_engine.py::build_index` reuses the stored index or builds one (next section).
5. **Retention.** If a limit is set, `app/index_store.py::auto_prune` runs. It never removes the PDF in use, and a failure is only logged.
6. **Retrieve.** `app/retrieval.py::search`: in `hybrid` mode, vector search and BM25 merged by reciprocal rank fusion. A follow-up question is also searched together with the previous question, and the two rankings are fused (`app/chat.py::_retrieve`).
7. **Prompt.** `app/llm_engine.py::format_rag_prompt` puts the rules, the excerpts (each labelled with its page and OKF concept) and the earlier turns in delimited blocks, and strips look-alikes of those tags from the text.
8. **Generate.** `app/llm_engine.py::generate_openai_stream` or `app/llm_engine.py::generate_local_stream` streams the answer. The viewer jumps to the page of the best chunk.
9. **Note.** If OCR could not read some scanned pages, `app/chat.py::unread_pages_notice` ends the answer with a note naming them. `history_pairs` leaves the note out of later prompts.

Errors meant for users are subclasses of `app/exceptions.py::PDFChatbotError`; `app/exceptions.py::wrap_provider_error` turns provider failures (bad key, rate limit, timeout) into them. Any other exception is logged with a short reference code, and the user sees only the code (`app/chat.py::run_turn`).

## Building an index

`app/pdf_engine.py::build_index`, in order:[^pr26]

1. Validate the PDF again and hash its content with SHA-256 (`app/pdf_engine.py::file_fingerprint`). The collection name (`app/pdf_engine.py::get_collection_name`) is the first 40 hex characters of the hash, the backend kind, and a hash of the embedding model, the chunk size and overlap, `app/pdf_engine.py::EXTRACTION_VERSION` and the OCR setup (`app/pdf_engine.py::ocr_tag`).
2. Take the build lock of that collection, so a second request for the same PDF waits for the first build instead of repeating it.
3. Reuse the collection if it has chunks and either no incomplete marker or no attempts left (`app/pdf_engine.py::MAX_INDEX_ATTEMPTS`, 3).
4. Otherwise extract the pages (`app/pdf_engine.py::extract_pages`): text with tables as Markdown (`app/pdf_engine.py::_extract_page_markdown`). Pages that have an image but no text go to OCR if it is available (`app/pdf_engine.py::ocr_available`). A page that OCR failed on is unread; a page that OCR read and found empty is not.
5. Split the pages into chunks and embed them, outside the global lock.
6. Under the global lock: write the OKF bundle (`app/pdf_engine.py::build_okf_bundle`), add the chunks with their precomputed vectors (`app/pdf_engine.py::_PrecomputedEmbeddings`), and reopen the collection for a fresh handle.
7. Write the incomplete marker, or remove it when every page was read (`app/pdf_engine.py::_write_marker`). While retries are pending, the OCR text already read stays in memory, so a retry sends only the unread pages to OCR.

A retry that reads no more pages than before keeps the index and counts the attempt. A first build that fails removes the empty collection it created (`app/pdf_engine.py::_discard_collection`).

## Modules

- [app/config.py](../../../app/config.py): reads and validates the `PDFCHAT_*` settings; an invalid value stops the app at startup. `app/config.py::Settings`, `app/config.py::Settings.from_env`, `app/config.py::get_settings`. Tests: [test_config.py](../../../tests/test_config.py).
- [app/exceptions.py](../../../app/exceptions.py): errors whose messages are safe to show to users. `app/exceptions.py::PDFChatbotError`, `app/exceptions.py::wrap_provider_error`.
- [app/logging_config.py](../../../app/logging_config.py): logging that redacts `sk-` keys. `app/logging_config.py::RedactSecretsFilter`, `app/logging_config.py::setup_logging`. Tests: `tests/test_config.py::test_log_filter_redacts_api_keys`.
- [app/pdf_engine.py](../../../app/pdf_engine.py): validation, page rendering, extraction with Markdown tables, OCR, chunking, Chroma indexing and the per-PDF OKF bundle. Tests: [test_pdf_engine.py](../../../tests/test_pdf_engine.py), [test_ocr.py](../../../tests/test_ocr.py).
- [app/ocr_remote.py](../../../app/ocr_remote.py): sends scanned pages, as PNG images, to an OpenAI-compatible vision server. `app/ocr_remote.py::recognize_pages`, `app/ocr_remote.py::clean_output`. Tests: [test_ocr_remote.py](../../../tests/test_ocr_remote.py).
- [app/retrieval.py](../../../app/retrieval.py): BM25 over the stored chunks, and reciprocal rank fusion. `app/retrieval.py::search`, `app/retrieval.py::BM25Index`, `app/retrieval.py::reciprocal_rank_fusion`. Tests: [test_retrieval.py](../../../tests/test_retrieval.py).
- [app/llm_engine.py](../../../app/llm_engine.py): the prompt, OpenAI streaming and the local model. `app/llm_engine.py::format_rag_prompt`, `app/llm_engine.py::LocalModelManager`. Tests: [test_llm_engine.py](../../../tests/test_llm_engine.py), [test_local_model_integration.py](../../../tests/test_local_model_integration.py).
- [app/chat.py](../../../app/chat.py): one question and answer, with no Gradio in it. Tests: [test_chat.py](../../../tests/test_chat.py).
- [app/index_store.py](../../../app/index_store.py): lists, deletes and prunes stored indexes; also a command-line tool (`python -m app.index_store`). `app/index_store.py::list_indexes`, `app/index_store.py::delete_for_pdf`, `app/index_store.py::prune`, `app/index_store.py::main`. Tests: [test_index_store.py](../../../tests/test_index_store.py).
- [app/ui.py](../../../app/ui.py): the Gradio interface: upload and page viewer, settings, and the button that deletes stored indexes. Questions go to `app/chat.py::run_turn`; the UI holds no answering logic. Tests: [test_ui.py](../../../tests/test_ui.py).
- [run.py](../../../run.py): starts the server, warns about unsafe settings and turns SIGTERM into a clean stop. `run.py::main`. Tests: [test_run.py](../../../tests/test_run.py).
- [evals/](../../../evals): offline evaluations, not run in CI; method and results are in [docs/benchmark.md](../../benchmark.md). Tests: [test_retrieval_eval.py](../../../tests/test_retrieval_eval.py) and the three `tests/test_qasper_*.py` files.

## Stored state

Everything that persists is under `PDFCHAT_CHROMA_DIR`: `./.chromadb` by default, `/data/chroma` in the Docker image.[^readme]

| Path | What it holds | Written by | Removed by |
|---|---|---|---|
| Chroma collection `pdf_<40 hex>_<openai or local>_<8 hex>` | The chunks and vectors of one PDF for one embedding setup | `build_index` | `app/index_store.py::_delete`, behind the UI button, the CLI and retention |
| `bundles/pdf_<40 hex>/` | The PDF's OKF bundle. Its modification time is the index's "last used" time (`app/pdf_engine.py::_mark_used`) | `build_index` | `_delete`, once no collection of that PDF is left |
| `incomplete/<collection>.json` | The unread pages and the build attempts of an incomplete index (`app/pdf_engine.py::incomplete_marker`) | `_write_marker` | `_write_marker` when nothing is unread; `_delete` |

Held in memory by the process only: OCR text while retries are pending (`app/pdf_engine.py::_ocr_cache`, at most 500 pages), the BM25 indexes of at most four collections (`app/retrieval.py::_keyword_index`), the local model, at most two local embedding models (`app/pdf_engine.py::_local_embeddings`), and the settings.

## Concurrency

The app is one process, and every lock below exists only in that process (README Limitations).[^readme]

| Lock or limit | What it serialises | Where |
|---|---|---|
| Gradio queue | `PDFCHAT_MAX_CONCURRENT_REQUESTS` (2) requests at a time, at most `PDFCHAT_QUEUE_MAX_SIZE` (20) waiting | `app/ui.py::create_ui` |
| Global index lock | The short Chroma and bundle reads and writes; deleting an index takes it too | `app/pdf_engine.py::_INDEX_LOCK`, `app/index_store.py::_delete` |
| Build lock | One build per collection; other requests for it wait | `app/pdf_engine.py::_build_lock` |
| Tesseract lock | One Tesseract page at a time in the whole server | `app/pdf_engine.py::_TESSERACT_LOCK` |
| Table lock | One table-finder call at a time | `app/pdf_engine.py::_TABLES_LOCK` |
| Remote OCR | At most four pages in flight for each PDF being read | `app/ocr_remote.py::REMOTE_CONCURRENCY` |
| Local generation | One generation at a time; a waiting request fails with a "busy" error after `PDFCHAT_LOCAL_TOKEN_TIMEOUT_S` | `app/llm_engine.py::LocalModelManager`, `app/llm_engine.py::generate_local_stream` |

Different PDFs are extracted and embedded in parallel; only the Tesseract and table steps go one page at a time.[^pr26]

## The OKF bundles the app writes

These are not this wiki. Since PR #10, every indexed PDF also gets its own OKF v0.2 bundle from `app/pdf_engine.py::build_okf_bundle`:[^pr10]

- `index.md` with `okf_version: "0.2"`, `document.md` (type `PDF Document`), `pages/index.md`, and one `pages/page_NNNN.md` (type `PDF Page`) for each page with text, tables kept as Markdown.
- Each page's frontmatter has `generated.by: pdf_chatbot/pymupdf`, a `sources` entry for the PDF, and `page`, `table_count` and `sha256`.
- Each chunk carries `okf_concept` and `okf_path` metadata, and the prompt labels each excerpt with its concept (`app/llm_engine.py::format_rag_prompt`).
- `tests/test_pdf_engine.py::test_index_pdf_writes_conformant_okf_v02_bundle` checks the bundle.

[^readme]: README (How it works, Managing stored indexes)
[^pr26]: PR #26: Report and retry unread OCR pages; index PDFs in parallel
[^pr10]: PR #10: extract PDFs into OKF v0.2 Markdown bundles with table preservation
