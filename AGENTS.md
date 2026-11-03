# AGENTS.md

Instructions for coding agents in this repository. The [README](README.md) is the source of truth for setup, settings and behaviour; this file points into it.

## Commands

```bash
pip install -r requirements-dev.txt     # the app, the OpenAI backend, pytest and ruff
ruff check . && ruff format --check .
pytest --cov                            # offline and fast; CI fails below 85% coverage
pytest -m local_model                   # after pip install -r requirements-local.txt
pytest -m ocr                           # needs the Tesseract program with English data
python run.py                           # serves http://127.0.0.1:7860
```

## Layout

- `app/`: the application. `app/chat.py` answers one question, `app/pdf_engine.py` extracts, reads (OCR) and indexes PDFs, and `app/ui.py` is the Gradio interface.
- `tests/`: the pytest suite. `tests/conftest.py` isolates every test from the environment.
- `evals/`: offline benchmarks, not run in CI. Method and results are in [docs/benchmark.md](docs/benchmark.md).
- `docs/agent/wiki/`: the agent wiki, below.

## Rules

- Keep the tests offline: embeddings and chat models are replaced by fakes (README Development). Mark a test that needs torch and transformers `local_model`, and one that needs Tesseract `ocr` (`pyproject.toml`).
- Run ruff and the tests before every commit. CI runs both on Python 3.10 to 3.13 (`.github/workflows/ci.yml`).
- Never commit keys or local data. `.env`, `.chromadb/` and `.eval_cache/` are git-ignored, and settings come from `PDFCHAT_*` environment variables (README Configuration).
- Keep Gradio below 6: the UI uses APIs that Gradio 6 removed (README Limitations).
- When you change how pages are extracted, chunked or labelled, bump `EXTRACTION_VERSION` in `app/pdf_engine.py`, or stored indexes keep the old text ([gotchas](docs/agent/wiki/gotchas.md) G4).

## Agent wiki

[docs/agent/wiki/index.md](docs/agent/wiki/index.md) is a reading aid kept by agents, in the Open Knowledge Format (OKF v0.2): the current state, the architecture, every pull request, open items and gotchas. It is not the source of truth. Before relying on a page, check its `stale_after` and the commit it was checked against; the code, the tests and the README win. `python -m tests.test_agent_wiki` lists the stale pages, and `tests/test_agent_wiki.py` checks the wiki in CI. How to refresh it is in [conventions](docs/agent/wiki/conventions.md).
