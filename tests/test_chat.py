import os

import pytest
from langchain_core.documents import Document

from app import chat, config, llm_engine, pdf_engine
from app.exceptions import GenerationError, IndexingError

DOCS = [
    Document(page_content="alpha", metadata={"page": 2}),
    Document(page_content="beta", metadata={"page": 0}),
]


@pytest.fixture
def pipeline(monkeypatch):
    """Replace indexing, retrieval and generation with recorders so only chat.py runs."""
    state = {"index_keys": [], "docs": list(DOCS), "pieces": ["Hello ", "world"]}

    def index_pdf(path, backend, key, settings):
        state["index_keys"].append(key)
        return "db"

    def retrieve(db, question, k, mode=None):
        state["k"] = k
        state["mode"] = mode
        return state["docs"]

    def openai_stream(prompt, key, temperature, max_tokens, settings=None):
        state["prompt"] = prompt
        yield from state["pieces"]

    monkeypatch.setattr(pdf_engine, "index_pdf", index_pdf)
    monkeypatch.setattr(pdf_engine, "retrieve_context", retrieve)
    monkeypatch.setattr(llm_engine, "generate_openai_stream", openai_stream)
    return state


def run(
    settings, question="What is alpha?", pdf="/x.pdf", backend=config.BACKEND_OPENAI, key="sk-u"
):
    return list(chat.run_turn(question, pdf, backend, key, 0.3, 256, True, settings))


@pytest.fixture
def settings(tmp_path):
    return config.Settings(chroma_dir=tmp_path, max_question_chars=50)


# ------------------------------------------------------------------ happy path


def test_happy_path_streams_progress_then_answer(pipeline, settings):
    updates = run(settings)
    contents = [u.content for u in updates]
    assert "Indexing" in contents[0] and "Searching" in contents[1]
    assert contents[-2:] == ["Hello ", "Hello world"]
    assert not any(u.is_error for u in updates)
    # The viewer is pointed at the best chunk's page (0-based) once retrieval is done.
    assert [u.page for u in updates][:2] == [None, None]
    assert all(u.page == 2 for u in updates[2:])
    assert pipeline["k"] == settings.retriever_k
    assert pipeline["mode"] == settings.retrieval_mode
    assert "alpha" in pipeline["prompt"] and "What is alpha?" in pipeline["prompt"]


# ------------------------------------------------------------ input validation


@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question_is_rejected(pipeline, settings, question):
    (update,) = run(settings, question=question)
    assert update.is_error and "question" in update.content


def test_overlong_question_is_rejected(pipeline, settings):
    (update,) = run(settings, question="x" * 51)
    assert update.is_error and "limit" in update.content
    assert pipeline["index_keys"] == []


def test_missing_pdf_is_rejected(pipeline, settings):
    (update,) = run(settings, pdf="")
    assert update.is_error and "upload" in update.content.lower()


def test_unknown_backend_is_rejected(pipeline, settings):
    (update,) = run(settings, backend="Mystery")
    assert update.is_error


# ------------------------------------------------------------------- API keys


def test_openai_without_any_key_asks_for_one(pipeline, settings):
    (update,) = run(settings, key="  ")
    assert update.is_error and "API key" in update.content
    assert pipeline["index_keys"] == []


def test_users_own_key_wins_over_server_key(pipeline, tmp_path):
    s = config.Settings(chroma_dir=tmp_path, server_openai_api_key="sk-server")
    run(s, key=" sk-user ")
    assert pipeline["index_keys"] == ["sk-user"]


def test_server_key_is_a_fallback_only_when_enabled(pipeline, tmp_path):
    enabled = config.Settings(chroma_dir=tmp_path, server_openai_api_key="sk-server")
    run(enabled, key="")
    assert pipeline["index_keys"] == ["sk-server"]

    disabled = config.Settings(
        chroma_dir=tmp_path, server_openai_api_key="sk-server", use_server_key=False
    )
    (update,) = run(disabled, key="")
    assert update.is_error


def test_user_keys_are_never_written_to_process_environment(pipeline, tmp_path):
    """Regression: the old UI did os.environ['OPENAI_API_KEY'] = <user's key>, which
    leaked one visitor's key to every other visitor of a shared server."""
    before = dict(os.environ)
    run(config.Settings(chroma_dir=tmp_path), key="sk-visitor-secret")
    assert dict(os.environ) == before
    assert "OPENAI_API_KEY" not in os.environ


# --------------------------------------------------------------- error handling


def test_known_errors_show_their_safe_message(pipeline, settings, monkeypatch):
    def failing_index(*args):
        raise IndexingError("Failed to index the PDF.")

    monkeypatch.setattr(pdf_engine, "index_pdf", failing_index)
    updates = run(settings)
    assert updates[-1].is_error and "Failed to index the PDF." in updates[-1].content


def test_unexpected_errors_hide_details_and_give_a_reference(pipeline, settings, monkeypatch):
    def explode(*args):
        raise RuntimeError("password=hunter2 in /srv/app/secret.py")

    monkeypatch.setattr(pdf_engine, "index_pdf", explode)
    last = run(settings)[-1]
    assert last.is_error and "reference" in last.content
    assert "hunter2" not in last.content and "/srv/app" not in last.content


def test_generation_error_mid_stream_is_reported(pipeline, settings, monkeypatch):
    def broken_stream(*args, **kwargs):
        yield "partial"
        raise GenerationError("The local model failed while generating.")

    monkeypatch.setattr(llm_engine, "generate_openai_stream", broken_stream)
    updates = run(settings)
    assert updates[-1].is_error and "failed while generating" in updates[-1].content


def test_no_retrieved_context_is_an_error(pipeline, settings):
    pipeline["docs"] = []
    assert run(settings)[-1].is_error


def test_empty_model_answer_is_an_error(pipeline, settings):
    pipeline["pieces"] = ["", "  "]
    last = run(settings)[-1]
    assert last.is_error and "empty answer" in last.content


# ------------------------------------------------------------------ local model


def test_local_backend_warns_about_first_use_download(pipeline, settings, monkeypatch):
    seen = {}

    def local_stream(prompt, thinking, temperature, max_tokens, settings=None):
        seen["thinking"] = thinking
        yield "ok"

    monkeypatch.setattr(llm_engine, "generate_local_stream", local_stream)
    updates = run(settings, backend=config.BACKEND_LOCAL, key="")
    assert any("Loading the local model" in u.content for u in updates)
    assert updates[-1].content == "ok" and seen["thinking"] is True
    assert pipeline["index_keys"] == [""]  # local backend never needs an API key


# ---------------------------------------------------------------------- memory


def msgs(*pairs):
    out = []
    for user, answer in pairs:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": answer})
    return out


def test_history_pairs_keeps_the_last_n_good_turns_in_order():
    messages = msgs(("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4"))
    assert chat.history_pairs(messages, 2) == [("q3", "a3"), ("q4", "a4")]


def test_history_pairs_is_empty_when_memory_is_off_or_there_is_no_chat():
    assert chat.history_pairs(msgs(("q", "a")), 0) == []
    assert chat.history_pairs(None, 3) == [] and chat.history_pairs([], 3) == []


def test_history_pairs_drops_errors_progress_and_unanswered_questions():
    messages = [
        *msgs(
            ("bad", "❌ Please provide an OpenAI API key"),
            ("slow", "⏳ Indexing the PDF"),
            ("empty", "  "),
            ("good", "A real answer."),
        ),
        {"role": "user", "content": "in progress"},
        {"role": "assistant", "content": ""},
    ]
    assert chat.history_pairs(messages, 5) == [("good", "A real answer.")]


def test_history_pairs_removes_reasoning_even_when_it_was_cut_off():
    messages = msgs(("q1", "<think>hmm\nsecret</think>The answer."), ("q2", "<think>unfinished"))
    assert chat.history_pairs(messages, 5) == [("q1", "The answer.")]


def test_history_pairs_truncates_long_text_and_tolerates_odd_content():
    long = "x" * 5000
    ((question, answer),) = chat.history_pairs(msgs((long, long)), 3)
    assert len(question) == chat.HISTORY_QUESTION_CHARS and question.endswith("…")
    assert len(answer) == chat.HISTORY_ANSWER_CHARS
    odd = [
        {"role": "user", "content": [{"text": "block question"}, "tail"]},
        {"role": "assistant", "content": {"text": "dict answer"}},
        {"role": "user", "content": 5},
        {"role": "assistant", "content": "ignored: no question text"},
        "not a message",
    ]
    assert chat.history_pairs(odd, 3) == [("block question tail", "dict answer")]


def test_history_reaches_the_prompt_and_follow_ups_search_with_the_previous_question(
    pipeline, settings, monkeypatch
):
    queries = []

    def retrieve(db, question, k, mode=None):
        queries.append(question)
        return pipeline["docs"]

    monkeypatch.setattr(pdf_engine, "retrieve_context", retrieve)
    updates = list(
        chat.run_turn(
            "and its cost?",
            "/x.pdf",
            config.BACKEND_OPENAI,
            "sk-u",
            0.3,
            256,
            True,
            settings,
            history=[("What is alpha?", "A letter.")],
        )
    )
    assert not any(u.is_error for u in updates)
    assert queries == ["and its cost?", "What is alpha? and its cost?"]
    assert "User: What is alpha?\nAssistant: A letter." in pipeline["prompt"]


def test_first_question_searches_once_and_has_no_history_block(pipeline, settings, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pdf_engine, "retrieve_context", lambda *a, **k: calls.append(a[1]) or pipeline["docs"]
    )
    run(settings)
    assert len(calls) == 1 and "<history>" not in pipeline["prompt"]


def test_follow_up_ranking_prefers_chunks_found_by_both_searches(settings, monkeypatch):
    a, b, c, d = (Document(page_content=x, metadata={"page": i}) for i, x in enumerate("abcd"))
    results = {"cost?": [a, b, c], "alpha cost?": [d, b, a]}
    monkeypatch.setattr(pdf_engine, "retrieve_context", lambda db, q, k, mode=None: results[q])
    three = config.Settings(**{**settings.__dict__, "retriever_k": 3})
    fused = chat._retrieve("db", "cost?", [("alpha", "x")], three)
    # a and b were found by both searches, so they outrank d (previous-question search only);
    # c (current-question search only, ranked lower) is displaced.
    assert [doc.page_content for doc in fused] == ["a", "b", "d"]
