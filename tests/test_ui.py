import json
import urllib.request

import gradio as gr
import pytest

from app import chat, config, pdf_engine
from app.ui import create_ui
from tests.samples import SAMPLE_PAGES

SERVER_KEY = "sk-server-side-secret-key-0000"


@pytest.fixture
def settings(tmp_path):
    return config.Settings(chroma_dir=tmp_path / "chroma", server_openai_api_key=SERVER_KEY)


def handler(demo, name):
    """Return the Python function behind a named event handler."""
    return next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == name)


# ------------------------------------------------------------------- security


def test_server_api_key_is_never_sent_to_the_browser(settings):
    """Regression: the old UI pre-filled the key textbox from OPENAI_API_KEY, so every
    visitor received the server owner's key in the page config."""
    demo = create_ui(settings)
    rendered = json.dumps(demo.get_config_file(), default=str)
    assert SERVER_KEY not in rendered
    key_box = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "OpenAI API Key")
    assert key_box.value in ("", None)
    assert key_box.type == "password"


def test_key_placeholder_reflects_whether_a_server_key_is_usable(tmp_path):
    def placeholder(**kw):
        demo = create_ui(config.Settings(chroma_dir=tmp_path, **kw))
        box = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "OpenAI API Key")
        return box.placeholder

    assert "server" in placeholder(server_openai_api_key="sk-x")
    assert placeholder() == "sk-..."
    assert placeholder(server_openai_api_key="sk-x", use_server_key=False) == "sk-..."


# ------------------------------------------------------------------- handlers


def test_uploading_a_valid_pdf_loads_the_viewer(settings, make_pdf):
    demo = create_ui(settings)
    path = make_pdf(SAMPLE_PAGES)
    out = handler(demo, "handle_pdf_load")(path)
    loaded_path, page, total, image, label, jump = out
    assert (loaded_path, page, total, label) == (path, 0, 3, "Page 1 of 3")
    assert image is not None and jump["maximum"] == 3


def test_uploading_an_invalid_file_shows_an_error_toast(settings, tmp_path):
    demo = create_ui(settings)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(gr.Error):
        handler(demo, "handle_pdf_load")(str(bad))


def test_clearing_the_upload_resets_the_viewer(settings):
    out = handler(create_ui(settings), "handle_pdf_load")(None)
    assert out[:3] == ("", 0, 0) and out[3] is None


def test_navigation_is_clamped_and_safe_without_a_document(settings, make_pdf):
    navigate = handler(create_ui(settings), "navigate_page")
    path = make_pdf(SAMPLE_PAGES)
    assert navigate("prev", path, 0, 3)[0] == 0
    assert navigate("next", path, 2, 3)[0] == 2
    assert navigate("next", path, 0, 3)[1] == "Page 2 of 3"
    assert navigate("next", "", 0, 0)[1] == "Page 0 of 0"


def test_jump_clamps_out_of_range_pages(settings, make_pdf):
    jump = handler(create_ui(settings), "jump_to_page")
    path = make_pdf(SAMPLE_PAGES)
    assert jump(99, path, 3)[:2] == (2, "Page 3 of 3")
    assert jump(-4, path, 3)[:2] == (0, "Page 1 of 3")
    assert jump(None, path, 3)[0] == 0


def _is_skip(value):
    return isinstance(value, dict) and value.get("__type__") == "update" and len(value) == 1


def test_streaming_updates_the_viewer_once_and_never_yields_state_objects(
    settings, make_pdf, monkeypatch
):
    """Regression: the old handler re-sent (or blanked) the page image on every token and
    yielded gr.State objects as outputs."""
    path = make_pdf(SAMPLE_PAGES)
    renders = []
    real_render = pdf_engine.render_pdf_page
    monkeypatch.setattr(
        pdf_engine, "render_pdf_page", lambda p, n: renders.append(n) or real_render(p, n)
    )
    updates = [
        chat.TurnUpdate("Indexing"),
        chat.TurnUpdate("Generating", page=1),
        chat.TurnUpdate("Hel", page=1),
        chat.TurnUpdate("Hello", page=1),
    ]
    monkeypatch.setattr(chat, "run_turn", lambda *a, **k: iter(updates))

    generate = handler(create_ui(settings), "generate_response")
    yields = list(generate([], "question?", config.BACKEND_OPENAI, "", path, 0.3, 256, True, 3))

    history = yields[-1][0]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "question?" and history[1]["content"] == "Hello"
    assert all(y[1] == "" for y in yields)  # input box cleared immediately
    assert not any(isinstance(v, gr.State) for y in yields for v in y)
    assert renders == [1]  # rendered exactly once, not once per streamed chunk
    jump_yield = next(y for y in yields if y[2] == 1)
    assert jump_yield[3] == "Page 2 of 3" and jump_yield[5] == 2
    assert all(_is_skip(v) for y in yields if y is not jump_yield for v in y[2:])


def test_blank_question_does_not_start_a_turn(settings, monkeypatch):
    monkeypatch.setattr(chat, "run_turn", lambda *a, **k: pytest.fail("must not run"))
    generate = handler(create_ui(settings), "generate_response")
    (only,) = list(generate([], "   ", config.BACKEND_OPENAI, "", "/x.pdf", 0.3, 256, True, 0))
    assert all(_is_skip(v) for v in only)


# ---------------------------------------------------------------- live server


def test_server_starts_and_serves_the_app(settings):
    demo = create_ui(settings)
    _, url, _ = demo.launch(
        server_name="127.0.0.1", prevent_thread_lock=True, quiet=True, show_api=False
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - local URL
            body = response.read().decode()
        assert response.status == 200
        assert "PDF Retrieval Chatbot" in body
        assert SERVER_KEY not in body
    finally:
        demo.close()


def test_generate_response_passes_earlier_turns_to_the_chat(settings, monkeypatch, make_pdf):
    seen = {}

    def fake_run_turn(*args, **kwargs):
        seen["history"] = kwargs.get("history")
        return iter([chat.TurnUpdate("done")])

    monkeypatch.setattr(chat, "run_turn", fake_run_turn)
    earlier = [
        {"role": "user", "content": "first?"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "oops"},
        {"role": "assistant", "content": "❌ something failed"},
    ]
    generate = handler(create_ui(settings), "generate_response")
    list(
        generate(
            earlier, "second?", config.BACKEND_OPENAI, "", make_pdf(SAMPLE_PAGES), 0.3, 256, True, 3
        )
    )
    assert seen["history"] == [("first?", "first answer")]


def test_uploading_or_clearing_a_pdf_clears_the_chat(settings):
    demo = create_ui(settings)
    chatbot = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "Chat History")
    clears = [
        f
        for f in demo.fns.values()
        if getattr(f.fn, "__name__", "") == "<lambda>"
        and [b._id for b in f.outputs] == [chatbot._id]
    ]
    assert len(clears) >= 2 and all(f.fn() == [] for f in clears)
