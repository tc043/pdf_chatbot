import run
from app import config


def test_invalid_configuration_exits_with_a_clear_message(monkeypatch, capsys):
    monkeypatch.setenv("PDFCHAT_PORT", "not-a-number")
    assert run.main() == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err and "PDFCHAT_PORT" in err


def test_startup_failure_returns_nonzero_and_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("PDFCHAT_CHROMA_DIR", str(tmp_path / "db"))

    def broken_ui(settings):
        raise RuntimeError("cannot build ui")

    monkeypatch.setattr("app.ui.create_ui", broken_ui)
    assert run.main() == 1
    assert "failed to start" in caplog.text


def test_keyboard_interrupt_is_a_clean_shutdown(monkeypatch, tmp_path):
    monkeypatch.setenv("PDFCHAT_CHROMA_DIR", str(tmp_path / "db"))

    class FakeDemo:
        def launch(self, **kwargs):
            self.kwargs = kwargs
            raise KeyboardInterrupt

    demo = FakeDemo()
    monkeypatch.setattr("app.ui.create_ui", lambda settings: demo)
    assert run.main() == 0
    # Secure launch options are always passed through.
    assert demo.kwargs["server_name"] == "127.0.0.1"
    assert demo.kwargs["show_api"] is False
    assert demo.kwargs["share"] is False
    assert demo.kwargs["max_file_size"] == f"{config.get_settings().max_pdf_mb}mb"


def test_public_binding_logs_security_warnings(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("PDFCHAT_CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("PDFCHAT_HOST", "0.0.0.0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-key-123456")

    class FakeDemo:
        def launch(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr("app.ui.create_ui", lambda settings: FakeDemo())
    assert run.main() == 0
    assert "without authentication" in caplog.text
    assert "spend your OpenAI credit" in caplog.text
    assert "sk-server-key" not in caplog.text
