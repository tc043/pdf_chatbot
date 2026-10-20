import dataclasses
import datetime as dt
import os
import time

import pytest

from app import chat, config, index_store, pdf_engine
from tests.samples import SAMPLE_PAGES


@pytest.fixture
def build(make_pdf, settings, fake_embeddings):
    """Index a distinct PDF and return (path, fingerprint)."""
    counter = {"n": 0}

    def _build(backend=config.BACKEND_LOCAL):
        counter["n"] += 1
        pages = [f"Document {counter['n']} " + text for text in SAMPLE_PAGES]
        path = make_pdf(pages)
        pdf_engine.index_pdf(path, backend, "sk-x", settings)
        return path, pdf_engine.file_fingerprint(path)

    return _build


def age(settings, fingerprint, days):
    """Pretend a PDF's index was last used ``days`` days ago."""
    when = time.time() - days * 86400
    os.utime(pdf_engine.get_bundle_dir(fingerprint, settings), (when, when))


def test_list_is_empty_before_anything_is_indexed(settings):
    assert index_store.list_indexes(settings) == []


def test_list_reports_collections_with_their_fingerprint_and_size(build, settings):
    _, fingerprint = build()
    (info,) = index_store.list_indexes(settings)
    assert info.fingerprint == fingerprint[:40]
    assert info.kind == "local" and info.chunks >= 3 and info.bundle_bytes > 0
    assert info.last_used is not None


def test_reusing_an_index_refreshes_its_last_used_time(build, settings):
    path, fingerprint = build()
    age(settings, fingerprint, 10)
    before = index_store.list_indexes(settings)[0].last_used
    pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert index_store.list_indexes(settings)[0].last_used > before


def test_delete_for_pdf_removes_collections_and_bundle_but_not_other_pdfs(build, settings):
    path, fingerprint = build()
    _, other = build()
    assert index_store.delete_for_pdf(path, settings) == 1
    assert [i.fingerprint for i in index_store.list_indexes(settings)] == [other[:40]]
    assert not pdf_engine.get_bundle_dir(fingerprint, settings).exists()
    assert pdf_engine.get_bundle_dir(other, settings).exists()
    assert index_store.delete_for_pdf(path, settings) == 0  # nothing left to delete


def test_a_deleted_index_is_rebuilt_on_the_next_use(build, settings):
    path, _ = build()
    index_store.delete_for_pdf(path, settings)
    db = pdf_engine.index_pdf(path, config.BACKEND_LOCAL, settings=settings)
    assert db.get()["ids"]
    assert len(index_store.list_indexes(settings)) == 1


def test_the_bundle_is_kept_while_another_setup_still_uses_it(build, make_pdf, settings):
    path, fingerprint = build()
    pdf_engine.index_pdf(path, config.BACKEND_OPENAI, "sk-x", settings)  # second collection
    both = index_store.list_indexes(settings)
    assert {i.kind for i in both} == {"local", "openai"}
    index_store._delete(settings, [i for i in both if i.kind == "openai"])
    assert pdf_engine.get_bundle_dir(fingerprint, settings).exists()
    index_store._delete(settings, index_store.list_indexes(settings))
    assert not pdf_engine.get_bundle_dir(fingerprint, settings).exists()


def test_prune_by_age_spares_recent_and_protected_pdfs(build, settings):
    _, old = build()
    _, older_but_protected = build()
    _, recent = build()
    age(settings, old, 40)
    age(settings, older_but_protected, 50)
    removed = index_store.prune(settings, older_than_days=30, protect={older_but_protected})
    assert removed == 1
    left = {i.fingerprint for i in index_store.list_indexes(settings)}
    assert left == {older_but_protected[:40], recent[:40]}


def test_prune_keep_retains_the_newest_pdfs(build, settings):
    fingerprints = [build()[1] for _ in range(4)]
    for days, fingerprint in zip((4, 3, 2, 1), fingerprints, strict=True):
        age(settings, fingerprint, days)
    assert index_store.prune(settings, keep=2) == 2
    left = {i.fingerprint for i in index_store.list_indexes(settings)}
    assert left == {fingerprints[2][:40], fingerprints[3][:40]}


def test_prune_with_no_rules_deletes_nothing(build, settings):
    build()
    assert index_store.prune(settings) == 0 and len(index_store.list_indexes(settings)) == 1


def test_clear_removes_everything_including_unrecognised_collections(build, settings):
    build()
    client = index_store._client(settings)
    client.create_collection("legacy_collection_without_fingerprint")
    assert len(index_store.list_indexes(settings)) == 2
    assert index_store.clear(settings) == 2
    assert index_store.list_indexes(settings) == []


def test_unrecognised_collections_are_never_pruned_by_age(build, settings):
    index_store._client(settings).create_collection("legacy_collection")
    assert index_store.prune(settings, older_than_days=1, keep=1) == 0


# ------------------------------------------------------------------ retention


def test_auto_prune_is_off_by_default(build, settings):
    _, fingerprint = build()
    age(settings, fingerprint, 400)
    assert index_store.auto_prune(settings, protect="x" * 40) == 0
    assert len(index_store.list_indexes(settings)) == 1


def test_auto_prune_applies_configured_limits_but_protects_the_current_pdf(build, settings):
    _, stale = build()
    _, current = build()
    age(settings, stale, 90)
    age(settings, current, 90)
    limited = dataclasses.replace(settings, index_max_age_days=30)
    assert index_store.auto_prune(limited, protect=current) == 1
    assert [i.fingerprint for i in index_store.list_indexes(settings)] == [current[:40]]


def test_chat_applies_retention_after_indexing_and_survives_failures(
    settings, monkeypatch, make_pdf
):
    calls = []
    monkeypatch.setattr(index_store, "auto_prune", lambda s, protect: calls.append(protect) or 0)
    path = make_pdf(SAMPLE_PAGES)
    chat._apply_retention(path, settings)  # limits are 0: skipped entirely
    assert calls == []
    limited = dataclasses.replace(settings, index_max_count=5)
    chat._apply_retention(path, limited)
    assert calls == [pdf_engine.file_fingerprint(path)]

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(index_store, "auto_prune", boom)
    chat._apply_retention(path, limited)  # must not raise


# ------------------------------------------------------------------------ CLI


def run_cli(settings, monkeypatch, *argv):
    monkeypatch.setenv("PDFCHAT_CHROMA_DIR", str(settings.chroma_dir))
    config.get_settings.cache_clear()
    return index_store.main(list(argv))


def test_cli_list_delete_prune_and_clear(build, settings, monkeypatch, capsys):
    assert run_cli(settings, monkeypatch, "list") == 0
    assert "No stored indexes" in capsys.readouterr().out

    _, first = build()
    _, second = build()
    age(settings, first, 5)
    run_cli(settings, monkeypatch, "list")
    listing = capsys.readouterr().out
    assert first[:12] in listing and second[:12] in listing and "local" in listing

    run_cli(settings, monkeypatch, "prune", "--keep", "1")
    assert "Deleted 1" in capsys.readouterr().out

    run_cli(settings, monkeypatch, "delete", second[:10])
    assert "Deleted 1" in capsys.readouterr().out
    assert index_store.list_indexes(settings) == []

    build()
    with pytest.raises(SystemExit):
        run_cli(settings, monkeypatch, "clear")  # needs --yes
    assert len(index_store.list_indexes(settings)) == 1
    run_cli(settings, monkeypatch, "clear", "--yes")
    assert index_store.list_indexes(settings) == []


@pytest.mark.parametrize("argv", [["delete", "abc"], ["prune"]])
def test_cli_rejects_incomplete_commands(settings, monkeypatch, argv):
    with pytest.raises(SystemExit):
        run_cli(settings, monkeypatch, *argv)


def test_cli_delete_refuses_an_ambiguous_prefix(settings, monkeypatch):
    twins = [
        index_store.IndexInfo(f"c{i}", "ab" * 6 + str(i) * 28, "local", 1, 0, None) for i in (1, 2)
    ]
    monkeypatch.setattr(index_store, "list_indexes", lambda s: twins)
    monkeypatch.setattr(index_store, "_delete", lambda *a: pytest.fail("must not delete"))
    with pytest.raises(SystemExit):
        run_cli(settings, monkeypatch, "delete", "abababab")


def test_age_comparison_uses_timezone_aware_datetimes(build, settings):
    build()
    info = index_store.list_indexes(settings)[0]
    assert info.last_used.tzinfo is not None
    assert info.last_used <= dt.datetime.now(dt.timezone.utc)


@pytest.mark.parametrize("argv", [["prune", "--older-than-days", "-1"], ["prune", "--keep", "-3"]])
def test_cli_rejects_negative_prune_limits_instead_of_deleting_everything(
    build, settings, monkeypatch, argv
):
    _, fingerprint = build()
    age(settings, fingerprint, 1)
    with pytest.raises(SystemExit):
        run_cli(settings, monkeypatch, *argv)
    assert len(index_store.list_indexes(settings)) == 1


def test_cli_rejects_non_numeric_prune_limits(settings, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(settings, monkeypatch, "prune", "--keep", "many")
