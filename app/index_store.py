"""Managing the on-disk index store: list, delete and prune stored PDF indexes.

Every indexed PDF leaves a Chroma collection (named from the file's content fingerprint, see
:func:`app.pdf_engine.get_collection_name`) and an OKF bundle directory. Nothing removes them
automatically unless a limit is configured, so this module lets operators inspect and delete
them. It is also a small command line tool::

    python -m app.index_store list
    python -m app.index_store delete <fingerprint-prefix>
    python -m app.index_store prune --older-than-days 30 --keep 20
    python -m app.index_store clear --yes

Deleting an index is always safe: it is rebuilt from the PDF on the next question.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import shutil
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import config, pdf_engine

logger = logging.getLogger(__name__)

# pdf_<40 hex chars of the content fingerprint>_<openai|local>_<8 hex chars of the setup hash>
_NAME_RE = re.compile(r"^pdf_(?P<fingerprint>[0-9a-f]{40})_(?P<kind>openai|local)_[0-9a-f]{8}$")


@dataclass(frozen=True)
class IndexInfo:
    """One stored Chroma collection."""

    collection: str
    fingerprint: str | None  # None for collections this app version did not name
    kind: str | None
    chunks: int
    bundle_bytes: int  # size of the PDF's OKF bundle (shared by its collections)
    last_used: dt.datetime | None  # when the index was built or last reused


def _client(settings: config.Settings) -> Any:
    from chromadb import PersistentClient
    from chromadb.config import Settings as ChromaSettings

    return PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _directory_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def _last_used(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        return None


def list_indexes(settings: config.Settings | None = None) -> list[IndexInfo]:
    """Return every stored collection, most recently used first (unknown age last)."""
    settings = settings or config.get_settings()
    if not settings.chroma_dir.is_dir():
        return []
    client = _client(settings)
    infos = []
    for entry in client.list_collections():
        name = getattr(entry, "name", entry)
        match = _NAME_RE.match(name)
        fingerprint = match["fingerprint"] if match else None
        bundle = pdf_engine.get_bundle_dir(fingerprint, settings) if fingerprint else None
        infos.append(
            IndexInfo(
                collection=name,
                fingerprint=fingerprint,
                kind=match["kind"] if match else None,
                chunks=client.get_collection(name).count(),
                bundle_bytes=_directory_bytes(bundle) if bundle else 0,
                last_used=_last_used(bundle) if bundle else None,
            )
        )
    epoch = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    return sorted(infos, key=lambda info: info.last_used or epoch, reverse=True)


def _delete(settings: config.Settings, infos: Sequence[IndexInfo]) -> int:
    """Delete the given collections, and bundles no remaining collection needs."""
    if not infos:
        return 0
    with pdf_engine._INDEX_LOCK:
        client = _client(settings)
        for info in infos:
            client.delete_collection(info.collection)
        deleted = {info.collection for info in infos}
        remaining = set()
        for entry in client.list_collections():
            name = getattr(entry, "name", entry)
            match = _NAME_RE.match(name)
            if match and name not in deleted:
                remaining.add(match["fingerprint"])
        for fingerprint in {i.fingerprint for i in infos if i.fingerprint} - remaining:
            shutil.rmtree(pdf_engine.get_bundle_dir(fingerprint, settings), ignore_errors=True)
    logger.info("Deleted %d stored index(es)", len(infos))
    return len(infos)


def delete_for_pdf(file_path: str, settings: config.Settings | None = None) -> int:
    """Delete the stored indexes (all embedding setups) of one PDF; returns how many."""
    settings = settings or config.get_settings()
    fingerprint = pdf_engine.file_fingerprint(file_path)
    return _delete(
        settings, [i for i in list_indexes(settings) if i.fingerprint == fingerprint[:40]]
    )


def prune(
    settings: config.Settings | None = None,
    *,
    older_than_days: int = 0,
    keep: int = 0,
    protect: Collection[str] = (),
) -> int:
    """Delete indexes unused for ``older_than_days`` days and/or beyond the ``keep`` newest PDFs.

    A value of 0 disables that rule. Indexes whose age is unknown, and PDFs whose fingerprint is
    in ``protect`` (for example the one being used right now), are never removed.
    """
    settings = settings or config.get_settings()
    if not older_than_days and not keep:
        return 0
    protect = {fingerprint[:40] for fingerprint in protect}
    candidates = [i for i in list_indexes(settings) if i.last_used and i.fingerprint]
    doomed: list[IndexInfo] = []
    if older_than_days:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=older_than_days)
        doomed += [i for i in candidates if i.last_used < cutoff]
    if keep:
        seen: list[str] = []  # newest first, one entry per PDF
        for info in candidates:
            if info.fingerprint not in seen:
                seen.append(info.fingerprint)
        stale = set(seen[keep:])
        doomed += [i for i in candidates if i.fingerprint in stale]
    unique = {i.collection: i for i in doomed if i.fingerprint not in protect}
    return _delete(settings, list(unique.values()))


def clear(settings: config.Settings | None = None) -> int:
    """Delete every stored index, including collections this version cannot identify."""
    settings = settings or config.get_settings()
    return _delete(settings, list_indexes(settings))


def auto_prune(settings: config.Settings, protect: str) -> int:
    """Apply the configured retention limits, never removing the PDF in use (``protect``)."""
    if not (settings.index_max_age_days or settings.index_max_count):
        return 0
    return prune(
        settings,
        older_than_days=settings.index_max_age_days,
        keep=settings.index_max_count,
        protect={protect},
    )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _format_table(infos: Sequence[IndexInfo]) -> str:
    if not infos:
        return "No stored indexes."
    lines = [f"{'fingerprint':<14}{'backend':<9}{'chunks':>7}{'bundle':>10}  last used"]
    for i in infos:
        used = i.last_used.strftime("%Y-%m-%d %H:%M") if i.last_used else "unknown"
        label = i.fingerprint[:12] if i.fingerprint else i.collection[:12]
        lines.append(
            f"{label:<14}{i.kind or '?':<9}{i.chunks:>7}{i.bundle_bytes / 1024:>8.0f}KB  {used}"
        )
    return "\n".join(lines)


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more (0 turns the rule off)")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the stored PDF indexes.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show stored indexes")
    delete_cmd = commands.add_parser("delete", help="delete the indexes of one PDF")
    delete_cmd.add_argument("fingerprint", help="fingerprint prefix shown by 'list' (6+ chars)")
    prune_cmd = commands.add_parser("prune", help="delete old or surplus indexes")
    prune_cmd.add_argument("--older-than-days", type=_non_negative_int, default=0)
    prune_cmd.add_argument(
        "--keep", type=_non_negative_int, default=0, help="keep only the N newest PDFs"
    )
    clear_cmd = commands.add_parser("clear", help="delete every stored index")
    clear_cmd.add_argument("--yes", action="store_true", help="confirm the deletion")
    args = parser.parse_args(argv)

    settings = config.get_settings()
    if args.command == "list":
        print(_format_table(list_indexes(settings)))
    elif args.command == "delete":
        if len(args.fingerprint) < 6:
            parser.error("give at least 6 characters of the fingerprint")
        matches = [
            i for i in list_indexes(settings) if (i.fingerprint or "").startswith(args.fingerprint)
        ]
        if len({i.fingerprint for i in matches}) > 1:
            parser.error("that prefix matches several PDFs; give more characters")
        print(f"Deleted {_delete(settings, matches)} index(es).")
    elif args.command == "prune":
        if not (args.older_than_days or args.keep):
            parser.error("give --older-than-days and/or --keep")
        print(f"Deleted {prune(settings, older_than_days=args.older_than_days, keep=args.keep)}.")
    elif args.command == "clear":
        if not args.yes:
            parser.error("this deletes every stored index; repeat with --yes to confirm")
        print(f"Deleted {clear(settings)} index(es).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
