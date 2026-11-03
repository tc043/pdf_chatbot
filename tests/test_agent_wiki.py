"""Checks for the agent wiki in docs/agent/wiki, an Open Knowledge Format (OKF v0.2) bundle.

The checks are deterministic: none of them compares a date with the clock, so the suite cannot
start failing on its own. To list the pages whose ``stale_after`` has passed, run
``python -m tests.test_agent_wiki`` from the repository root.
"""

from __future__ import annotations

import ast
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "docs" / "agent" / "wiki"
ROUTER = ROOT / "AGENTS.md"
RESERVED = ("index.md", "log.md")  # OKF §8 and §9: no frontmatter, except okf_version
REQUIRED_KEYS = (
    "type",
    "title",
    "description",
    "sources",
    "generated",
    "verified",
    "status",
    "stale_after",
)
STATUSES = {"draft", "stable", "deprecated"}

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_FENCED = re.compile(r"^```.*?^```[^\n]*$", re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_INDEX_ENTRY = re.compile(r"^\* \[[^\]]+\]\(([^)\s]+)\) - (.+)$", re.MULTILINE)
_ANCHOR = re.compile(r"`([\w./-]+\.py)::([\w.]+)`")
_FOOTNOTE_REF = re.compile(r"\[\^([^\]\s]+)\](?!:)")
_FOOTNOTE_DEF = re.compile(r"^\[\^([^\]\s]+)\]:", re.MULTILINE)
_LOG_HEADING = re.compile(r"^## (.*)$", re.MULTILINE)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ACTOR = re.compile(r"(?:human|process):\S+|[\w.-]+/[\w.+-]+")  # OKF §7
_LINE_NUMBER = re.compile(r"\.py:\d|#L\d")
_SECRET = re.compile(
    r"sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|hf_[A-Za-z0-9]{20,}"
    r"|\.modal\.run"
)


def wiki_files() -> list[Path]:
    return sorted(WIKI.glob("*.md"))


def concept_pages() -> list[Path]:
    return [path for path in wiki_files() if path.name not in RESERVED]


def checked_files() -> list[Path]:
    """The wiki, and the router that points to it when the repository has one."""
    return [*wiki_files(), ROUTER] if ROUTER.is_file() else wiki_files()


def split_frontmatter(path: Path) -> tuple[dict[str, Any] | None, str]:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if match is None:
        return None, text
    return yaml.safe_load(match.group(1)), text[match.end() :]


def parse_instant(value: object) -> dt.datetime | None:
    """A timestamp written as an ISO 8601 string with an explicit offset, else ``None``.

    Unquoted, PyYAML would already have turned it into a ``datetime``; that is rejected too,
    because the OKF reference parser keeps timestamps as text.
    """
    if not isinstance(value, str):
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def prose(text: str) -> str:
    """The text without code blocks and inline code, where brackets are not links."""
    return _INLINE_CODE.sub("", _FENCED.sub("", text))


def _names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def defines(tree: ast.Module, dotted: str) -> bool:
    """True when ``dotted`` (``name`` or ``Class.member``) is defined in the module."""
    scope: list[ast.stmt] = tree.body
    for part in dotted.split("."):
        node = next((node for node in scope if part in _names(node)), None)
        if node is None:
            return False
        scope = getattr(node, "body", [])
    return True


def stale_pages(now: dt.datetime) -> list[str]:
    """The pages whose ``stale_after`` has passed at ``now``, or cannot be read."""
    stale = []
    for path in concept_pages():
        meta, _ = split_frontmatter(path)
        when = parse_instant((meta or {}).get("stale_after"))
        if when is None or now >= when:
            stale.append(path.name)
    return stale


def test_the_wiki_has_its_reserved_files_and_pages():
    assert (WIKI / "index.md").is_file()
    assert (WIKI / "log.md").is_file()
    assert concept_pages()


def test_the_index_lists_every_file_with_its_description():
    meta, body = split_frontmatter(WIKI / "index.md")
    assert meta == {"okf_version": "0.2"}, "the root index carries only okf_version"
    entries = _INDEX_ENTRY.findall(body)
    targets = [target for target, _ in entries]
    assert len(targets) == len(set(targets)), "a file is listed twice"
    assert set(targets) == {path.name for path in wiki_files()} - {"index.md"}
    for target, text in entries:
        page_meta, _ = split_frontmatter(WIKI / target)
        if page_meta is not None:
            description = page_meta.get("description") or ""
            assert description, f"{target} has no description"
            assert description in text, f"the entry for {target} lacks its description"


@pytest.mark.parametrize("path", concept_pages(), ids=lambda path: path.name)
def test_page_frontmatter_is_complete(path):
    meta, _ = split_frontmatter(path)
    assert isinstance(meta, dict), "no YAML frontmatter"
    missing = [key for key in REQUIRED_KEYS if key not in meta]
    assert not missing, f"missing keys: {missing}"
    for key in ("type", "title", "description"):
        assert isinstance(meta[key], str), f"{key} is not text"
        assert meta[key].strip(), f"{key} is empty"

    sources = meta["sources"]
    assert isinstance(sources, list), "sources is not a list"
    assert sources, "a page needs at least one source"
    for source in sources:
        assert isinstance(source, dict), f"not a mapping: {source!r}"
        resource = source.get("resource")
        assert isinstance(resource, str), f"a source has no resource: {source!r}"
        assert resource.strip(), f"a source has no resource: {source!r}"
    ids = [source["id"] for source in sources if "id" in source]
    assert len(ids) == len(set(ids)), "two sources share an id"

    generated = meta["generated"]
    assert isinstance(generated, dict), "generated is not a mapping"
    assert _ACTOR.fullmatch(str(generated.get("by", ""))), "generated.by is not an actor"
    assert parse_instant(generated.get("at")), "generated.at is not a quoted, offset timestamp"

    verified = meta["verified"]
    assert isinstance(verified, (list, dict)), "verified is not a list or a mapping"
    for entry in [verified] if isinstance(verified, dict) else verified:
        assert isinstance(entry, dict), f"not a mapping: {entry!r}"
        assert str(entry.get("by", "")).startswith("human:"), "only a maintainer verifies a page"
        assert parse_instant(entry.get("at")), "verified.at is not a quoted, offset timestamp"

    assert meta["status"] in STATUSES, f"status is {meta['status']!r}"
    assert parse_instant(meta["stale_after"]), "stale_after is not a quoted, offset timestamp"


@pytest.mark.parametrize("path", concept_pages(), ids=lambda path: path.name)
def test_footnotes_name_the_page_sources(path):
    meta, body = split_frontmatter(path)
    assert isinstance(meta, dict), "no YAML frontmatter"
    ids = {source.get("id") for source in meta.get("sources") or [] if isinstance(source, dict)}
    text = prose(body)
    references = set(_FOOTNOTE_REF.findall(text))
    definitions = set(_FOOTNOTE_DEF.findall(text))
    assert references <= definitions, f"footnotes without a definition: {references - definitions}"
    assert definitions <= ids, f"footnotes that name no source: {definitions - ids}"


def test_the_log_is_dated_newest_first():
    text = (WIKI / "log.md").read_text(encoding="utf-8")
    assert not text.startswith("---"), "log.md has no frontmatter"
    headings = [heading.strip() for heading in _LOG_HEADING.findall(text)]
    assert headings, "the log has no dated entries"
    bad = [heading for heading in headings if not _ISO_DATE.fullmatch(heading)]
    assert not bad, f"headings that are not YYYY-MM-DD dates: {bad}"
    dates = [dt.date.fromisoformat(heading) for heading in headings]
    assert dates == sorted(set(dates), reverse=True), "the dates are not unique and newest first"


@pytest.mark.parametrize("path", checked_files(), ids=lambda path: path.name)
def test_relative_links_resolve(path):
    _, body = split_frontmatter(path)
    rooted, broken = [], []
    for target in _LINK.findall(prose(body)):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if target.startswith("/"):
            rooted.append(target)
        elif not (path.parent / target.split("#", 1)[0]).exists():
            broken.append(target)
    assert not rooted, f"links must be relative (see conventions): {rooted}"
    assert not broken, f"links to nothing: {broken}"


@pytest.mark.parametrize("path", concept_pages(), ids=lambda path: path.name)
def test_source_paths_exist(path):
    meta, _ = split_frontmatter(path)
    assert isinstance(meta, dict), "no YAML frontmatter"
    missing = []
    for source in meta.get("sources") or []:
        resource = str(source.get("resource", "")) if isinstance(source, dict) else ""
        if "://" in resource or " " in resource:
            continue  # a URL, or a description such as a range of commits
        if not (path.parent / resource).exists():
            missing.append(resource)
    assert not missing, f"sources that do not exist: {missing}"


@pytest.mark.parametrize("path", checked_files(), ids=lambda path: path.name)
def test_code_anchors_name_something_that_exists(path):
    trees: dict[str, ast.Module] = {}
    missing = []
    for file_name, dotted in _ANCHOR.findall(path.read_text(encoding="utf-8")):
        source = ROOT / file_name
        if not source.is_file():
            missing.append(f"{file_name}::{dotted}")
            continue
        if file_name not in trees:
            trees[file_name] = ast.parse(source.read_text(encoding="utf-8"))
        if not defines(trees[file_name], dotted):
            missing.append(f"{file_name}::{dotted}")
    assert not missing, f"anchors that name nothing: {missing}"


@pytest.mark.parametrize("path", checked_files(), ids=lambda path: path.name)
def test_no_line_numbers_or_secrets(path):
    text = path.read_text(encoding="utf-8")
    assert not _LINE_NUMBER.search(text), "name the code (path::name), not a line number"
    assert not _SECRET.search(text), "looks like a key, a token or a private server address"


def test_stale_pages_compares_with_the_time_it_is_given():
    pages = [path.name for path in concept_pages()]
    assert stale_pages(dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)) == []
    assert stale_pages(dt.datetime(9999, 1, 1, tzinfo=dt.timezone.utc)) == pages


def test_defines_follows_dotted_names():
    tree = ast.parse("X = 1\nclass A:\n    y: int = 2\n    def f(self): ...\ndef g(): ...\n")
    assert all(defines(tree, name) for name in ("X", "A", "A.y", "A.f", "g"))
    assert not any(defines(tree, name) for name in ("Y", "A.g", "g.x", "X.y"))


if __name__ == "__main__":
    now = dt.datetime.now(dt.timezone.utc)
    stale = stale_pages(now)
    for name in stale:
        print(f"stale: {name}")
    print(f"{len(stale)} of {len(concept_pages())} pages are stale at {now:%Y-%m-%dT%H:%M:%SZ}.")
    sys.exit(1 if stale else 0)
