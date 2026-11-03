"""PDF handling: validation, page rendering, OKF v0.2 bundle extraction, and vector indexing."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from app import config, ocr_remote, retrieval
from app.exceptions import (
    IndexingError,
    ModelUnavailableError,
    PDFChatbotError,
    PDFValidationError,
    RetrievalError,
    wrap_provider_error,
)

if TYPE_CHECKING:  # pragma: no cover
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

RENDER_DPI = 200
OKF_VERSION = "0.2"
# Bump when extraction output changes so cached collections are rebuilt, not reused.
EXTRACTION_VERSION = "2"
OKF_PRODUCER = "pdf_chatbot/pymupdf"
OCR_DPI = 200

# Guards the Chroma store and the bundle directories. It is held only for short reads and
# writes: extraction, OCR and embedding happen outside it, so one slow PDF does not stall the
# others. Builds of the same index are serialised by a per-index lock (see _build_lock).
_INDEX_LOCK = threading.Lock()
_build_locks: dict[str, threading.Lock] = {}  # one small lock per index built by this process
_build_locks_guard = threading.Lock()

# An index with scanned pages that OCR could not read is rebuilt on later requests (only the
# unread pages are sent to OCR again) until this many attempts, then kept as it is.
MAX_INDEX_ATTEMPTS = 3
# Text that OCR read, per (document, OCR setup, page), so a retry does not read pages again.
# A document's entries are dropped once it needs no more retries or its index is deleted.
_OCR_CACHE_PAGES = 500
_ocr_cache: OrderedDict[tuple[str, str, int], str] = OrderedDict()
_ocr_cache_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Validation and rendering
# --------------------------------------------------------------------------- #
def validate_pdf(file_path: str | None, settings: config.Settings | None = None) -> int:
    """Check that ``file_path`` is a readable, unencrypted PDF within limits.

    Returns the page count. Raises :class:`PDFValidationError` with a user-safe
    message otherwise.
    """
    settings = settings or config.get_settings()
    if not file_path:
        raise PDFValidationError("No PDF has been uploaded yet.")

    path = Path(file_path)
    if not path.is_file():
        raise PDFValidationError("The PDF could not be found. Please upload it again.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_pdf_mb:
        raise PDFValidationError(
            f"The PDF is {size_mb:.1f} MB; the limit is {settings.max_pdf_mb} MB."
        )

    try:
        with pymupdf.open(path) as doc:
            if not doc.is_pdf:
                raise PDFValidationError("The uploaded file is not a PDF.")
            if doc.needs_pass:
                raise PDFValidationError("The PDF is password-protected, which is not supported.")
            pages = doc.page_count
    except PDFValidationError:
        raise
    except Exception as exc:
        logger.warning("Could not open %s as a PDF: %s", path.name, exc)
        raise PDFValidationError("The file could not be read as a PDF; it may be corrupt.") from exc

    if pages == 0:
        raise PDFValidationError("The PDF has no pages.")
    if pages > settings.max_pdf_pages:
        raise PDFValidationError(
            f"The PDF has {pages} pages; the limit is {settings.max_pdf_pages}."
        )
    return pages


def render_pdf_page(file_path: str | None, page_num: int) -> tuple[Image.Image | None, int]:
    """Render one page (0-based, clamped to the valid range) as a PIL image.

    Returns ``(image, total_pages)``, or ``(None, 0)`` when there is no file.
    """
    if not file_path or not Path(file_path).is_file():
        return None, 0

    try:
        with pymupdf.open(file_path) as doc:
            total_pages = doc.page_count
            if total_pages == 0:
                return None, 0
            index = min(max(int(page_num), 0), total_pages - 1)
            zoom = RENDER_DPI / 72
            pix = doc[index].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception as exc:
        logger.warning("Could not render page %s of %s: %s", page_num, file_path, exc)
        raise PDFValidationError("That page could not be rendered.") from exc
    return image, total_pages


# --------------------------------------------------------------------------- #
# Index and OKF bundle naming
# --------------------------------------------------------------------------- #
def file_fingerprint(file_path: str) -> str:
    """SHA-256 of the file contents (streamed, so large files stay out of memory)."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_collection_name(
    fingerprint: str,
    backend: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    ocr: str = "",
) -> str:
    """Deterministic Chroma collection name for one (document, embedding setup) pair.

    Keyed on the document *content* rather than its filename, so two different
    PDFs that share a filename never collide and an edited PDF is re-indexed.
    The embedding model, chunking parameters and extraction format version are
    part of the key so changing them never reuses vectors of the wrong dimension,
    granularity or text format. ``ocr`` is the OCR language when OCR is active (else
    empty), so enabling OCR rebuilds indexes that were made without it; indexes made
    with OCR off keep their old names.
    Result is at most 60 characters and uses only ``[a-z0-9_]``.
    """
    kind = "openai" if backend == config.BACKEND_OPENAI else "local"
    setup = f"{embedding_model}|{chunk_size}|{chunk_overlap}|x{EXTRACTION_VERSION}"
    if ocr:
        setup += f"|ocr={ocr}"
    tag = hashlib.sha256(setup.encode("utf-8")).hexdigest()[:8]
    return f"pdf_{fingerprint[:40]}_{kind}_{tag}"


def _mark_used(fingerprint: str, settings: config.Settings) -> None:
    """Record that an index was used (the bundle directory's mtime is its 'last used' time)."""
    try:
        os.utime(get_bundle_dir(fingerprint, settings))
    except OSError:
        logger.debug("Could not update the last-used time of %s", fingerprint[:12])


def get_bundle_dir(fingerprint: str, settings: config.Settings | None = None) -> Path:
    """Return the on-disk path for the PDF's Open Knowledge Format (OKF) bundle."""
    settings = settings or config.get_settings()
    return settings.chroma_dir / "bundles" / f"pdf_{fingerprint[:40]}"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _local_embeddings(model_name: str) -> Any:
    """Load the local sentence-transformers model once per process."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ModelUnavailableError(
            "Local embeddings need the optional dependencies: pip install -r requirements-local.txt"
        ) from exc
    logger.info("Loading local embedding model %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def get_embeddings_model(
    backend: str,
    openai_api_key: str | None = None,
    settings: config.Settings | None = None,
) -> Any:
    """Return the embeddings implementation for ``backend``."""
    settings = settings or config.get_settings()
    if backend == config.BACKEND_OPENAI:
        if not openai_api_key:
            raise ModelUnavailableError("An OpenAI API key is required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=openai_api_key,
            timeout=settings.openai_timeout_s,
            max_retries=settings.openai_max_retries,
        )
    return _local_embeddings(settings.local_embedding_model)


# --------------------------------------------------------------------------- #
# Table-aware extraction & Open Knowledge Format (OKF v0.2) bundle builder
# --------------------------------------------------------------------------- #
def _boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    threshold: float = 0.5,
) -> bool:
    """Return True if at least ``threshold`` of box ``a``'s area lies inside box ``b``."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    a_area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return (inter_area / a_area) >= threshold


def _extract_page_markdown(page: pymupdf.Page) -> tuple[str, int]:
    """Extract a page's text and tables in vertical reading order as Markdown.

    Returns ``(markdown_text, table_count)``. If no tables are present on the page,
    returns ``page.get_text("text").strip()`` so plain-text pages remain byte-identical.
    """
    tables: list[tuple[tuple[float, float, float, float], str]] = []
    try:
        finder = page.find_tables()
        for tab in getattr(finder, "tables", ()):
            md_table = tab.to_markdown().strip()
            if md_table:
                tables.append((tuple(tab.bbox), md_table))
    except Exception:
        tables = []

    if not tables:
        return page.get_text("text").strip(), 0

    items: list[tuple[float, float, str]] = [
        (bbox[1], bbox[0], md_table) for bbox, md_table in tables
    ]
    for block in page.get_text("blocks"):
        if len(block) >= 7 and block[6] != 0:
            continue  # Skip image blocks
        bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4].strip()
        if not text:
            continue
        if any(_boxes_overlap((bx0, by0, bx1, by1), t_bbox) for t_bbox, _ in tables):
            continue
        items.append((by0, bx0, text))

    items.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
    return "\n\n".join(part for _, _, part in items).strip(), len(tables)


def _one_line_summary(text: str, max_chars: int = 140) -> str:
    """Create a single-line OKF description from the first non-table prose line."""
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#*- ").strip()
        if cleaned and not cleaned.startswith("|"):
            return cleaned[: max_chars - 1] + "..." if len(cleaned) > max_chars else cleaned
    return "Extracted PDF page content."


def _yaml_quote(value: str) -> str:
    """Safely quote a scalar string for YAML 1.2 frontmatter without external dependencies."""
    return json.dumps(value, ensure_ascii=False)


def _render_okf_page_concept(
    doc: Document,
    *,
    prev_concept: str | None,
    next_concept: str | None,
    generated_at: str,
) -> str:
    """Format one page :class:`Document` as a complete OKF v0.2 Markdown file."""
    source = str(doc.metadata.get("source", "document.pdf"))
    page_zero = int(doc.metadata.get("page", 0))
    page_one = page_zero + 1
    table_count = int(doc.metadata.get("table_count", 0))
    sha256 = str(doc.metadata.get("sha256", ""))
    title = f"{source} - Page {page_one}"
    description = _one_line_summary(doc.page_content)
    resource = f"{source}#page={page_one}"

    tags = ["pdf", f"page-{page_one}"]
    if table_count > 0:
        tags.append("tables")
    if doc.metadata.get("ocr"):
        tags.append("ocr")
    tags_yaml = ", ".join(_yaml_quote(t) for t in tags)

    frontmatter = [
        "---",
        'type: "PDF Page"',
        f"title: {_yaml_quote(title)}",
        f"description: {_yaml_quote(description)}",
        f"resource: {_yaml_quote(resource)}",
        f"tags: [{tags_yaml}]",
        "status: stable",
        f"generated: {{ by: {_yaml_quote(OKF_PRODUCER)}, at: {generated_at} }}",
        "sources:",
        "  - id: source-pdf",
        f"    resource: {_yaml_quote(source)}",
        f"    title: {_yaml_quote(source)}",
        f"page: {page_one}",
        f"table_count: {table_count}",
    ]
    if sha256:
        frontmatter.append(f"sha256: {_yaml_quote(sha256)}")
    frontmatter.append("---")

    nav_links = ["[Document Overview](/document.md)"]
    if prev_concept:
        nav_links.append(f"[Previous Page](/{prev_concept}.md)")
    if next_concept:
        nav_links.append(f"[Next Page](/{next_concept}.md)")
    nav_line = " | ".join(nav_links)

    return (
        "\n".join(frontmatter)
        + f"\n\n# Page {page_one}\n\n"
        + f"{doc.page_content}\n\n"
        + f"## Related Concepts\n\n{nav_line}\n"
    )


def build_okf_bundle(
    documents: list[Document],
    bundle_dir: Path,
    *,
    fingerprint: str = "",
) -> Path:
    """Write a complete Open Knowledge Format (OKF v0.2) bundle for ``documents``.

    Structure:
    * ``index.md`` — Bundle root progressive-disclosure index (with ``okf_version: "0.2"``)
    * ``document.md`` — Root ``PDF Document`` concept linking to all extracted pages
    * ``pages/index.md`` — Directory listing for ``pages/``
    * ``pages/page_XXXX.md`` — Individual ``PDF Page`` concepts with YAML frontmatter
    """
    pages_dir = bundle_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_name = (
        str(documents[0].metadata.get("source", "document.pdf")) if documents else "document.pdf"
    )

    concept_ids = [
        f"pages/page_{int(d.metadata.get('page', i)) + 1:04d}" for i, d in enumerate(documents)
    ]
    page_index_entries: list[str] = []

    for idx, doc in enumerate(documents):
        concept_id = concept_ids[idx]
        prev_id = concept_ids[idx - 1] if idx > 0 else None
        next_id = concept_ids[idx + 1] if idx + 1 < len(concept_ids) else None

        doc.metadata["okf_concept"] = f"/{concept_id}"
        doc.metadata["okf_path"] = f"{concept_id}.md"
        if fingerprint:
            doc.metadata["sha256"] = fingerprint

        page_one = int(doc.metadata.get("page", idx)) + 1
        summary = _one_line_summary(doc.page_content)
        rel_file = f"page_{page_one:04d}.md"
        page_index_entries.append(f"* [Page {page_one}]({rel_file}) - {summary}")

        concept_md = _render_okf_page_concept(
            doc,
            prev_concept=prev_id,
            next_concept=next_id,
            generated_at=generated_at,
        )
        (bundle_dir / f"{concept_id}.md").write_text(concept_md, encoding="utf-8")

    # pages/index.md (OKF §8: subdirectory index.md has no frontmatter)
    pages_index_md = "# Pages\n\n" + "\n".join(page_index_entries) + "\n"
    (pages_dir / "index.md").write_text(pages_index_md, encoding="utf-8")

    # document.md (Root concept describing the PDF asset as a whole)
    total_tables = sum(int(d.metadata.get("table_count", 0)) for d in documents)
    doc_desc = (
        f"Extracted knowledge bundle for {source_name} "
        f"({len(documents)} pages, {total_tables} tables)."
    )
    doc_links = "\n".join(
        f"* [Page {int(d.metadata.get('page', i)) + 1}](/{concept_ids[i]}.md) - "
        f"{_one_line_summary(d.page_content)}"
        for i, d in enumerate(documents)
    )
    document_md = "\n".join(
        [
            "---",
            'type: "PDF Document"',
            f"title: {_yaml_quote(source_name)}",
            f"description: {_yaml_quote(doc_desc)}",
            f"resource: {_yaml_quote(source_name)}",
            'tags: ["pdf", "document"]',
            "status: stable",
            f"generated: {{ by: {_yaml_quote(OKF_PRODUCER)}, at: {generated_at} }}",
            f"sha256: {_yaml_quote(fingerprint)}",
            "---",
            "",
            f"# {source_name}",
            "",
            f"Extracted **{len(documents)}** text-bearing pages and "
            f"**{total_tables}** Markdown tables.",
            "",
            "## Pages",
            "",
            doc_links,
            "",
        ]
    )
    (bundle_dir / "document.md").write_text(document_md, encoding="utf-8")

    # Root index.md (OKF §8 & §12: root index.md may carry okf_version frontmatter)
    root_index_md = "\n".join(
        [
            "---",
            f'okf_version: "{OKF_VERSION}"',
            "---",
            "",
            "# Document",
            "",
            f"* [{source_name}](document.md) - Overview of {source_name} ({len(documents)} pages)",
            "",
            "# Directories",
            "",
            f"* [Pages](pages/) - Individual page concepts ({len(documents)} pages)",
            "",
        ]
    )
    (bundle_dir / "index.md").write_text(root_index_md, encoding="utf-8")
    return bundle_dir


# --------------------------------------------------------------------------- #
# Extraction and indexing
# --------------------------------------------------------------------------- #
def _tessdata_dir() -> Path | None:
    """Tesseract's language-data folder, or ``None`` when Tesseract is not installed."""
    try:
        return Path(pymupdf.get_tessdata())
    except Exception:
        return None


def ocr_available(settings: config.Settings | None = None) -> bool:
    """True when OCR is switched on and its engine can run.

    Tesseract needs every requested language installed. The remote engine counts as available
    once a server URL is configured (no network probe).
    """
    settings = settings or config.get_settings()
    if settings.ocr != config.OCR_AUTO:
        return False
    if settings.ocr_engine == config.OCR_ENGINE_REMOTE:
        return bool(settings.ocr_remote_url)
    tessdata = _tessdata_dir()
    return tessdata is not None and all(
        (tessdata / f"{language}.traineddata").is_file()
        for language in settings.ocr_language.split("+")
    )


def ocr_tag(settings: config.Settings) -> str:
    """The OCR setup that is part of an index's name ("" when OCR is not active).

    Tesseract indexes keep their original tag (the language), so existing indexes stay valid.
    Text read by a remote model differs from Tesseract's, so it must never share an index. The
    server address is part of the tag (hashed, as a URL may carry credentials), because another
    server can read the same model name differently.
    """
    if not ocr_available(settings):
        return ""
    if settings.ocr_engine == config.OCR_ENGINE_REMOTE:
        server = hashlib.sha256(settings.ocr_remote_url.encode()).hexdigest()[:8]
        return f"remote:{settings.ocr_remote_model}@{server}"
    return settings.ocr_language


def _page_has_images(page: pymupdf.Page) -> bool:
    try:
        return bool(page.get_images(full=False))
    except Exception:
        return False


def _ocr_page(page: pymupdf.Page, language: str) -> str | None:
    """Read a page's text with Tesseract; ``None`` if OCR failed (``""`` if it found no text)."""
    try:
        textpage = page.get_textpage_ocr(language=language, dpi=OCR_DPI, full=True)
        return page.get_text("text", textpage=textpage).strip()
    except Exception:
        logger.warning("OCR failed for page %d", page.number + 1, exc_info=True)
        return None


def _ocr_pages(
    doc: pymupdf.Document, indexes: Sequence[int], fingerprint: str, settings: config.Settings
) -> list[str | None]:
    """OCR the pages at ``indexes`` (``None`` for a failed page), reusing text read earlier."""
    tag = ocr_tag(settings)
    keys = [(fingerprint, tag, index) for index in indexes]
    with _ocr_cache_lock:
        texts: list[str | None] = [_ocr_cache.get(key) for key in keys]
    todo = [n for n, text in enumerate(texts) if text is None]
    if len(todo) < len(keys):
        logger.info("Reusing OCR text of %d page(s) read earlier", len(keys) - len(todo))
    if not todo:
        return texts
    logger.info("Running OCR on %d page(s)", len(todo))
    pages = [doc[indexes[n]] for n in todo]
    if settings.ocr_engine == config.OCR_ENGINE_REMOTE:
        fresh = ocr_remote.recognize_pages(pages, settings)
    else:
        fresh = [_ocr_page(page, settings.ocr_language) for page in pages]
    with _ocr_cache_lock:
        for n, text in zip(todo, fresh, strict=True):
            texts[n] = text
            if text is not None:
                _ocr_cache[keys[n]] = text
                _ocr_cache.move_to_end(keys[n])
        while len(_ocr_cache) > _OCR_CACHE_PAGES:
            _ocr_cache.popitem(last=False)
    return texts


def forget_ocr_text(fingerprint: str) -> None:
    """Drop the cached OCR text of a document (``fingerprint`` may be a prefix)."""
    with _ocr_cache_lock:
        for key in [key for key in _ocr_cache if key[0].startswith(fingerprint)]:
            del _ocr_cache[key]


def _no_text_message(settings: config.Settings, ocr_candidates: int, ocr_attempted: bool) -> str:
    base = "No selectable text was found in this PDF"
    if ocr_attempted:
        hint = (
            " Check that the OCR server is reachable."
            if settings.ocr_engine == config.OCR_ENGINE_REMOTE
            else ""
        )
        return f"{base}, and OCR could not read any text from its scanned pages.{hint}"
    if settings.ocr == config.OCR_OFF:
        return f"{base}. If it is a scanned document, enable OCR (PDFCHAT_OCR=auto)."
    if ocr_candidates == 0:
        return f"{base}, and it has no scanned page images to read with OCR."
    return (
        f"{base}. It looks scanned, but OCR is unavailable: install Tesseract with the "
        f"'{settings.ocr_language}' language data (see the README)."
    )


@dataclass(frozen=True)
class Extraction:
    """The pages of a PDF that have text, and the scanned pages OCR failed to read."""

    documents: list[Document]
    unread_pages: list[int] = field(default_factory=list)  # 1-based page numbers


def extract_pages(
    file_path: str, settings: config.Settings | None = None, *, fingerprint: str | None = None
) -> Extraction:
    """Extract one :class:`Document` per page that has text (with Markdown tables).

    Pages with no selectable text but with an image are read with OCR when it is available
    (``PDFCHAT_OCR``); those pages are tagged ``ocr: True`` in their metadata. A scanned page
    whose OCR *failed* (an error, not a page without text) is listed in ``unread_pages``.
    """
    settings = settings or config.get_settings()
    source = Path(file_path).name
    documents: list[Document] = []
    scanned: list[int] = []  # indexes of pages that have no text but do have an image
    unread: list[int] = []
    with pymupdf.open(file_path) as doc:
        for index, page in enumerate(doc):
            markdown_text, table_count = _extract_page_markdown(page)
            if markdown_text:
                documents.append(_page_document(source, index, markdown_text, table_count))
            elif _page_has_images(page):
                scanned.append(index)

        ocr_attempted = False
        if scanned and ocr_available(settings):
            if len(scanned) > settings.ocr_max_pages:
                raise PDFValidationError(
                    f"{len(scanned)} pages of this PDF need OCR; the limit is "
                    f"{settings.ocr_max_pages} (PDFCHAT_OCR_MAX_PAGES)."
                )
            ocr_attempted = True
            fingerprint = fingerprint or file_fingerprint(file_path)
            texts = _ocr_pages(doc, scanned, fingerprint, settings)
            for index, text in zip(scanned, texts, strict=True):
                if text is None:
                    unread.append(index + 1)
                elif text:
                    documents.append(_page_document(source, index, text, 0, ocr=True))
    if not documents:
        raise PDFValidationError(_no_text_message(settings, len(scanned), ocr_attempted))
    if unread:
        logger.warning(
            "OCR could not read %d of %d scanned page(s): %s",
            len(unread),
            len(scanned),
            ", ".join(map(str, unread)),
        )
    documents.sort(key=lambda d: d.metadata["page"])
    return Extraction(documents, unread)


def extract_documents(file_path: str, settings: config.Settings | None = None) -> list[Document]:
    """The pages of :func:`extract_pages` that have text (unread scanned pages are skipped)."""
    return extract_pages(file_path, settings).documents


def _page_document(
    source: str, index: int, text: str, table_count: int, *, ocr: bool = False
) -> Document:
    metadata: dict[str, Any] = {
        "source": source,
        "page": index,
        "table_count": table_count,
        "okf_concept": f"/pages/page_{index + 1:04d}",
    }
    if ocr:
        metadata["ocr"] = True
    return Document(page_content=text, metadata=metadata)


def _split(documents: list[Document], settings: config.Settings) -> list[Document]:
    """Split documents while preferring Markdown section and table boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


@dataclass(frozen=True)
class IndexResult:
    """A ready index and the scanned pages missing from it because OCR could not read them."""

    db: Chroma
    unread_pages: list[int] = field(default_factory=list)  # 1-based page numbers
    will_retry: bool = False  # the unread pages are tried again on the next request


class _PrecomputedEmbeddings(Embeddings):
    """Hands Chroma vectors computed earlier, so embedding happens outside ``_INDEX_LOCK``."""

    def __init__(self, texts: Sequence[str], vectors: Sequence[list[float]]) -> None:
        self._vectors = dict(zip(texts, vectors, strict=True))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - never queried
        raise NotImplementedError


def incomplete_marker(collection: str, settings: config.Settings) -> Path:
    """File recording the unread pages and build attempts of an incomplete index."""
    return settings.chroma_dir / "incomplete" / f"{collection}.json"


def _read_marker(path: Path) -> tuple[list[int], int] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [int(p) for p in data["unread_pages"]], int(data["attempts"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring an unreadable marker %s", path.name, exc_info=True)
        return None


def _write_marker(path: Path, unread_pages: list[int], attempts: int) -> None:
    if not unread_pages:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unread_pages": unread_pages, "attempts": attempts}), "utf-8")


def _build_lock(collection: str) -> threading.Lock:
    with _build_locks_guard:
        return _build_locks.setdefault(collection, threading.Lock())


def _open_collection(name: str, embeddings: Any, settings: config.Settings) -> Chroma:
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        # Documents are private: never send (even anonymous) usage telemetry.
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def _translate_indexing_error(exc: Exception) -> PDFChatbotError:
    translated = wrap_provider_error(exc, "index the PDF")
    if translated.__class__ is PDFChatbotError:
        translated = IndexingError(str(translated))
    return translated


def build_index(
    file_path: str,
    backend: str,
    openai_api_key: str | None = None,
    settings: config.Settings | None = None,
) -> IndexResult:
    """Load, split and index a PDF into a persistent Chroma collection.

    An existing, fully built collection for the same file content and embedding setup is
    reused. If building fails part-way, the half-built collection is removed so a later
    request cannot silently reuse it. Scanned pages that OCR could not read are reported in
    the result; such an index is rebuilt on later requests (up to ``MAX_INDEX_ATTEMPTS``
    builds) as soon as OCR reads at least one more of them.
    """
    settings = settings or config.get_settings()
    validate_pdf(file_path, settings)

    embedding_model = (
        settings.openai_embedding_model
        if backend == config.BACKEND_OPENAI
        else settings.local_embedding_model
    )
    fingerprint = file_fingerprint(file_path)
    name = get_collection_name(
        fingerprint,
        backend,
        embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        ocr=ocr_tag(settings),
    )
    embeddings = get_embeddings_model(backend, openai_api_key, settings)
    config.ensure_directories(settings)
    marker = incomplete_marker(name, settings)

    # Requests for the same index wait here for the first build instead of repeating it;
    # different PDFs are extracted and embedded in parallel.
    with _build_lock(name):
        with _INDEX_LOCK:
            db = _open_collection(name, embeddings, settings)
            exists = bool(db.get(limit=1)["ids"])
        previous = _read_marker(marker) if exists else None
        _mark_used(fingerprint, settings)  # also keeps retention from pruning it mid-build
        if exists and (previous is None or previous[1] >= MAX_INDEX_ATTEMPTS):
            logger.info("Reusing existing index %s", name)
            return IndexResult(db, previous[0] if previous else [])

        try:
            extraction = extract_pages(file_path, settings, fingerprint=fingerprint)
        except PDFChatbotError:
            if previous is None:
                raise
            extraction = None  # e.g. the OCR server is down: keep the incomplete index
        if extraction is None or (
            previous is not None and len(extraction.unread_pages) >= len(previous[0])
        ):
            # A retry that read no more pages (``previous`` is set: a first build re-raised
            # above): keep the index instead of embedding it again.
            attempts = previous[1] + 1
            _write_marker(marker, previous[0], attempts)
            logger.info("OCR still cannot read pages of %s (attempt %d)", name, attempts)
            if attempts >= MAX_INDEX_ATTEMPTS:
                forget_ocr_text(fingerprint)
            return IndexResult(db, previous[0], will_retry=attempts < MAX_INDEX_ATTEMPTS)

        # Splitting is deterministic: the chunks are split again below, after the bundle has
        # added its OKF metadata to the pages, and get the same texts.
        texts = [chunk.page_content for chunk in _split(extraction.documents, settings)]
        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as exc:
            logger.exception("Embedding failed for collection %s", name)
            raise _translate_indexing_error(exc) from exc

        logger.info(
            "Indexing %d chunks from %d pages into %s", len(texts), len(extraction.documents), name
        )
        with _INDEX_LOCK:
            if exists:  # replacing an incomplete index
                try:
                    db.delete_collection()
                except Exception:  # e.g. an operator deleted it meanwhile
                    logger.warning("Could not remove incomplete index %s", name, exc_info=True)
                db = _open_collection(name, embeddings, settings)
            build_okf_bundle(
                extraction.documents, get_bundle_dir(fingerprint, settings), fingerprint=fingerprint
            )
            # The bundle adds OKF metadata to the pages; the chunks must carry it too.
            chunks = _split(extraction.documents, settings)
            ids = [f"{fingerprint[:16]}-{i:05d}" for i in range(len(chunks))]
            writer = _open_collection(name, _PrecomputedEmbeddings(texts, vectors), settings)
            try:
                writer.add_documents(chunks, ids=ids)
            except Exception as exc:
                logger.exception("Indexing failed for collection %s", name)
                try:
                    db.delete_collection()
                except Exception:
                    logger.warning("Could not remove partial collection %s", name, exc_info=True)
                raise _translate_indexing_error(exc) from exc
        attempts = (previous[1] + 1) if previous else 1
        _write_marker(marker, extraction.unread_pages, attempts)
        will_retry = bool(extraction.unread_pages) and attempts < MAX_INDEX_ATTEMPTS
        if not will_retry:
            forget_ocr_text(fingerprint)
        return IndexResult(db, extraction.unread_pages, will_retry=will_retry)


def index_pdf(
    file_path: str,
    backend: str,
    openai_api_key: str | None = None,
    settings: config.Settings | None = None,
) -> Chroma:
    """:func:`build_index`, returning just the index."""
    return build_index(file_path, backend, openai_api_key, settings).db


def retrieve_context(
    db: Chroma, question: str, k: int, mode: str = config.RETRIEVAL_VECTOR
) -> list[Document]:
    """Return the ``k`` best chunks for ``question`` (vector search, or hybrid with BM25)."""
    try:
        return retrieval.search(db, question, k, mode)
    except Exception as exc:
        logger.exception("Retrieval failed")
        translated = wrap_provider_error(exc, "search the document")
        if translated.__class__ is PDFChatbotError:
            translated = RetrievalError(str(translated))
        raise translated from exc
