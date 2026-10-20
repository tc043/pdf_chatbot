"""PDF handling: validation, page rendering, OKF v0.2 bundle extraction, and vector indexing."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from app import config
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
OKF_PRODUCER = "pdf_chatbot/pymupdf"

# Indexing is CPU/IO heavy and Chroma's persistent client is not designed for
# concurrent writers to the same collection, so index builds are serialised.
_INDEX_LOCK = threading.Lock()


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
) -> str:
    """Deterministic Chroma collection name for one (document, embedding setup) pair.

    Keyed on the document *content* rather than its filename, so two different
    PDFs that share a filename never collide and an edited PDF is re-indexed.
    The embedding model and chunking parameters are part of the key so changing
    them never reuses vectors of the wrong dimension or granularity.
    Result is at most 60 characters and uses only ``[a-z0-9_]``.
    """
    kind = "openai" if backend == config.BACKEND_OPENAI else "local"
    setup = f"{embedding_model}|{chunk_size}|{chunk_overlap}"
    tag = hashlib.sha256(setup.encode("utf-8")).hexdigest()[:8]
    return f"pdf_{fingerprint[:40]}_{kind}_{tag}"


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
def extract_documents(file_path: str) -> list[Document]:
    """Extract one :class:`Document` per page that has selectable text (with Markdown tables)."""
    source = Path(file_path).name
    documents: list[Document] = []
    with pymupdf.open(file_path) as doc:
        for index, page in enumerate(doc):
            markdown_text, table_count = _extract_page_markdown(page)
            if markdown_text:
                documents.append(
                    Document(
                        page_content=markdown_text,
                        metadata={
                            "source": source,
                            "page": index,
                            "table_count": table_count,
                            "okf_concept": f"/pages/page_{index + 1:04d}",
                        },
                    )
                )
    if not documents:
        raise PDFValidationError(
            "No selectable text was found in this PDF. Scanned (image-only) PDFs need OCR, "
            "which is not supported."
        )
    return documents


def _split(documents: list[Document], settings: config.Settings) -> list[Document]:
    """Split documents while preferring Markdown section and table boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def index_pdf(
    file_path: str,
    backend: str,
    openai_api_key: str | None = None,
    settings: config.Settings | None = None,
) -> Chroma:
    """Load, split and index a PDF into a persistent Chroma collection.

    An existing, fully built collection for the same file content and embedding
    setup is reused. If building fails part-way, the half-built collection is
    removed so a later request cannot silently reuse it.
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
        fingerprint, backend, embedding_model, settings.chunk_size, settings.chunk_overlap
    )
    embeddings = get_embeddings_model(backend, openai_api_key, settings)

    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma

    config.ensure_directories(settings)
    with _INDEX_LOCK:
        db = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
            # Documents are private: never send (even anonymous) usage telemetry.
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        if db.get(limit=1)["ids"]:
            logger.info("Reusing existing index %s", name)
            return db

        documents = extract_documents(file_path)
        bundle_dir = get_bundle_dir(fingerprint, settings)
        build_okf_bundle(documents, bundle_dir, fingerprint=fingerprint)

        chunks = _split(documents, settings)
        ids = [f"{fingerprint[:16]}-{i:05d}" for i in range(len(chunks))]
        logger.info("Indexing %d chunks from %d pages into %s", len(chunks), len(documents), name)
        try:
            db.add_documents(chunks, ids=ids)
        except Exception as exc:
            logger.exception("Indexing failed for collection %s", name)
            try:
                db.delete_collection()
            except Exception:
                logger.warning("Could not remove partial collection %s", name, exc_info=True)
            translated = wrap_provider_error(exc, "index the PDF")
            if translated.__class__ is PDFChatbotError:
                translated = IndexingError(str(translated))
            raise translated from exc
    return db


def retrieve_context(db: Chroma, question: str, k: int) -> list[Document]:
    """Return the ``k`` chunks most similar to ``question``."""
    try:
        return db.similarity_search(question, k=k)
    except Exception as exc:
        logger.exception("Retrieval failed")
        translated = wrap_provider_error(exc, "search the document")
        if translated.__class__ is PDFChatbotError:
            translated = RetrievalError(str(translated))
        raise translated from exc
