"""PDF handling: validation, page rendering, text extraction, and vector indexing."""

from __future__ import annotations

import hashlib
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
# Index naming
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
# Extraction and indexing
# --------------------------------------------------------------------------- #
def extract_documents(file_path: str) -> list[Document]:
    """Extract one :class:`Document` per page that has selectable text."""
    source = Path(file_path).name
    documents: list[Document] = []
    with pymupdf.open(file_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(page_content=text, metadata={"source": source, "page": index})
                )
    if not documents:
        raise PDFValidationError(
            "No selectable text was found in this PDF. Scanned (image-only) PDFs need OCR, "
            "which is not supported."
        )
    return documents


def _split(documents: list[Document], settings: config.Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
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
