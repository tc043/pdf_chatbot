import os
import tempfile
import uuid
import fitz  # PyMuPDF
from celery import Celery
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Document, DocumentChunk
from app.services.storage import storage_service
from app.services.embeddings import embedding_service

# Initialize Celery app
celery_app = Celery("pdf_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Simple recursive-style character splitter for document chunking.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

@celery_app.task(name="process_pdf")
def process_pdf_task(document_id: str, model_type: str) -> bool:
    """
    Background worker task to download a PDF from S3, parse its content,
    generate vector embeddings, and save the chunks to the database.
    """
    db: Session = SessionLocal()
    try:
        doc_uuid = uuid.UUID(document_id) if isinstance(document_id, str) else document_id
        # Fetch document metadata
        doc = db.query(Document).filter(Document.id == doc_uuid).first()
        if not doc:
            print(f"Document {document_id} not found.")
            return False

        doc.status = "processing"
        db.commit()

        # Create temporary file to download PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_path = temp_pdf.name
            
        try:
            # Download file from S3
            storage_service.download_file(doc.s3_key, temp_path)

            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(temp_path)
            total_pages = len(pdf_doc)
            doc.page_count = total_pages

            # Iterate over pages and extract text
            for page_num in range(total_pages):
                page = pdf_doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    continue

                # Split page text into chunks
                chunks = split_text(text, settings.DEFAULT_CHUNK_SIZE, settings.DEFAULT_CHUNK_OVERLAP)

                # Store each chunk with its vector embedding
                for chunk_text in chunks:
                    embedding_vector = embedding_service.generate_embedding(chunk_text, model_type)
                    
                    chunk_obj = DocumentChunk(
                        document_id=doc.id,
                        page_number=page_num + 1, # 1-indexed pages for users
                        content=chunk_text,
                        embedding=embedding_vector
                    )
                    db.add(chunk_obj)
            
            pdf_doc.close()
            doc.status = "ready"
            db.commit()
            print(f"Successfully processed Document: {doc.filename}")
            return True

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        db.rollback()
        # Mark document as failed
        doc = db.query(Document).filter(Document.id == doc_uuid).first()
        if doc:
            doc.status = "failed"
            doc.error_message = str(e)
            db.commit()
        print(f"Error processing document {document_id}: {str(e)}")
        return False
    finally:
        db.close()
