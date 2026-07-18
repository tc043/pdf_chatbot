import os
import uuid
import tempfile
import fitz  # PyMuPDF
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response, BackgroundTasks, status, Query, Request
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.core.config import settings
from app.core.database import get_db
from app.routers.auth import get_current_user
from app.models import User, Document
from app.schemas import DocumentOut
from app.services.storage import storage_service
from app.worker import process_pdf_task

def get_current_user_from_token(
    request: Request,
    token: str = Query(None),
    db: Session = Depends(get_db)
) -> User:
    # Try header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    try:
        user_uuid = uuid.UUID(user_id)
    except (ValueError, TypeError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_uuid).first()
    if user is None:
        raise credentials_exception
    return user

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    model_type: str = "OpenAI",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Uploads a PDF, creates DB record, and enqueues RAG chunking in Celery background task.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    file_size = len(file_bytes)
    
    # Write to local temp file to read page count via PyMuPDF before backgrounding
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(file_bytes)
        temp_path = temp_pdf.name

    try:
        pdf_doc = fitz.open(temp_path)
        page_count = len(pdf_doc)
        pdf_doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDF document structure: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Prepare storage keys
    doc_id = uuid.uuid4()
    s3_key = f"{current_user.id}/{doc_id}_{file.filename}"

    # Upload to S3/MinIO
    try:
        storage_service.upload_file(file_bytes, s3_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file: {str(e)}")

    # Save metadata in DB
    db_doc = Document(
        id=doc_id,
        user_id=current_user.id,
        filename=file.filename,
        s3_key=s3_key,
        file_size=file_size,
        page_count=page_count,
        status="processing"
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # Enqueue task in Celery worker
    process_pdf_task.delay(str(db_doc.id), model_type)

    return db_doc

@router.get("/", response_model=List[DocumentOut])
def list_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Lists all documents owned by the current tenant.
    """
    return db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.created_at.desc()).all()

@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Retrieves the indexing status of a specific document.
    """
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@router.get("/{document_id}/pages/{page_num}")
def get_pdf_page_image(
    document_id: uuid.UUID,
    page_num: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """
    Downloads the PDF from S3 and renders the requested page as a PNG image.
    Used for web document viewers without requiring client-side PDF rendering.
    """
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if page_num < 1 or page_num > doc.page_count:
        raise HTTPException(status_code=400, detail="Page number out of range.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_path = temp_pdf.name

    try:
        storage_service.download_file(doc.s3_key, temp_path)
        pdf_doc = fitz.open(temp_path)
        
        # page_num is 1-indexed from client, convert to 0-index for PyMuPDF
        page = pdf_doc[page_num - 1]
        zoom = 150 / 72  # standard 150 DPI zoom factor
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        png_bytes = pix.tobytes("png")
        pdf_doc.close()
        
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering page: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
