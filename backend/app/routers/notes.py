from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import List
import uuid

from app.core.database import get_db
from app.routers.auth import get_current_user
from app.models import User, Note
from app.schemas import NoteOut, NoteCreate

router = APIRouter(prefix="/notes", tags=["notes"])

@router.get("/", response_model=List[NoteOut])
def list_notes(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Lists all notes owned by the current authenticated user.
    """
    return db.query(Note).filter(Note.user_id == current_user.id).order_by(Note.updated_at.desc()).all()

@router.post("/", response_model=NoteOut, status_code=status.HTTP_201_CREATED)
def create_note(
    note_in: NoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Creates a new note (custom or pinned chat response).
    """
    note = Note(
        user_id=current_user.id,
        title=note_in.title,
        content=note_in.content
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note

@router.put("/{note_id}", response_model=NoteOut)
def update_note(
    note_id: uuid.UUID,
    note_in: NoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Updates an existing note's title and content.
    """
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == current_user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    note.title = note_in.title
    note.content = note_in.content
    db.commit()
    db.refresh(note)
    return note

@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(
    note_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deletes a note.
    """
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == current_user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    db.delete(note)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
