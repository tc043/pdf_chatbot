from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, EmailStr, UUID4

# User Schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: UUID4
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

# Document Schemas
class DocumentOut(BaseModel):
    id: UUID4
    filename: str
    file_size: int
    page_count: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Chat Message Schemas
class Citation(BaseModel):
    page: int
    content: str
    chunk_id: str

class MessageCreate(BaseModel):
    content: str
    document_ids: Optional[List[UUID4]] = None

class MessageOut(BaseModel):
    id: UUID4
    role: str
    content: str
    citations: List[Dict[str, Any]] = []
    created_at: datetime

    class Config:
        from_attributes = True

# Conversation Schemas
class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"
    document_id: Optional[UUID4] = None

class ConversationOut(BaseModel):
    id: UUID4
    title: str
    document_id: Optional[UUID4] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = []

    class Config:
        from_attributes = True

# Note Schemas
class NoteCreate(BaseModel):
    title: Optional[str] = "Untitled Note"
    content: str

class NoteOut(BaseModel):
    id: UUID4
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
