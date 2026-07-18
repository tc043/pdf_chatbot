import json
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db, SessionLocal
from app.routers.auth import get_current_user
from app.models import User, Conversation, Message, Document
from app.schemas import ConversationOut, ConversationCreate, ConversationDetail, MessageCreate
from app.services.llm import llm_service

router = APIRouter(prefix="/chats", tags=["chats"])

@router.post("/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    conv_in: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Creates a new conversation thread linked optionally to a document.
    """
    conversation = Conversation(
        user_id=current_user.id,
        title=conv_in.title,
        document_id=conv_in.document_id
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Lists all active conversation sessions for the current authenticated user.
    """
    return db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.updated_at.desc()).all()

@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns full message transcript and citations for a conversation thread.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation session not found.")
    return conversation

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deletes a conversation history thread.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    
    db.delete(conversation)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/conversations/{conversation_id}/messages")
def post_chat_message(
    conversation_id: uuid.UUID,
    message_in: MessageCreate,
    model_type: str = "OpenAI",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Executes a vector search query against the conversation's document(s),
    streams back the LLM answer, and logs the exchange to history on completion.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation thread not found.")

    doc_ids = []
    if message_in.document_ids:
        doc_ids = message_in.document_ids
    elif conversation.document_id:
        doc_ids = [conversation.document_id]

    if not doc_ids:
        raise HTTPException(status_code=400, detail="No source documents selected for this chat.")

    # Retrieve context from DB using pgvector
    chunks = llm_service.retrieve_context(db, doc_ids, message_in.content, model_type)
    
    # Save the user query to the database
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=message_in.content
    )
    db.add(user_msg)
    db.commit()

    # Prep document citations to stream to the client first
    citations = [{"page": chunk.page_number, "content": chunk.content, "chunk_id": str(chunk.id)} for chunk in chunks]

    # Return server-sent stream
    def response_stream():
        # Step 1: Send references first
        yield f"event: citations\ndata: {json.dumps({'chunks': citations})}\n\n"
        
        prompt = llm_service.format_rag_prompt(message_in.content, chunks)
        full_response = ""

        # Step 2: Stream tokens from LLM
        try:
            if model_type == "OpenAI":
                stream = llm_service.generate_openai_stream(prompt)
            else:
                stream = llm_service.generate_local_stream(prompt)

            for token in stream:
                full_response += token
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            return

        # Step 3: Write completed assistant response to DB
        db_gen = SessionLocal()
        try:
            assistant_msg = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response,
                citations=citations
            )
            db_gen.add(assistant_msg)
            
            # Touch conversation to update updated_at timestamp
            db_conversation = db_gen.query(Conversation).filter(Conversation.id == conversation.id).first()
            if db_conversation:
                db_conversation.title = message_in.content[:40] + ("..." if len(message_in.content) > 40 else "")
            
            db_gen.commit()
        except Exception as e:
            db_gen.rollback()
            print(f"Error logging response: {str(e)}")
        finally:
            db_gen.close()

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")


from pydantic import BaseModel
class GuideRequest(BaseModel):
    document_ids: List[uuid.UUID]
    guide_type: str  # briefing, faq, study_guide, podcast

@router.post("/generate-guide")
def generate_guide(
    req: GuideRequest,
    model_type: str = "OpenAI",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Gathers key chunks from all selected documents and generates a styled guide
    (Briefing Document, FAQ list, Study Guide, or audio script podcast) as a stream.
    """
    doc_ids = req.document_ids
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No source documents selected.")

    # Fetch top representative chunks from each document
    chunks = []
    for doc_id in doc_ids:
        # Check permissions
        doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
        if not doc:
            continue
        
        # Query representative summary content chunks
        doc_chunks = llm_service.retrieve_context(
            db, 
            [doc_id], 
            "summary of key concepts, main outline, briefing takeaways, core terms", 
            model_type
        )
        chunks.extend(doc_chunks)

    if not chunks:
        raise HTTPException(status_code=404, detail="No readable content found in selected documents.")

    prompt = llm_service.format_guide_prompt(req.guide_type, chunks)

    def guide_stream():
        try:
            if model_type == "OpenAI":
                stream = llm_service.generate_openai_stream(prompt)
            else:
                stream = llm_service.generate_local_stream(prompt)

            for token in stream:
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            return

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(guide_stream(), media_type="text/event-stream")
