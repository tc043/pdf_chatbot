import os
from typing import Generator, List, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import DocumentChunk
from app.services.embeddings import embedding_service

class LLMService:
    _local_model = None
    _local_tokenizer = None

    @classmethod
    def _load_local_model(cls):
        if cls._local_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Loading local MiniCPM5-1B model (approx. 2.2GB)...")
            cls._local_tokenizer = AutoTokenizer.from_pretrained(settings.LOCAL_MODEL_NAME)
            cls._local_model = AutoModelForCausalLM.from_pretrained(
                settings.LOCAL_MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu"
            ).eval()
        return cls._local_model, cls._local_tokenizer

    def retrieve_context(self, db: Session, document_ids: Any, question: str, model_type: str = "OpenAI") -> List[DocumentChunk]:
        """
        Calculates question embedding and retrieves the top-k document chunks using pgvector.
        Supports single or multiple document IDs.
        """
        if not isinstance(document_ids, list):
            document_ids = [document_ids]

        # Generate question embedding
        query_vector = embedding_service.generate_embedding(question, model_type)
        
        # SQLite compatibility fallback (in-memory cosine similarity calculation)
        if db.bind.dialect.name == "sqlite":
            import json
            all_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(document_ids)).all()
            
            def cosine_similarity(v1, v2):
                dot_prod = sum(x * y for x, y in zip(v1, v2))
                mag1 = sum(x * x for x in v1) ** 0.5
                mag2 = sum(x * x for x in v2) ** 0.5
                if mag1 > 0 and mag2 > 0:
                    return dot_prod / (mag1 * mag2)
                return 0.0

            scored_chunks = []
            for chunk in all_chunks:
                try:
                    c_vec = json.loads(chunk.embedding) if isinstance(chunk.embedding, str) else chunk.embedding
                    sim = cosine_similarity(query_vector, c_vec)
                except Exception:
                    sim = 0.0
                scored_chunks.append((sim, chunk))
                
            # Sort by similarity score descending
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in scored_chunks[:settings.RETRIEVER_K]]

        # Query PostgreSQL database (pgvector native calculation)
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id.in_(document_ids))
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(settings.RETRIEVER_K)
            .all()
        )
        return chunks

    def format_rag_prompt(self, question: str, chunks: List[DocumentChunk]) -> str:
        """
        Constructs context text and prompts the LLM.
        """
        context_text = ""
        for chunk in chunks:
            context_text += f"\n[Page {chunk.page_number} Context]:\n{chunk.content}\n"

        prompt = (
            f"You are a helpful assistant analyzing a PDF document. "
            f"Answer the question based only on the context provided below.\n"
            f"If the context does not contain the answer, explain what is missing, "
            f"do not make up information.\n\n"
            f"--- Context ---\n{context_text}\n"
            f"--- Question ---\n{question}"
        )
        return prompt

    def format_guide_prompt(self, guide_type: str, chunks: List[DocumentChunk]) -> str:
        """
        Constructs a prompt to generate specialized notebook guides.
        """
        context_text = ""
        for i, chunk in enumerate(chunks):
            context_text += f"\n[Document Chunk {i+1}]:\n{chunk.content}\n"
            
        if guide_type == "briefing":
            instructions = (
                "Generate a professional, high-quality Briefing Document based on the provided document contexts.\n"
                "Structure the response with clear markdown headings:\n"
                "- **Executive Summary**: A concise, high-level summary of the documents.\n"
                "- **Key Takeaways**: Bullet points highlighting the most important findings or facts.\n"
                "- **Detailed Analysis**: A breakdown of the primary themes, arguments, or data points.\n"
                "Be objective, highly readable, and thorough."
            )
        elif guide_type == "faq":
            instructions = (
                "Generate a Frequently Asked Questions (FAQ) list based on the provided document contexts.\n"
                "Structure the response with clear markdown headings:\n"
                "Provide 5-8 relevant questions that a reader would ask, followed by clear, detailed answers citing the source sections where appropriate.\n"
                "Format each FAQ item like:\n"
                "### Q: [Question]\n"
                "**A:** [Detailed Answer]"
            )
        elif guide_type == "study_guide":
            instructions = (
                "Generate a detailed Study Guide based on the provided document contexts.\n"
                "Structure the response with clear markdown headings:\n"
                "- **Key Concepts & Terms**: Define the core terminology and central theories/ideas.\n"
                "- **Discussion Questions**: 5 thought-provoking review questions to test comprehension.\n"
                "- **Summary Outline**: A hierarchical list of the main topics covered."
            )
        elif guide_type == "podcast":
            instructions = (
                "Generate an Audio Overview Podcast Transcript based on the provided document contexts.\n"
                "Create a lively, engaging dialogue between two hosts: Dan (a curious, energetic co-host) and Lisa (an expert analyst).\n"
                "They should discuss the documents, explain the key points in simple terms, and debate any interesting details.\n"
                "Structure the response in a script format, starting each speaking line with '**Dan:**' or '**Lisa:**'.\n"
                "Avoid other markdown formatting wrappers inside the dialog, just clean script lines so they can be parsed easily.\n"
                "Example:\n"
                "**Dan:** Hey Lisa, have you seen this new report?\n"
                "**Lisa:** Yes, Dan! It's actually got some fascinating insights about..."
            )
        else:
            instructions = "Summarize the key information in the provided document contexts."
            
        prompt = (
            f"You are a world-class assistant analyzing these documents. Here is the context content:\n"
            f"--- Context ---\n{context_text}\n\n"
            f"--- Task ---\n{instructions}\n\n"
            f"Generate the response now based ONLY on the context. Do not invent facts."
        )
        return prompt

    def generate_openai_stream(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> Generator[str, None, None]:
        """
        Streams completions from OpenAI.
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        chat = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )

        for chunk in chat.stream([HumanMessage(content=prompt)]):
            yield chunk.content

    def generate_local_stream(self, prompt: str, temperature: float = 0.9, max_tokens: int = 1024) -> Generator[str, None, None]:
        """
        Streams completions from the local MiniCPM model.
        """
        model, tokenizer = self._load_local_model()
        messages = [{"role": "user", "content": prompt}]
        
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt"
            ).to("cpu")
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cpu")

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

llm_service = LLMService()
