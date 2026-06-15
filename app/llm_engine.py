import os
from threading import Thread
from typing import Generator
from app import config

class LocalModelManager:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loading = False
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def load_model(self, progress_callback=None):
        """
        Loads the MiniCPM5-1B-SFT model and tokenizer on CPU.
        Uses singleton pattern to ensure it's loaded only once.
        """
        if self.model is not None:
            return True
            
        if self.is_loading:
            if progress_callback:
                progress_callback("Model loading is already in progress...")
            return False
            
        self.is_loading = True
        try:
            if progress_callback:
                progress_callback("Importing PyTorch and Transformers...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            if progress_callback:
                progress_callback("Downloading / loading model & tokenizer (approx. 2.2GB)...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LOCAL_MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu"
            ).eval()
            
            if progress_callback:
                progress_callback("Model loaded successfully!")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error loading local model: {str(e)}")
            raise e
        finally:
            self.is_loading = False

def format_rag_prompt(question: str, context_docs) -> str:
    """
    Format the RAG context and question into a prompt.
    """
    context_text = ""
    for i, doc in enumerate(context_docs):
        page = doc.metadata.get("page", 0) + 1
        context_text += f"\n[Page {page} Context]:\n{doc.page_content}\n"
    
    prompt = (
        f"You are a helpful assistant analyzing a PDF document. "
        f"Answer the question based only on the context provided below.\n"
        f"If the context does not contain the answer, explain what is missing, "
        f"do not make up information.\n\n"
        f"--- Context ---\n{context_text}\n"
        f"--- Question ---\n{question}"
    )
    return prompt

def generate_openai_stream(
    prompt: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> Generator[str, None, None]:
    """
    Generates a streaming response from OpenAI API.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    chat = ChatOpenAI(
        openai_api_key=api_key,
        model=config.OPENAI_MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True
    )
    
    for chunk in chat.stream([HumanMessage(content=prompt)]):
        yield chunk.content

def generate_local_stream(
    prompt: str,
    enable_thinking: bool = True,
    temperature: float = 0.9,
    max_tokens: int = 1024,
    progress_callback=None
) -> Generator[str, None, None]:
    """
    Generates a streaming response from the local MiniCPM5-1B-SFT model.
    """
    manager = LocalModelManager.get_instance()
    if manager.model is None or manager.tokenizer is None:
        manager.load_model(progress_callback)
        
    tokenizer = manager.tokenizer
    model = manager.model
    
    messages = [{"role": "user", "content": prompt}]
    
    # Try using enable_thinking argument (new in MiniCPM5-1B)
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt"
        ).to("cpu")
    except TypeError:
        # Fallback if the tokenizer doesn't support the keyword arg
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cpu")
        
    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # inputs is a BatchEncoding dict, so unpack it to provide input_ids and attention_mask
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=0.95
    )
    
    # Run generation in a separate thread so it doesn't block the UI generator loop
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text
