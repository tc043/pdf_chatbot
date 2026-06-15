import os
import re
import hashlib
import fitz
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app import config

def render_pdf_page(file_path: str, page_num: int):
    """
    Renders a specific page of a PDF file as a PIL Image.
    Returns the PIL Image and the total number of pages in the PDF.
    """
    if not file_path or not os.path.exists(file_path):
        return None, 0
    
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    if page_num < 0:
        page_num = 0
    elif page_num >= total_pages:
        page_num = total_pages - 1
        
    page = doc[page_num]
    # Render page as image (200 DPI for high quality + fast loading)
    zoom = 200 / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    doc.close()
    
    return image, total_pages

def get_collection_name(file_path: str) -> str:
    """
    Generates a valid Chroma collection name from a PDF file path.
    """
    base = os.path.basename(file_path)
    # Chroma requires collection names:
    # - 3 to 63 characters
    # - Starts/ends with alphanumeric
    # - Contains only alphanumeric, underscores, hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
    
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "pdf_" + sanitized
    if not sanitized[-1].isalnum():
        sanitized = sanitized + "_pdf"
        
    if len(sanitized) > 63:
        h = hashlib.md5(base.encode('utf-8')).hexdigest()
        sanitized = sanitized[:45] + "_" + h[:15]
        
    while len(sanitized) < 3:
        sanitized += "_pdf"
        
    return sanitized

def get_embeddings_model(model_type: str, openai_api_key: str = None):
    """
    Returns the appropriate embeddings model based on the selected type.
    """
    if model_type == "OpenAI":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings.")
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        # Local model embeddings (MiniLM)
        return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_NAME)

def index_pdf(file_path: str, model_type: str, openai_api_key: str = None) -> Chroma:
    """
    Loads, splits, and indexes a PDF file into a persistent Chroma database.
    Reuses existing collection if the PDF has already been indexed.
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    collection_name = get_collection_name(file_path)
    # Suffix collection name with clean model type identifier to avoid mixing embedding dimensions
    model_suffix = "openai" if "openai" in model_type.lower() else "local"
    collection_name = f"{collection_name}_{model_suffix}"
    
    embeddings = get_embeddings_model(model_type, openai_api_key)
    
    # Check if the Chroma collection already exists with items
    db = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Check if collection is already populated
    try:
        col = db._client.get_collection(collection_name)
        if col.count() > 0:
            return db
    except Exception:
        # Collection doesn't exist, proceed to create it
        pass
        
    # Load and split PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.DEFAULT_CHUNK_SIZE,
        chunk_overlap=config.DEFAULT_CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    
    # Store in persistent database
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
        collection_name=collection_name
    )
    
    return db
