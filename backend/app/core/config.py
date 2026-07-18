import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "Enterprise PDF Chatbot"
    API_V1_STR: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str = "postgresql://admin:supersecretpassword@postgres/pdf_rag"
    
    # Redis Settings
    REDIS_URL: str = "redis://redis:6379/0"
    
    # S3 / MinIO Settings
    S3_ENDPOINT: str = "http://minio:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadminpassword"
    S3_BUCKET: str = "pdf-uploads"
    
    # Security Settings
    JWT_SECRET: str = "change-this-to-a-very-secure-secret-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 Hours
    
    # LLM Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    LOCAL_MODEL_NAME: str = "openbmb/MiniCPM5-1B-SFT"
    LOCAL_EMBEDDING_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Settings
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    RETRIEVER_K: int = 3

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
