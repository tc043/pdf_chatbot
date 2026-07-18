import numpy as np
from typing import List
from app.core.config import settings

class EmbeddingService:
    _local_model = None

    @classmethod
    def _get_local_model(cls):
        if cls._local_model is None:
            from sentence_transformers import SentenceTransformer
            # Load local embeddings model lazily
            cls._local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_NAME)
        return cls._local_model

    def generate_embedding(self, text: str, model_type: str = "OpenAI") -> List[float]:
        """
        Generates a 1536-dimensional vector embedding for the input text.
        Pads local embeddings with zeros to match standard 1536 dimension.
        """
        if model_type == "OpenAI" and settings.OPENAI_API_KEY:
            try:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-3-small" # 1536 dimensions
                )
                return embeddings.embed_query(text)
            except Exception as e:
                # Fallback to local if OpenAI fails
                print(f"OpenAI embedding failed, falling back to local: {str(e)}")
        
        # Local model embedding
        model = self._get_local_model()
        vector = model.encode(text).tolist()
        return self._pad_vector(vector, 1536)

    def _pad_vector(self, vec: List[float], target_dim: int = 1536) -> List[float]:
        """
        Pads a vector with zeros to reach the target dimension.
        """
        if len(vec) >= target_dim:
            return vec[:target_dim]
        padded = np.zeros(target_dim)
        padded[:len(vec)] = vec
        return padded.tolist()

embedding_service = EmbeddingService()
