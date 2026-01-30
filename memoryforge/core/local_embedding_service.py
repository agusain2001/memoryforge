"""
Local embedding service using sentence-transformers.

Provides free, local embeddings without requiring an API key.
Default model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default local model - good balance of speed and quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbeddingService:
    """
    Local embedding service using sentence-transformers.
    
    Benefits:
    - 100% free (no API costs)
    - Works offline
    - Fast (runs locally)
    - Privacy-preserving (no data sent externally)
    
    Tradeoffs:
    - Smaller embedding dimension (384 vs 1536)
    - Slightly lower quality than OpenAI
    - First load downloads the model (~90MB)
    """
    
    _model = None  # Class-level cache for the model
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the local embedding service.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        self.model_name = model_name
        self._dimension: Optional[int] = None
    
    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        if LocalEmbeddingService._model is None or self._dimension is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install memoryforge[local]"
                )
            
            logger.info(f"Loading local embedding model: {self.model_name}")
            LocalEmbeddingService._model = SentenceTransformer(self.model_name)
            self._dimension = LocalEmbeddingService._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        
        return LocalEmbeddingService._model
    
    def generate(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    
    async def generate_async(self, text: str) -> list[float]:
        """
        Async wrapper for embedding generation.
        
        Runs synchronously since local models are fast enough.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, text)
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            self._get_model()
        return self._dimension or 384  # Default for MiniLM
