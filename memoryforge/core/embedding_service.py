"""
OpenAI embedding service for MemoryForge.

Generates text embeddings using text-embedding-3-small model.
Includes retry logic and async-safe design.
"""

import asyncio
import logging
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)

# Default model for embeddings
DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


class EmbeddingService:
    """
    OpenAI embedding service with retry logic.
    
    This service handles embedding generation and includes:
    - Exponential backoff for retries
    - Rate limit handling
    - Batch embedding support
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize the embedding service."""
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._request_queue: asyncio.Queue = asyncio.Queue()
    
    def generate(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self._generate_with_retry(text)
    
    def _generate_with_retry(self, text: str) -> list[float]:
        """Generate embedding with exponential backoff retry."""
        last_error: Optional[Exception] = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                )
                return response.data[0].embedding
                
            except RateLimitError as e:
                last_error = e
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)
                
            except APIError as e:
                last_error = e
                if e.status_code and e.status_code >= 500:
                    # Server error, retry
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"API server error, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                else:
                    # Client error, don't retry
                    raise
        
        # All retries exhausted
        raise RuntimeError(
            f"Failed to generate embedding after {MAX_RETRIES} attempts: {last_error}"
        )
    
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
        
        # OpenAI supports batch embeddings
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fallback to individual embedding
            return [self.generate(text) for text in texts]
    
    async def generate_async(self, text: str) -> list[float]:
        """
        Async wrapper for embedding generation.
        
        This runs the synchronous OpenAI call in a thread pool
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, text)
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return EMBEDDING_DIMENSION
