"""
Embedding factory for MemoryForge.

Creates the appropriate embedding service based on configuration.
Supports both OpenAI (paid) and local sentence-transformers (free).
"""

import logging
from typing import Protocol, Union

from memoryforge.config import Config, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingServiceProtocol(Protocol):
    """Protocol defining the embedding service interface."""
    
    def generate(self, text: str) -> list[float]: ...
    def generate_batch(self, texts: list[str]) -> list[list[float]]: ...
    async def generate_async(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...


def create_embedding_service(config: Config) -> EmbeddingServiceProtocol:
    """
    Create an embedding service based on configuration.
    
    Args:
        config: MemoryForge configuration
        
    Returns:
        An embedding service instance (OpenAI or Local)
    """
    if config.embedding_provider == EmbeddingProvider.OPENAI:
        return _create_openai_service(config)
    else:
        return _create_local_service(config)


def _create_openai_service(config: Config) -> EmbeddingServiceProtocol:
    """Create OpenAI embedding service."""
    from memoryforge.core.embedding_service import EmbeddingService
    
    if not config.openai_api_key:
        raise ValueError(
            "OpenAI API key is required for OpenAI embeddings. "
            "Set it in config or use 'local' embedding provider instead."
        )
    
    logger.info(f"Using OpenAI embeddings: {config.openai_embedding_model}")
    return EmbeddingService(
        api_key=config.openai_api_key,
        model=config.openai_embedding_model,
    )


def _create_local_service(config: Config) -> EmbeddingServiceProtocol:
    """Create local sentence-transformers embedding service."""
    from memoryforge.core.local_embedding_service import LocalEmbeddingService
    
    logger.info(f"Using local embeddings: {config.local_embedding_model}")
    return LocalEmbeddingService(model_name=config.local_embedding_model)


def get_embedding_dimension(provider: EmbeddingProvider, model: str = "") -> int:
    """
    Get the embedding dimension for a provider/model combination.
    
    This is needed for Qdrant collection setup before the service is created.
    """
    if provider == EmbeddingProvider.OPENAI:
        # OpenAI text-embedding-3-small/large both use 1536
        return 1536
    else:
        # Common local model dimensions
        local_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
            "multi-qa-MiniLM-L6-cos-v1": 384,
        }
        return local_dimensions.get(model, 384)
