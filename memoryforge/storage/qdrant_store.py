"""
Qdrant vector store for MemoryForge.

This is a derived index from SQLite data. It can be rebuilt at any time.
Uses embedded mode for simplicity (no separate server required).
Supports variable embedding dimensions for different providers.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# Default dimension (for OpenAI text-embedding-3-small)
DEFAULT_EMBEDDING_DIMENSION = 1536


class QdrantStore:
    """Qdrant vector store for semantic search."""
    
    def __init__(
        self,
        storage_path: Path,
        project_id: Optional[UUID] = None,
        collection_name: Optional[str] = None,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ):
        """
        Initialize Qdrant in embedded mode.
        
        Args:
            storage_path: Path to store Qdrant data
            project_id: Project ID for scoped collection (v2). If provided,
                        collection name will be 'memories_{project_id[:8]}'
            collection_name: Override collection name (for backward compatibility)
            embedding_dimension: Dimension of embeddings (384 for local, 1536 for OpenAI)
        """
        self.storage_path = storage_path
        self.embedding_dimension = embedding_dimension
        self.project_id = project_id
        
        # v2: Per-project collection naming
        if collection_name:
            self.collection_name = collection_name
        elif project_id:
            # Use first 8 chars of project ID for shorter, readable names
            self.collection_name = f"memories_{str(project_id)[:8]}"
        else:
            self.collection_name = "memories"
        
        # Initialize embedded Qdrant client
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.storage_path))
        
        # Ensure collection exists with correct dimension
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            existing_dim = collection_info.config.params.vectors.size
            
            # Check if dimension matches
            if existing_dim != self.embedding_dimension:
                logger.warning(
                    f"Collection dimension mismatch: existing={existing_dim}, "
                    f"expected={self.embedding_dimension}. Recreating collection."
                )
                self.rebuild_collection()
            else:
                logger.debug(f"Collection '{self.collection_name}' already exists")
                
        except (UnexpectedResponse, ValueError, AttributeError):
            logger.info(f"Creating collection '{self.collection_name}' with dimension {self.embedding_dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
    
    def upsert(
        self,
        memory_id: UUID,
        embedding: list[float],
        memory_type: str,
        created_at: str,
    ) -> str:
        """
        Insert or update a vector in the collection.
        
        Returns the vector ID (same as memory_id string).
        """
        vector_id = str(memory_id)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        "memory_id": str(memory_id),
                        "memory_type": memory_type,
                        "created_at": created_at,
                    },
                )
            ],
        )
        
        logger.debug(f"Upserted vector for memory {memory_id}")
        return vector_id
    
    def delete(self, memory_id: UUID) -> bool:
        """Delete a vector from the collection."""
        vector_id = str(memory_id)
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[vector_id],
                ),
            )
            logger.debug(f"Deleted vector for memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector for memory {memory_id}: {e}")
            return False
    
    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_score: float = 0.5,
    ) -> list[dict]:
        """
        Search for similar vectors.
        
        Returns list of dicts with:
        - memory_id: UUID string
        - score: similarity score (0-1)
        - memory_type: type of memory
        - created_at: creation timestamp
        """
        # Build filter if memory_type is specified
        query_filter = None
        if memory_type:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="memory_type",
                        match=models.MatchValue(value=memory_type),
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=min_score,
        )
        
        return [
            {
                "memory_id": result.payload["memory_id"],
                "score": result.score,
                "memory_type": result.payload.get("memory_type"),
                "created_at": result.payload.get("created_at"),
            }
            for result in results
        ]
    
    def get_count(self) -> int:
        """Get the total number of vectors in the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0
    
    def rebuild_collection(self) -> None:
        """Delete and recreate the collection (for rebuilding index)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception:
            pass
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Recreated collection '{self.collection_name}' with dimension {self.embedding_dimension}")
    
    def close(self) -> None:
        """Close the Qdrant client connection."""
        self.client.close()
