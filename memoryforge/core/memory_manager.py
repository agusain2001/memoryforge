"""
Memory Manager for MemoryForge.

Central component handling all memory lifecycle operations:
- Create (unconfirmed)
- Confirm (finalize and index)
- Update
- Delete
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from memoryforge.models import Memory, MemoryCreate, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.embedding_service import EmbeddingService
from memoryforge.core.validation import ValidationLayer, ValidationError

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory lifecycle manager.
    
    Coordinates between SQLite (source of truth) and Qdrant (vector index).
    Enforces the confirmation flow for human-in-the-loop memory capture.
    """
    
    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        qdrant_store: QdrantStore,
        embedding_service: EmbeddingService,
        project_id: UUID,
    ):
        """Initialize the memory manager."""
        self.db = sqlite_db
        self.vector_store = qdrant_store
        self.embedding_service = embedding_service
        self.project_id = project_id
        self.validation = ValidationLayer()
    
    def create_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: MemorySource = MemorySource.MANUAL,
        auto_confirm: bool = False,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """
        Create a new memory (unconfirmed by default).
        
        Unconfirmed memories are stored in SQLite but NOT indexed in Qdrant.
        They are excluded from retrieval until confirmed.
        
        Args:
            content: The memory content
            memory_type: Type of memory (stack, decision, etc.)
            source: Source of the memory
            auto_confirm: If True, automatically confirm and index
            metadata: Optional additional metadata
            
        Returns:
            The created Memory object
        """
        # Validate and sanitize
        content = self.validation.sanitize_content(content)
        memory_create = MemoryCreate(
            content=content,
            type=memory_type,
            source=source,
            metadata=metadata or {},
        )
        self.validation.validate_memory_create(memory_create)
        
        # Create memory object
        memory = Memory(
            content=content,
            type=memory_type,
            source=source,
            project_id=self.project_id,
            confirmed=False,
            metadata=metadata or {},
        )
        
        # Store in SQLite (source of truth)
        self.db.create_memory(memory)
        logger.info(f"Created memory {memory.id} (unconfirmed)")
        
        # Auto-confirm if requested
        if auto_confirm:
            self.confirm_memory(memory.id)
            memory.confirmed = True
        
        return memory
    
    def confirm_memory(self, memory_id: UUID) -> bool:
        """
        Confirm a memory, making it eligible for retrieval.
        
        This:
        1. Marks the memory as confirmed in SQLite
        2. Generates an embedding
        3. Indexes the embedding in Qdrant
        
        Args:
            memory_id: The ID of the memory to confirm
            
        Returns:
            True if successful, False otherwise
        """
        # Get the memory
        memory = self.db.get_memory(memory_id)
        if memory is None:
            logger.error(f"Memory {memory_id} not found")
            return False
        
        if memory.confirmed:
            logger.debug(f"Memory {memory_id} already confirmed")
            return True
        
        try:
            # Generate embedding
            embedding = self.embedding_service.generate(memory.content)
            
            # Index in Qdrant
            vector_id = self.vector_store.upsert(
                memory_id=memory.id,
                embedding=embedding,
                memory_type=memory.type.value,
                created_at=memory.created_at.isoformat(),
            )
            
            # Save embedding reference
            self.db.save_embedding_reference(memory.id, vector_id)
            
            # Mark as confirmed in SQLite
            self.db.confirm_memory(memory_id)
            
            logger.info(f"Confirmed and indexed memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to confirm memory {memory_id}: {e}")
            return False
    
    def get_memory(self, memory_id: UUID) -> Optional[Memory]:
        """Get a memory by ID."""
        return self.db.get_memory(memory_id)
    
    def list_memories(
        self,
        confirmed_only: bool = True,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """
        List memories for the current project.
        
        Args:
            confirmed_only: Only return confirmed memories
            memory_type: Filter by memory type
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of Memory objects
        """
        return self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=confirmed_only,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        )
    
    def update_memory(self, memory_id: UUID, content: str) -> bool:
        """
        Update memory content.
        
        If the memory is confirmed, this also updates the embedding.
        
        Args:
            memory_id: The ID of the memory to update
            content: New content
            
        Returns:
            True if successful
        """
        # Validate and sanitize
        content = self.validation.sanitize_content(content)
        if len(content) < 1:
            raise ValidationError("Content cannot be empty")
        
        # Get existing memory
        memory = self.db.get_memory(memory_id)
        if memory is None:
            logger.error(f"Memory {memory_id} not found")
            return False
        
        # Update in SQLite
        if not self.db.update_memory(memory_id, content):
            return False
        
        # If confirmed, update the embedding
        if memory.confirmed:
            try:
                embedding = self.embedding_service.generate(content)
                self.vector_store.upsert(
                    memory_id=memory.id,
                    embedding=embedding,
                    memory_type=memory.type.value,
                    created_at=memory.created_at.isoformat(),
                )
                logger.info(f"Updated memory and embedding for {memory_id}")
            except Exception as e:
                logger.error(f"Failed to update embedding for {memory_id}: {e}")
                # Memory content is updated, but embedding failed
                # This is logged but not treated as a failure
        
        return True
    
    def delete_memory(self, memory_id: UUID) -> bool:
        """
        Delete a memory and its associated embedding.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if successful
        """
        # Get the memory to check if it's confirmed
        memory = self.db.get_memory(memory_id)
        if memory is None:
            logger.error(f"Memory {memory_id} not found")
            return False
        
        # If confirmed, delete from Qdrant first
        if memory.confirmed:
            self.vector_store.delete(memory_id)
        
        # Delete from SQLite (also removes embedding reference)
        result = self.db.delete_memory(memory_id)
        
        if result:
            logger.info(f"Deleted memory {memory_id}")
        
        return result
    
    def get_memory_count(self, confirmed_only: bool = True) -> int:
        """Get the count of memories for the current project."""
        return self.db.get_memory_count(self.project_id, confirmed_only)
    
    def get_unconfirmed_memories(self) -> list[Memory]:
        """Get all unconfirmed memories (pending confirmation)."""
        return self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=False,
        )
