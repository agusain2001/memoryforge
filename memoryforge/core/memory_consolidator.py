"""
Memory Consolidator for MemoryForge v2.

Consolidates similar memories and manages memory lifecycle:
- Finds memory pairs above similarity threshold
- Creates NEW consolidated memories (archives originals)
- Provides rollback support
- Manages staleness tracking

Key Rules:
1. Threshold starts at 0.90 (configurable)
2. last_accessed updates on RETRIEVAL only, not listing
3. Consolidation creates NEW memory, archives originals
4. No auto-delete - user approval required
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID, uuid4

from memoryforge.config import Config
from memoryforge.models import Memory, MemoryType, MemorySource
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.embedding_factory import EmbeddingServiceProtocol as EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationSuggestion:
    """A suggestion for consolidating similar memories."""
    
    source_memories: List[Memory]
    similarity_score: float
    suggested_content: str
    memory_type: MemoryType
    
    @property
    def source_ids(self) -> List[UUID]:
        """Get IDs of source memories."""
        return [m.id for m in self.source_memories]
    
    @property
    def source_count(self) -> int:
        """Number of source memories."""
        return len(self.source_memories)


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    
    consolidated_memory: Memory
    archived_memories: List[Memory]
    version_ids: List[UUID]  # Version entries created for rollback
    
    @property
    def archived_count(self) -> int:
        """Number of archived memories."""
        return len(self.archived_memories)


class MemoryConsolidator:
    """
    Consolidates and manages memory lifecycle.
    
    Key behaviors:
    - Similarity detection using vector embeddings
    - Consolidation creates NEW memory, archives sources
    - Full rollback support via version history
    - Staleness tracking for unused memories
    """
    
    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        qdrant_store: QdrantStore,
        embedding_service: EmbeddingService,
        project_id: UUID,
        threshold: float = 0.90,
    ):
        """
        Initialize the memory consolidator.
        
        Args:
            sqlite_db: SQLite database instance
            qdrant_store: Qdrant vector store
            embedding_service: Embedding service for similarity
            project_id: Current project ID
            threshold: Similarity threshold (0.7-0.99, default 0.90)
        """
        self.db = sqlite_db
        self.qdrant = qdrant_store
        self.embedding_service = embedding_service
        self.project_id = project_id
        self.threshold = max(0.7, min(0.99, threshold))
    
    def find_similar_pairs(
        self,
        limit: int = 50,
    ) -> List[Tuple[Memory, Memory, float]]:
        """
        Find memory pairs above similarity threshold.
        
        Args:
            limit: Maximum pairs to return
            
        Returns:
            List of (memory1, memory2, similarity_score) tuples
        """
        # Get all confirmed, non-archived memories
        memories = self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=True,
            include_archived=False,
            limit=500,  # Reasonable limit for comparison
        )
        
        if len(memories) < 2:
            return []
        
        similar_pairs = []
        seen_pairs = set()
        
        for i, memory in enumerate(memories):
            # Get embedding for this memory
            try:
                embedding = self.embedding_service.generate(memory.content)
            except Exception as e:
                logger.warning(f"Failed to embed memory {memory.id}: {e}")
                continue
            
            # Search for similar memories
            results = self.qdrant.search(
                embedding,
                limit=10,
                min_score=self.threshold,
            )
            
            for result in results:
                other_id = result.id
                score = result.score
                
                # Skip self
                if other_id == memory.id:
                    continue
                
                # Skip already seen pairs
                pair_key = tuple(sorted([str(memory.id), str(other_id)]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                # Get the other memory
                other_memory = self.db.get_memory(other_id)
                if not other_memory or other_memory.is_archived:
                    continue
                
                similar_pairs.append((memory, other_memory, score))
        
        # Sort by similarity score
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs[:limit]
    
    def suggest_consolidations(
        self,
        max_suggestions: int = 10,
    ) -> List[ConsolidationSuggestion]:
        """
        Generate consolidation suggestions for user review.
        
        Args:
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of ConsolidationSuggestion objects
        """
        pairs = self.find_similar_pairs(limit=max_suggestions * 2)
        
        suggestions = []
        used_ids = set()
        
        for mem1, mem2, score in pairs:
            # Skip if either memory already used in another suggestion
            if mem1.id in used_ids or mem2.id in used_ids:
                continue
            
            used_ids.add(mem1.id)
            used_ids.add(mem2.id)
            
            # Generate suggested content (combine both)
            suggested_content = self._generate_consolidated_content([mem1, mem2])
            
            # Use the type of the more recent memory
            memory_type = mem1.type if mem1.created_at > mem2.created_at else mem2.type
            
            suggestions.append(ConsolidationSuggestion(
                source_memories=[mem1, mem2],
                similarity_score=score,
                suggested_content=suggested_content,
                memory_type=memory_type,
            ))
            
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions
    
    def _generate_consolidated_content(self, memories: List[Memory]) -> str:
        """
        Generate suggested consolidated content from multiple memories.
        
        This is a simple merge - user can edit before confirming.
        """
        if len(memories) == 1:
            return memories[0].content
        
        # Sort by creation date (oldest first)
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        # Combine content with separator
        parts = []
        for i, mem in enumerate(sorted_memories):
            if i == 0:
                parts.append(mem.content)
            else:
                # Check if content is significantly different
                if mem.content.strip() != parts[-1].strip():
                    parts.append(mem.content)
        
        if len(parts) == 1:
            return parts[0]
        
        # Join with appropriate separator
        return "\n\n".join(parts)
    
    def consolidate(
        self,
        source_ids: List[UUID],
        merged_content: str,
        memory_type: Optional[MemoryType] = None,
    ) -> ConsolidationResult:
        """
        Create new consolidated memory and archive sources.
        
        Steps:
        1. Validate all source memories exist and are not archived
        2. Save versions of source memories for rollback
        3. Create new memory with merged content
        4. Mark source memories as archived (is_archived=True)
        5. Set consolidated_into on sources
        
        Args:
            source_ids: IDs of memories to consolidate
            merged_content: Content for the new consolidated memory
            memory_type: Type for new memory (default: use first source's type)
            
        Returns:
            ConsolidationResult with new memory and archived sources
            
        Raises:
            ValueError: If any source memory doesn't exist or is archived
        """
        if len(source_ids) < 2:
            raise ValueError("Need at least 2 memories to consolidate")
        
        # 1. Validate and fetch source memories
        source_memories = []
        for mid in source_ids:
            memory = self.db.get_memory(mid)
            if not memory:
                raise ValueError(f"Memory not found: {mid}")
            if memory.is_archived:
                raise ValueError(f"Memory already archived: {mid}")
            if memory.project_id != self.project_id:
                raise ValueError(f"Memory belongs to different project: {mid}")
            source_memories.append(memory)
        
        # Determine memory type
        if memory_type is None:
            memory_type = source_memories[0].type
        
        # 2. Save versions of source memories for rollback
        version_ids = []
        for memory in source_memories:
            version_num = self.db.get_next_version_number(memory.id)
            version_id = self.db.save_memory_version(
                memory_id=memory.id,
                content=memory.content,
                version=version_num,
            )
            version_ids.append(version_id)
        
        # 3. Create new consolidated memory
        new_memory = Memory(
            id=uuid4(),
            project_id=self.project_id,
            content=merged_content,
            type=memory_type,
            source=MemorySource.MANUAL,  # Consolidation is manual action
            confirmed=True,  # Consolidations are auto-confirmed
            created_at=datetime.utcnow(),
        )
        
        self.db.save_memory(new_memory)
        
        # Get embedding and index in Qdrant
        try:
            embedding = self.embedding_service.generate(merged_content)
            self.qdrant.upsert(new_memory.id, embedding)
        except Exception as e:
            logger.warning(f"Failed to index consolidated memory: {e}")
        
        # 4 & 5. Archive source memories
        archived_memories = []
        for memory in source_memories:
            self.db.archive_memory(
                memory_id=memory.id,
                consolidated_into=new_memory.id,
            )
            
            # Remove from Qdrant (archived memories shouldn't be searched)
            try:
                self.qdrant.delete(memory.id)
            except Exception as e:
                logger.warning(f"Failed to remove archived memory from index: {e}")
            
            # Refresh to get updated state
            archived = self.db.get_memory(memory.id)
            if archived:
                archived_memories.append(archived)
        
        logger.info(
            f"Consolidated {len(source_memories)} memories into {new_memory.id}"
        )
        
        return ConsolidationResult(
            consolidated_memory=new_memory,
            archived_memories=archived_memories,
            version_ids=version_ids,
        )
    
    def rollback_consolidation(
        self,
        consolidated_memory_id: UUID,
    ) -> List[Memory]:
        """
        Restore archived memories and delete consolidated one.
        
        Args:
            consolidated_memory_id: ID of the consolidated memory to undo
            
        Returns:
            List of restored memories
            
        Raises:
            ValueError: If memory doesn't exist or has no archived sources
        """
        # Get consolidated memory
        consolidated = self.db.get_memory(consolidated_memory_id)
        if not consolidated:
            raise ValueError(f"Memory not found: {consolidated_memory_id}")
        
        # Find archived memories that were consolidated into this one
        sources = self.db.get_archived_memories(consolidated_memory_id)
        
        if not sources:
            raise ValueError(
                f"No archived sources found for memory {consolidated_memory_id}"
            )
        
        # Restore each source memory
        restored = []
        for memory in sources:
            self.db.restore_archived_memory(memory.id)
            
            # Re-index in Qdrant
            try:
                embedding = self.embedding_service.generate(memory.content)
                self.qdrant.upsert(memory.id, embedding)
            except Exception as e:
                logger.warning(f"Failed to re-index restored memory: {e}")
            
            # Refresh to get updated state
            updated = self.db.get_memory(memory.id)
            if updated:
                restored.append(updated)
        
        # Delete consolidated memory
        self.db.delete_memory(consolidated_memory_id)
        try:
            self.qdrant.delete(consolidated_memory_id)
        except Exception:
            pass  # May not exist in index
        
        logger.info(
            f"Rolled back consolidation: restored {len(restored)} memories"
        )
        
        return restored
    
    def mark_stale(
        self,
        memory_id: UUID,
        reason: str,
    ) -> bool:
        """
        Mark a memory as stale.
        
        Args:
            memory_id: Memory to mark stale
            reason: Why the memory is stale
            
        Returns:
            True if marked successfully
        """
        memory = self.db.get_memory(memory_id)
        if not memory:
            return False
        
        self.db.mark_stale(memory_id, reason)
        return True
    
    def clear_stale(self, memory_id: UUID) -> bool:
        """
        Clear the stale flag from a memory.
        
        Args:
            memory_id: Memory to clear stale from
            
        Returns:
            True if cleared successfully
        """
        memory = self.db.get_memory(memory_id)
        if not memory:
            return False
        
        self.db.clear_stale(memory_id)
        return True
    
    def get_stale_memories(self) -> List[Memory]:
        """
        Get all stale memories for the current project.
        
        Returns:
            List of stale memories
        """
        return self.db.get_stale_memories(self.project_id)
    
    def find_unused_memories(
        self,
        days_unused: int = 30,
    ) -> List[Memory]:
        """
        Find memories that haven't been accessed recently.
        
        Uses last_accessed field which is only updated on retrieval.
        
        Args:
            days_unused: Number of days since last access
            
        Returns:
            List of unused memories
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days_unused)
        
        # Get all confirmed memories
        memories = self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=True,
            include_archived=False,
            limit=1000,
        )
        
        unused = []
        for memory in memories:
            # If never accessed, check creation date
            if memory.last_accessed is None:
                if memory.created_at < cutoff:
                    unused.append(memory)
            elif memory.last_accessed < cutoff:
                unused.append(memory)
        
        return unused
    
    def get_consolidation_stats(self) -> dict:
        """
        Get statistics about consolidation state.
        
        Returns:
            Dict with consolidation statistics
        """
        memories = self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=True,
            include_archived=False,
            limit=1000,
        )
        
        archived = self.db.get_all_archived_memories(self.project_id)
        stale = self.db.get_stale_memories(self.project_id)
        
        # Count similar pairs
        pairs = self.find_similar_pairs(limit=100)
        
        return {
            "active_memories": len(memories),
            "archived_memories": len(archived),
            "stale_memories": len(stale),
            "similar_pairs": len(pairs),
            "threshold": self.threshold,
        }
    
    def suggest_stale_for_consolidation(
        self,
        days_unused: int = 30,
        min_similarity: float = 0.85,
    ) -> List[ConsolidationSuggestion]:
        """
        Find stale/unused memories that could be consolidated or archived.
        
        Integrates staleness tracking with consolidation suggestions by:
        1. Finding memories not accessed in `days_unused` days
        2. Checking if similar active memories exist
        3. Suggesting consolidation if similar, archival if unique
        
        Args:
            days_unused: Days since last access to consider stale
            min_similarity: Minimum similarity for consolidation suggestion
            
        Returns:
            List of ConsolidationSuggestion with stale memories
        """
        unused = self.find_unused_memories(days_unused)
        suggestions = []
        
        for memory in unused:
            # Find similar active memories
            try:
                embedding = self.embedding_service.generate(memory.content)
            except Exception as e:
                logger.warning(f"Failed to embed memory {memory.id}: {e}")
                continue
            
            results = self.qdrant.search(
                embedding,
                limit=5,
                min_score=min_similarity,
            )
            
            # Filter out self and archived
            similar_active = []
            for result in results:
                if result.id == memory.id:
                    continue
                other = self.db.get_memory(result.id)
                if other and not other.is_archived and not other.is_stale:
                    similar_active.append((other, result.score))
            
            if similar_active:
                # Suggest merging stale memory into active one
                best_match, score = similar_active[0]
                suggestions.append(ConsolidationSuggestion(
                    source_memories=[memory, best_match],
                    similarity_score=score,
                    suggested_content=f"[Consolidated from stale: {memory.id}]\n{best_match.content}\n\nAdditional context: {memory.content}",
                    memory_type=best_match.type,
                ))
            else:
                # Mark as stale if no similar active memories
                if not memory.is_stale:
                    self.mark_stale(
                        memory.id,
                        f"Not accessed in {days_unused} days and no similar active memories"
                    )
        
        return suggestions
    
    def auto_archive_stale(
        self,
        days_stale: int = 90,
        dry_run: bool = True,
    ) -> List[Memory]:
        """
        Auto-archive memories that have been stale for a long time.
        
        Args:
            days_stale: Days since marked stale to auto-archive
            dry_run: If True, only return candidates without archiving
            
        Returns:
            List of archived (or to-be-archived) memories
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days_stale)
        stale = self.db.get_stale_memories(self.project_id)
        
        # Filter to memories stale for long enough
        # Note: We'd need a stale_since field for proper implementation
        candidates = [m for m in stale if m.last_accessed and m.last_accessed < cutoff]
        
        if not dry_run:
            for memory in candidates:
                self.archive_memory(memory.id)
        
        return candidates

