"""
Retrieval Engine for MemoryForge.

Handles semantic search and retrieval with:
- Query normalization
- Type/recency filtering
- Re-ranking
- Explainability metadata
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from memoryforge.models import Memory, MemoryType, SearchResult
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.embedding_factory import EmbeddingServiceProtocol as EmbeddingService
from memoryforge.core.validation import ValidationLayer

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Search and retrieval engine for memories.
    
    Provides semantic search with filtering, re-ranking, and explainability.
    Enforces max result count (default 5) for quality over quantity.
    """
    
    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        qdrant_store: QdrantStore,
        embedding_service: EmbeddingService,
        project_id: UUID,
        max_results: int = 5,
        min_score: float = 0.5,
    ):
        """Initialize the retrieval engine."""
        self.db = sqlite_db
        self.vector_store = qdrant_store
        self.embedding_service = embedding_service
        self.project_id = project_id
        self.max_results = max_results
        self.min_score = min_score
        self.validation = ValidationLayer()
    
    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        exclude_stale: bool = False,
    ) -> list[SearchResult]:
        """
        Search for relevant memories using semantic similarity.
        
        Args:
            query: The search query
            memory_type: Optional filter by memory type
            limit: Max results (defaults to max_results)
            min_score: Minimum similarity score (defaults to min_score)
            
        Returns:
            List of SearchResult objects with scores and explanations
        """
        # Validate query
        self.validation.validate_search_query(query)
        
        # Normalize query
        query = self._normalize_query(query)
        
        # Use defaults if not specified
        limit = min(limit or self.max_results, self.max_results)
        min_score = min_score or self.min_score
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate(query)
            
            # Search in Qdrant
            vector_results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit * 2,  # Fetch extra for re-ranking
                memory_type=memory_type.value if memory_type else None,
                min_score=min_score,
            )
            
            if not vector_results:
                logger.debug(f"No results found for query: {query[:50]}...")
                return []
            
            # Fetch full memory objects from SQLite
            results = []
            for vr in vector_results:
                memory = self.db.get_memory(UUID(vr["memory_id"]))
                # v2: Skip archived memories and optionally stale memories
                if memory and memory.confirmed and not memory.is_archived:
                    if exclude_stale and memory.is_stale:
                        continue
                    results.append({
                        "memory": memory,
                        "score": vr["score"],
                        "created_at": vr.get("created_at"),
                    })
            
            # Re-rank with recency boost
            results = self._rerank_results(results)
            
            # Limit final results
            results = results[:limit]
            
            # v2: Update last_accessed for retrieved memories (staleness tracking)
            for r in results:
                try:
                    self.db.update_last_accessed(r["memory"].id)
                except Exception as e:
                    logger.debug(f"Failed to update last_accessed: {e}")
            
            # Build search results with explanations
            search_results = []
            for r in results:
                explanation = self._generate_explanation(
                    query=query,
                    memory=r["memory"],
                    score=r["score"],
                )
                search_results.append(SearchResult(
                    memory=r["memory"],
                    score=r["score"],
                    explanation=explanation,
                ))
            
            logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Fallback to keyword search in SQLite
            return self._fallback_keyword_search(query, memory_type, limit)

    def get_timeline(self, limit: int = 20) -> list[Memory]:
        """
        Get memories in chronological order (most recent first).
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of memories sorted by creation date (desc)
        """
        return self.db.get_recent_memories(self.project_id, limit)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize a search query."""
        # Strip whitespace
        query = query.strip()
        
        # Remove excessive whitespace
        import re
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def _rerank_results(
        self,
        results: list[dict],
        recency_weight: float = 0.1,
        confidence_weight: float = 0.1,  # v3: confidence scoring
    ) -> list[dict]:
        """
        Re-rank results with recency and confidence boosts.
        
        More recent and higher-confidence memories get boosted scores.
        This provides a deterministic tie-breaker.
        """
        now = datetime.utcnow()
        
        for result in results:
            memory = result["memory"]
            base_score = result["score"]
            
            # Calculate recency boost (0-0.1)
            # Memories from the last 7 days get full boost
            age_days = (now - memory.created_at).days
            recency_boost = max(0, recency_weight * (1 - age_days / 30))
            
            # Apply type priority boost
            type_boost = self._get_type_priority(memory.type) * 0.05
            
            # v3: Apply confidence boost
            # High confidence memories get boosted, low confidence get penalized
            confidence_boost = (memory.confidence_score - 0.5) * confidence_weight
            
            # Combined score
            result["score"] = min(1.0, base_score + recency_boost + type_boost + confidence_boost)
        
        # Sort by adjusted score (descending), then by created_at (descending) as tie-breaker
        results.sort(
            key=lambda x: (x["score"], x["memory"].created_at),
            reverse=True,
        )
        
        return results
    
    def _get_type_priority(self, memory_type: MemoryType) -> float:
        """
        Get priority weight for a memory type.
        
        Stack and decision memories are weighted slightly higher
        as they're typically more important for context.
        """
        priorities = {
            MemoryType.STACK: 1.0,
            MemoryType.DECISION: 0.9,
            MemoryType.CONSTRAINT: 0.8,
            MemoryType.CONVENTION: 0.7,
            MemoryType.NOTE: 0.5,
        }
        return priorities.get(memory_type, 0.5)
    
    def _generate_explanation(
        self,
        query: str,
        memory: Memory,
        score: float,
    ) -> str:
        """
        Generate an explanation for why this result was selected.
        
        This is critical for building user trust.
        """
        type_label = memory.type.value.replace("_", " ").title()
        date_str = memory.created_at.strftime("%b %d, %Y")
        
        # Score interpretation
        if score >= 0.85:
            relevance = "highly relevant"
        elif score >= 0.7:
            relevance = "relevant"
        else:
            relevance = "partially relevant"
        
        return f"[{type_label}] {relevance} (score: {score:.2f}, stored {date_str})"
    
    def _fallback_keyword_search(
        self,
        query: str,
        memory_type: Optional[MemoryType],
        limit: int,
    ) -> list[SearchResult]:
        """
        Fallback keyword search when Qdrant is unavailable.
        
        Uses SQLite LIKE queries for basic text matching.
        """
        logger.warning("Using fallback keyword search")
        
        memories = self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=True,
            memory_type=memory_type,
            limit=100,
        )
        
        # Simple keyword matching
        query_lower = query.lower()
        keywords = query_lower.split()
        
        scored_memories = []
        for memory in memories:
            content_lower = memory.content.lower()
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                score = min(1.0, matches / len(keywords) * 0.7)  # Cap at 0.7 for keyword search
                scored_memories.append((memory, score))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for memory, score in scored_memories[:limit]:
            results.append(SearchResult(
                memory=memory,
                score=score,
                explanation=f"[Keyword match] score: {score:.2f}",
            ))
        
        return results
    
    def get_recent_memories(
        self,
        limit: int = 5,
        memory_type: Optional[MemoryType] = None,
    ) -> list[Memory]:
        """Get the most recent memories."""
        return self.db.list_memories(
            project_id=self.project_id,
            confirmed_only=True,
            memory_type=memory_type,
            limit=limit,
            offset=0,
        )
