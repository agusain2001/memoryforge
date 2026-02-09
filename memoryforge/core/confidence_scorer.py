"""
Confidence Scorer for MemoryForge v3.

Calculates and manages memory confidence scores based on:
- Human confirmation count
- Recency of creation/access
- Usage frequency
- Conflict history
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from memoryforge.models import Memory
from memoryforge.storage.sqlite_db import SQLiteDatabase


class ConfidenceScorer:
    """Calculates and manages memory confidence scores."""
    
    # Scoring weights
    WEIGHT_CONFIRMATION = 0.25
    WEIGHT_RECENCY = 0.25
    WEIGHT_USAGE = 0.25
    WEIGHT_CONFLICTS = 0.25
    
    # Decay constants
    RECENCY_HALF_LIFE_DAYS = 30  # Score halves every 30 days of inactivity
    
    def __init__(self, db: SQLiteDatabase):
        """Initialize the confidence scorer."""
        self.db = db
    
    def calculate_score(self, memory: Memory) -> float:
        """Calculate the confidence score for a memory.
        
        Score is based on:
        - Confirmation status (confirmed = higher)
        - Recency (recent access = higher)
        - Usage frequency (more accesses = higher, but diminishing returns)
        - Conflict history (more conflicts = lower)
        
        Returns a score between 0.0 and 1.0.
        """
        confirmation_score = self._confirmation_score(memory)
        recency_score = self._recency_score(memory)
        usage_score = self._usage_score(memory)
        conflict_score = self._conflict_score(memory.id)
        
        # Weighted average
        total = (
            confirmation_score * self.WEIGHT_CONFIRMATION +
            recency_score * self.WEIGHT_RECENCY +
            usage_score * self.WEIGHT_USAGE +
            conflict_score * self.WEIGHT_CONFLICTS
        )
        
        return min(1.0, max(0.0, total))
    
    def _confirmation_score(self, memory: Memory) -> float:
        """Score based on confirmation status."""
        return 1.0 if memory.confirmed else 0.3
    
    def _recency_score(self, memory: Memory) -> float:
        """Score based on how recently the memory was accessed/created."""
        now = datetime.utcnow()
        
        # Use last_accessed if available, otherwise created_at
        reference_time = memory.last_accessed or memory.created_at
        
        days_since = (now - reference_time).days
        
        # Exponential decay
        half_life = self.RECENCY_HALF_LIFE_DAYS
        score = 0.5 ** (days_since / half_life)
        
        return score
    
    def _usage_score(self, memory: Memory) -> float:
        """Score based on usage frequency.
        
        Uses diminishing returns formula: 1 - (1 / (1 + log(accesses + 1)))
        """
        import math
        
        # If never accessed, return base score
        if not memory.last_accessed:
            return 0.5
        
        # Count accesses by looking at memory versions and links
        # For now, use a simple heuristic based on whether it was accessed
        return 0.8 if memory.last_accessed else 0.5
    
    def _conflict_score(self, memory_id: UUID) -> float:
        """Score based on conflict history (more conflicts = lower score)."""
        conflicts = self.db.get_conflict_history(memory_id)
        conflict_count = len(conflicts)
        
        if conflict_count == 0:
            return 1.0
        elif conflict_count == 1:
            return 0.7
        elif conflict_count <= 3:
            return 0.5
        else:
            return 0.3
    
    def update_score(self, memory_id: UUID) -> float:
        """Recalculate and update the confidence score for a memory."""
        memory = self.db.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        
        new_score = self.calculate_score(memory)
        self.db.update_confidence_score(memory_id, new_score)
        
        return new_score
    
    def batch_update_scores(self, project_id: UUID) -> dict[UUID, float]:
        """Recalculate scores for all memories in a project.
        
        Returns a dict mapping memory_id -> new_score.
        """
        memories = self.db.list_memories(project_id, confirmed_only=False, limit=10000)
        results = {}
        
        for memory in memories:
            new_score = self.calculate_score(memory)
            self.db.update_confidence_score(memory.id, new_score)
            results[memory.id] = new_score
        
        return results
    
    def get_low_confidence(
        self,
        project_id: UUID,
        threshold: float = 0.5,
    ) -> list[Memory]:
        """Get memories with low confidence scores."""
        return self.db.get_low_confidence_memories(project_id, threshold)
    
    def get_confidence_details(self, memory_id: UUID) -> dict:
        """Get detailed confidence breakdown for a memory."""
        memory = self.db.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        
        conflicts = self.db.get_conflict_history(memory_id)
        
        return {
            "memory_id": str(memory_id),
            "current_score": memory.confidence_score,
            "breakdown": {
                "confirmation": {
                    "score": self._confirmation_score(memory),
                    "confirmed": memory.confirmed,
                },
                "recency": {
                    "score": self._recency_score(memory),
                    "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                    "created_at": memory.created_at.isoformat(),
                },
                "usage": {
                    "score": self._usage_score(memory),
                },
                "conflicts": {
                    "score": self._conflict_score(memory_id),
                    "conflict_count": len(conflicts),
                },
            },
            "calculated_score": self.calculate_score(memory),
        }
