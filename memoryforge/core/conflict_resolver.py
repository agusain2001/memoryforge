"""
Conflict Resolver for MemoryForge v3.

Handles sync conflicts with:
- Last-write-wins by default
- Manual merge option
- Full conflict history logging
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from memoryforge.models import Memory, ConflictLog, ConflictResolution
from memoryforge.storage.sqlite_db import SQLiteDatabase


class SyncConflict:
    """Represents a sync conflict between local and remote memory versions."""
    
    def __init__(
        self,
        memory_id: UUID,
        local_memory: Optional[Memory],
        remote_content: str,
        remote_updated_at: datetime,
    ):
        self.memory_id = memory_id
        self.local_memory = local_memory
        self.remote_content = remote_content
        self.remote_updated_at = remote_updated_at


class ConflictResolver:
    """Resolves sync conflicts between local and remote memories."""
    
    def __init__(self, db: SQLiteDatabase):
        """Initialize the conflict resolver."""
        self.db = db
    
    def detect_conflict(
        self,
        local_memory: Optional[Memory],
        remote_content: str,
        remote_updated_at: datetime,
    ) -> bool:
        """Check if there's a conflict between local and remote versions.
        
        A conflict exists if:
        - Both versions exist
        - Content differs
        - Remote was updated after local was last synced
        """
        if not local_memory:
            return False  # No conflict if local doesn't exist
        
        if local_memory.content == remote_content:
            return False  # No conflict if content is same
        
        # If both have been modified, it's a conflict
        local_updated = local_memory.updated_at or local_memory.created_at
        return remote_updated_at != local_updated
    
    def resolve_last_write_wins(
        self,
        conflict: SyncConflict,
    ) -> ConflictLog:
        """Resolve conflict using last-write-wins strategy.
        
        The version with the most recent update timestamp wins.
        """
        local = conflict.local_memory
        
        if not local:
            # No local version, remote wins by default
            return self._apply_remote(conflict, ConflictResolution.REMOTE_WINS)
        
        local_time = local.updated_at or local.created_at
        
        if conflict.remote_updated_at > local_time:
            return self._apply_remote(conflict, ConflictResolution.REMOTE_WINS)
        else:
            return self._apply_local(conflict, ConflictResolution.LOCAL_WINS)
    
    def resolve_manual(
        self,
        conflict: SyncConflict,
        merged_content: str,
        resolved_by: str = "user",
    ) -> ConflictLog:
        """Manually resolve a conflict with custom merged content."""
        # Update the memory with merged content
        self.db.update_memory(conflict.memory_id, merged_content)
        
        # Log the conflict
        return self.db.log_conflict(
            memory_id=conflict.memory_id,
            local_content=conflict.local_memory.content if conflict.local_memory else None,
            remote_content=conflict.remote_content,
            resolution=ConflictResolution.MANUAL,
            resolved_by=resolved_by,
        )
    
    def resolve_keep_local(self, conflict: SyncConflict) -> ConflictLog:
        """Keep the local version, discard remote."""
        return self._apply_local(conflict, ConflictResolution.LOCAL_WINS)
    
    def resolve_keep_remote(self, conflict: SyncConflict) -> ConflictLog:
        """Keep the remote version, discard local."""
        return self._apply_remote(conflict, ConflictResolution.REMOTE_WINS)
    
    def _apply_local(
        self,
        conflict: SyncConflict,
        resolution: ConflictResolution,
    ) -> ConflictLog:
        """Apply local version and log the conflict."""
        # Local is already the current state, just log the conflict
        return self.db.log_conflict(
            memory_id=conflict.memory_id,
            local_content=conflict.local_memory.content if conflict.local_memory else None,
            remote_content=conflict.remote_content,
            resolution=resolution,
            resolved_by="system",
        )
    
    def _apply_remote(
        self,
        conflict: SyncConflict,
        resolution: ConflictResolution,
    ) -> ConflictLog:
        """Apply remote version and log the conflict."""
        # Update to remote content
        self.db.update_memory(conflict.memory_id, conflict.remote_content)
        
        return self.db.log_conflict(
            memory_id=conflict.memory_id,
            local_content=conflict.local_memory.content if conflict.local_memory else None,
            remote_content=conflict.remote_content,
            resolution=resolution,
            resolved_by="system",
        )
    
    def list_conflicts(
        self,
        memory_id: Optional[UUID] = None,
    ) -> list[ConflictLog]:
        """List conflict history, optionally filtered by memory."""
        return self.db.get_conflict_history(memory_id)
    
    def get_conflict_count(self, memory_id: UUID) -> int:
        """Get the number of conflicts for a memory."""
        return len(self.db.get_conflict_history(memory_id))
