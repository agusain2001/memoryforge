"""
Sync Manager for MemoryForge.

Orchestrates the synchronization process:
1. Encrypts/Decrypts memories
2. Pushes local changes to backend
3. Pulls remote changes to local DB
4. Detects and reports conflicts

v2.1: Added conflict detection and integrity verification.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel

from memoryforge.config import Config
from memoryforge.models import Memory
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.sync.adapter import SyncAdapterProtocol
from memoryforge.sync.encryption import EncryptionLayer

logger = logging.getLogger(__name__)


class SyncConflictError(Exception):
    """Raised when a sync conflict is detected."""
    
    def __init__(self, memory_id: UUID, local_updated: datetime, remote_updated: datetime):
        self.memory_id = memory_id
        self.local_updated = local_updated
        self.remote_updated = remote_updated
        super().__init__(
            f"Conflict detected for memory {memory_id}: "
            f"local={local_updated.isoformat()}, remote={remote_updated.isoformat()}"
        )


class SyncIntegrityError(Exception):
    """Raised when data integrity check fails."""
    
    def __init__(self, memory_id: UUID, message: str = "Data integrity check failed"):
        self.memory_id = memory_id
        super().__init__(f"{message} for memory {memory_id}")


class SyncMetadata(BaseModel):
    """Metadata wrapper for synced memory files."""
    id: UUID
    project_id: UUID
    updated_at: datetime
    is_archived: bool
    is_stale: bool
    checksum: Optional[str] = None  # SHA256 of content for integrity
    # Content and other fields are inside encrypted_payload


class SyncResult(BaseModel):
    """Result of a sync operation."""
    exported: int = 0
    imported: int = 0
    conflicts: List[str] = []
    errors: List[str] = []
    
    @property
    def success(self) -> bool:
        return len(self.conflicts) == 0 and len(self.errors) == 0


class SyncManager:
    """
    Manages synchronization of memories with conflict detection.
    
    Features:
    - Timestamp-based conflict detection
    - Data integrity verification via checksums
    - Comprehensive merge logic
    """
    
    def __init__(
        self,
        db: SQLiteDatabase,
        adapter: SyncAdapterProtocol,
        encryption: EncryptionLayer,
        project_id: UUID,
    ):
        self.db = db
        self.adapter = adapter
        self.encryption = encryption
        self.project_id = project_id
    
    def export_memories(self, force: bool = False) -> SyncResult:
        """
        Export local memories to sync backend with conflict detection.
        
        Args:
            force: If True, overwrite all remote files (skip conflict check)
            
        Returns:
            SyncResult with counts and any conflicts/errors
        """
        result = SyncResult()
        self.adapter.initialize()
        
        # Get all memories for this project
        memories = self.db.list_memories(
            project_id=self.project_id,
            include_archived=True,
            limit=10000,
        )
        
        for memory in memories:
            filename = f"{memory.id}.json"
            
            try:
                # Check for conflicts unless force mode
                if not force:
                    conflict = self._check_conflict(memory, filename)
                    if conflict:
                        result.conflicts.append(str(conflict))
                        continue
                
                # Serialize and encrypt with integrity checksum
                payload = self._create_payload(memory)
                self.adapter.write_file(filename, payload)
                result.exported += 1
                
            except Exception as e:
                logger.error(f"Failed to export memory {memory.id}: {e}")
                result.errors.append(f"Export failed for {memory.id}: {e}")
                
        return result
    
    def import_memories(self, force: bool = False) -> SyncResult:
        """
        Import remote memories to local DB with conflict detection.
        
        Args:
            force: If True, overwrite local memories (skip conflict check)
            
        Returns:
            SyncResult with counts and any conflicts/errors
        """
        result = SyncResult()
        remote_files = self.adapter.list_files()
        
        for filename in remote_files:
            try:
                content = self.adapter.read_file(filename)
                if not content:
                    continue
                
                # Parse and verify integrity
                memory, remote_updated = self._parse_payload(content)
                
                # Check if it belongs to current project
                if memory.project_id != self.project_id:
                    continue
                
                # Check if exists locally
                existing = self.db.get_memory(memory.id)
                if existing:
                    # Check for conflicts unless force mode
                    if not force:
                        local_updated = existing.updated_at or existing.created_at
                        
                        # Conflict if both have been modified
                        if local_updated and remote_updated:
                            if local_updated > remote_updated:
                                # Local is newer - potential conflict
                                result.conflicts.append(
                                    f"Local memory {memory.id} is newer than remote"
                                )
                                continue
                    
                    # Apply merge logic
                    self._merge_memory(existing, memory, remote_updated)
                else:
                    # New memory from remote
                    self.db.save_memory(memory)
                    result.imported += 1
                    
            except SyncIntegrityError as e:
                result.errors.append(str(e))
            except Exception as e:
                logger.error(f"Failed to import {filename}: {e}")
                result.errors.append(f"Import failed for {filename}: {e}")
                
        return result
    
    def _check_conflict(self, memory: Memory, filename: str) -> Optional[SyncConflictError]:
        """Check if there's a conflict with remote version."""
        remote_content = self.adapter.read_file(filename)
        if not remote_content:
            return None  # No remote version, no conflict
        
        try:
            wrapper = json.loads(remote_content)
            remote_updated = datetime.fromisoformat(wrapper["updated_at"])
            
            local_updated = memory.updated_at or memory.created_at
            
            # If remote is newer than our last sync, there's a potential conflict
            # We consider it a conflict if both have been modified
            if remote_updated and local_updated:
                # Allow 1 second tolerance for near-simultaneous updates
                time_diff = abs((remote_updated - local_updated).total_seconds())
                if time_diff > 1 and remote_updated > local_updated:
                    return SyncConflictError(memory.id, local_updated, remote_updated)
            
        except Exception as e:
            logger.warning(f"Could not check conflict for {filename}: {e}")
        
        return None
    
    def _merge_memory(
        self,
        local: Memory,
        remote: Memory,
        remote_updated: datetime,
    ) -> None:
        """
        Merge remote memory changes into local.
        
        Merge strategy:
        - Archive status: If either is archived, archive local
        - Stale status: Union of stale flags
        - Content: Remote wins if remote is newer
        """
        changed = False
        
        # Archive status - if remote is archived, archive local
        if remote.is_archived and not local.is_archived:
            self.db.archive_memory(local.id, remote.consolidated_into)
            changed = True
        
        # Stale status
        if remote.is_stale and not local.is_stale:
            self.db.mark_stale(local.id, remote.stale_reason or "Synced from remote")
            changed = True
        
        # Content update - only if remote is definitively newer
        local_updated = local.updated_at or local.created_at
        if remote_updated > local_updated:
            if remote.content != local.content:
                self.db.update_memory(local.id, remote.content)
                changed = True
        
        if changed:
            logger.info(f"Merged changes for memory {local.id}")
    
    def _create_payload(self, memory: Memory) -> str:
        """Create encrypted JSON payload with integrity checksum."""
        # Dump full memory to JSON
        json_data = memory.model_dump_json()
        
        # Calculate checksum for integrity verification
        checksum = hashlib.sha256(json_data.encode()).hexdigest()[:32]
        
        # Encrypt
        encrypted_data = self.encryption.encrypt(json_data)
        
        # Wrap with metadata
        wrapper = {
            "id": str(memory.id),
            "project_id": str(memory.project_id),
            "updated_at": (memory.updated_at or datetime.utcnow()).isoformat(),
            "checksum": checksum,
            "encrypted_data": encrypted_data,
        }
        return json.dumps(wrapper, indent=2)
    
    def _parse_payload(self, content: str) -> Tuple[Memory, datetime]:
        """
        Parse and decrypt payload with integrity verification.
        
        Returns:
            Tuple of (Memory, updated_at timestamp)
            
        Raises:
            SyncIntegrityError: If checksum doesn't match
        """
        wrapper = json.loads(content)
        encrypted_data = wrapper["encrypted_data"]
        expected_checksum = wrapper.get("checksum")
        updated_at = datetime.fromisoformat(wrapper["updated_at"])
        
        # Decrypt
        decrypted_json = self.encryption.decrypt(encrypted_data)
        
        # Verify integrity if checksum present
        if expected_checksum:
            actual_checksum = hashlib.sha256(decrypted_json.encode()).hexdigest()[:32]
            if actual_checksum != expected_checksum:
                memory_id = UUID(wrapper["id"])
                raise SyncIntegrityError(memory_id, "Checksum mismatch - data may be corrupted")
        
        # Parse Memory
        memory = Memory.model_validate_json(decrypted_json)
        return memory, updated_at
