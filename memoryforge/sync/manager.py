"""
Sync Manager for MemoryForge.

Orchestrates the synchronization process:
1. Encrypts/Decrypts memories
2. Pushes local changes to backend
3. Pulls remote changes to local DB
"""

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


class SyncMetadata(BaseModel):
    """Metadata wrapper for synced memory files."""
    id: UUID
    project_id: UUID
    updated_at: datetime
    is_archived: bool
    is_stale: bool
    # Content and other fields are inside encrypted_payload


class SyncManager:
    """
    Manages synchronization of memories.
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
    
    def export_memories(self, force: bool = False) -> int:
        """
        Export local memories to sync backend.
        
        Args:
            force: If True, overwrite all remote files
            
        Returns:
            Number of memories exported
        """
        self.adapter.initialize()
        
        # Get all memories for this project
        memories = self.db.list_memories(
            project_id=self.project_id,
            include_archived=True,
            limit=10000,
        )
        
        count = 0
        for memory in memories:
            filename = f"{memory.id}.json"
            
            # Check if remote exists and is newer
            if not force:
                remote_mod = self.adapter.get_last_modified(filename)
                # Simple logic: if remote exists, don't overwrite unless force
                # Ideally check timestamps, but we lack detailed memory.updated_at
                if remote_mod:
                    # Logic: if memory is archived locally but not remote -> Push
                    # If we don't track update time, this is hard.
                    # For now: Push if missing or force.
                    continue
            
            # Serialize and encrypt
            payload = self._create_payload(memory)
            self.adapter.write_file(filename, payload)
            count += 1
            
        return count
    
    def import_memories(self) -> int:
        """
        Import remote memories to local DB.
        
        Returns:
            Number of memories imported/updated
        """
        remote_files = self.adapter.list_files()
        count = 0
        
        for filename in remote_files:
            try:
                content = self.adapter.read_file(filename)
                if not content:
                    continue
                
                memory = self._parse_payload(content)
                
                # Check if it belongs to current project
                if memory.project_id != self.project_id:
                    continue
                
                # Check if exists locally
                existing = self.db.get_memory(memory.id)
                if existing:
                    # Update if remote has changed status (e.g. archived)
                    if memory.is_archived and not existing.is_archived:
                        self.db.archive_memory(memory.id, memory.consolidated_into)
                    # Add more merge logic here
                else:
                    # New memory
                    self.db.save_memory(memory)
                    count += 1
                    
            except Exception as e:
                logger.error(f"Failed to import {filename}: {e}")
                
        return count
    
    def _create_payload(self, memory: Memory) -> str:
        """Create encrypted JSON payload."""
        # Dump full memory to JSON
        json_data = memory.model_dump_json()
        
        # Encrypt
        encrypted_data = self.encryption.encrypt(json_data)
        
        # Wrap with metadata
        wrapper = {
            "id": str(memory.id),
            "project_id": str(memory.project_id),
            "updated_at": datetime.utcnow().isoformat(),
            "encrypted_data": encrypted_data,
        }
        return json.dumps(wrapper, indent=2)
    
    def _parse_payload(self, content: str) -> Memory:
        """Parse and decrypt payload."""
        wrapper = json.loads(content)
        encrypted_data = wrapper["encrypted_data"]
        
        # Decrypt
        decrypted_json = self.encryption.decrypt(encrypted_data)
        
        # Parse Memory
        return Memory.model_validate_json(decrypted_json)
