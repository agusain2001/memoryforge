"""
Integration tests for sync roundtrip functionality.

Tests the full sync workflow: encrypt → export → import → decrypt
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import pytest

# Skip entire module if cryptography not installed
pytest.importorskip(
    "cryptography",
    reason="sync integration tests require: pip install 'memoryforge[sync]'"
)

from memoryforge.models import Memory, MemoryType, MemorySource
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.sync.encryption import EncryptionLayer
from memoryforge.sync.local_file_adapter import LocalFileAdapter
from memoryforge.sync.manager import SyncManager


def test_encrypt_decrypt_roundtrip(encryption_key):
    """Test that encryption and decryption work correctly."""
    encryption = EncryptionLayer(encryption_key)
    original = "hello world from memoryforge"
    
    # Encrypt
    encrypted = encryption.encrypt(original)
    
    # Verify it's actually encrypted (not same as original)
    assert encrypted != original
    
    # Decrypt
    decrypted = encryption.decrypt(encrypted)
    
    # Verify roundtrip
    assert decrypted == original


def test_export_creates_file_on_disk(sync_manager, sample_memory, tmp_path):
    """Test that export creates a file on disk."""
    sync_dir = tmp_path / "sync"
    
    # Export
    result = sync_manager.export_memories()
    
    # Verify file created
    files = list(sync_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].name == f"{sample_memory.id}.json"
    
    # Verify file not empty
    assert files[0].stat().st_size > 0
    
    # Verify exported count
    assert result.exported == 1


def test_export_then_import_roundtrip(temp_db, temp_project, tmp_path, encryption_key, sample_memory):
    """Test full export → import roundtrip between two databases."""
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    
    # Setup first sync manager (machine A)
    encryption = EncryptionLayer(encryption_key)
    adapter_a = LocalFileAdapter(sync_dir)
    manager_a = SyncManager(temp_db, adapter_a, encryption, temp_project.id)
    
    # Export from machine A
    result = manager_a.export_memories()
    assert result.exported == 1
    
    # Setup second database (machine B)
    db_b_path = tmp_path / "db_b" / "memoryforge.db"
    db_b = SQLiteDatabase(db_b_path)
    
    # Create same project in db_b
    project_b = temp_db.get_project(temp_project.id)
    db_b.create_project(project_b)
    
    # Setup second sync manager (machine B, same sync dir)
    adapter_b = LocalFileAdapter(sync_dir)
    manager_b = SyncManager(db_b, adapter_b, encryption, temp_project.id)
    
    # Import to machine B
    result = manager_b.import_memories()
    assert result.imported == 1
    
    # Verify memory in db_b
    memories = db_b.list_memories(project_id=temp_project.id, confirmed_only=False)
    assert len(memories) == 1
    
    imported_memory = memories[0]
    assert imported_memory.content == sample_memory.content
    assert imported_memory.type == sample_memory.type
    assert imported_memory.project_id == sample_memory.project_id
    assert imported_memory.confirmed == True


def test_roundtrip_preserves_all_fields(temp_db, temp_project, tmp_path, encryption_key):
    """Test that roundtrip preserves non-default fields."""
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    
    # Create memory with non-default fields
    memory = Memory(
        content="Complex memory with metadata",
        type=MemoryType.DECISION,
        source=MemorySource.MANUAL,
        project_id=temp_project.id,
        confirmed=True,
        is_stale=True,
        confidence_score=0.65,
        metadata={"framework": "fastapi", "version": "0.100"},
    )
    temp_db.create_memory(memory)
    
    # Setup sync manager A
    encryption = EncryptionLayer(encryption_key)
    adapter_a = LocalFileAdapter(sync_dir)
    manager_a = SyncManager(temp_db, adapter_a, encryption, temp_project.id)
    
    # Export
    result = manager_a.export_memories()
    assert result.exported == 1
    
    # Setup second database
    db_b_path = tmp_path / "db_b" / "memoryforge.db"
    db_b = SQLiteDatabase(db_b_path)
    project_b = temp_db.get_project(temp_project.id)
    db_b.create_project(project_b)
    
    # Import to db_b
    adapter_b = LocalFileAdapter(sync_dir)
    manager_b = SyncManager(db_b, adapter_b, encryption, temp_project.id)
    result = manager_b.import_memories()
    assert result.imported == 1
    
    # Verify all fields preserved
    imported = db_b.list_memories(project_id=temp_project.id, confirmed_only=False)[0]
    assert imported.is_stale == True
    assert imported.confidence_score == 0.65
    assert imported.metadata == {"framework": "fastapi", "version": "0.100"}


def test_conflict_detected_without_force(sync_manager, sample_memory, tmp_path):
    """Test that conflict is detected when remote file is newer."""
    sync_dir = tmp_path / "sync"
    
    # Export once
    result = sync_manager.export_memories()
    assert result.exported == 1
    
    # Tamper with the exported file - set updated_at to 1 hour in future
    sync_file = next(sync_dir.glob("*.json"))
    payload = json.loads(sync_file.read_text())
    payload["updated_at"] = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    sync_file.write_text(json.dumps(payload))
    
    # Try to export again without force
    result = sync_manager.export_memories(force=False)
    
    # Should have conflict, no export
    assert len(result.conflicts) > 0
    assert result.exported == 0


def test_force_push_overwrites_conflict(sync_manager, sample_memory, tmp_path):
    """Test that force mode overwrites conflicts."""
    sync_dir = tmp_path / "sync"
    
    # Export once
    result = sync_manager.export_memories()
    assert result.exported == 1
    
    # Tamper with the file
    sync_file = next(sync_dir.glob("*.json"))
    payload = json.loads(sync_file.read_text())
    payload["updated_at"] = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    sync_file.write_text(json.dumps(payload))
    
    # Export with force=True
    result = sync_manager.export_memories(force=True)
    
    # Should have no conflicts, export succeeds
    assert len(result.conflicts) == 0
    assert result.exported == 1
    
    # Verify file reflects local content
    new_payload = json.loads(sync_file.read_text())
    # The updated_at should be reset to local time (not future)
    new_updated = datetime.fromisoformat(new_payload["updated_at"])
    assert new_updated < datetime.utcnow() + timedelta(seconds=10)


def test_import_skips_different_project(temp_db, tmp_path, encryption_key):
    """Test that import skips memories from different projects."""
    from memoryforge.models import Project
    
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    
    # Create project A
    project_a = Project(name="project-a", root_path="/tmp/a")
    temp_db.create_project(project_a)
    
    # Create memory in project A
    memory_a = Memory(
        content="Memory for project A",
        type=MemoryType.STACK,
        source=MemorySource.MANUAL,
        project_id=project_a.id,
        confirmed=True,
    )
    temp_db.create_memory(memory_a)
    
    # Export from project A
    encryption = EncryptionLayer(encryption_key)
    adapter = LocalFileAdapter(sync_dir)
    manager_a = SyncManager(temp_db, adapter, encryption, project_a.id)
    result = manager_a.export_memories()
    assert result.exported == 1
    
    # Create project B
    project_b = Project(name="project-b", root_path="/tmp/b")
    temp_db.create_project(project_b)
    
    # Try to import to project B
    manager_b = SyncManager(temp_db, adapter, encryption, project_b.id)
    result = manager_b.import_memories()
    
    # Should not import (different project)
    assert result.imported == 0
    
    # Verify project B has no memories
    memories_b = temp_db.list_memories(project_id=project_b.id, confirmed_only=False)
    assert len(memories_b) == 0


def test_import_is_idempotent(temp_db, temp_project, tmp_path, encryption_key, sample_memory):
    """Test that importing twice doesn't create duplicates."""
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    
    # Export
    encryption = EncryptionLayer(encryption_key)
    adapter = LocalFileAdapter(sync_dir)
    manager = SyncManager(temp_db, adapter, encryption, temp_project.id)
    result = manager.export_memories()
    assert result.exported == 1
    
    # Import first time
    result = manager.import_memories()
    # Should be 0 because memory already exists in same db
    assert result.imported == 0
    
    # Import second time
    result = manager.import_memories()
    assert result.imported == 0
    
    # Verify only one memory exists
    memories = temp_db.list_memories(project_id=temp_project.id, confirmed_only=False)
    assert len(memories) == 1
