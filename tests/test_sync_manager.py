"""
Tests for sync manager - conflict detection and data integrity.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase

# Check if sync dependencies are available
try:
    from memoryforge.sync.manager import (
        SyncManager,
        SyncResult, 
        SyncConflictError,
        SyncIntegrityError,
    )
    from memoryforge.sync.encryption import EncryptionLayer
    from memoryforge.sync.local_file_adapter import LocalFileAdapter
    HAS_SYNC = True
except ImportError:
    HAS_SYNC = False


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = SQLiteDatabase(db_path)
        yield db


@pytest.fixture
def project(temp_db):
    """Create a test project."""
    project = Project(
        name="test-project",
        root_path="/test/path",
    )
    return temp_db.create_project(project)


@pytest.fixture
def mock_adapter():
    """Create a mock sync adapter."""
    adapter = Mock()
    adapter.initialize.return_value = None
    adapter.list_files.return_value = []
    adapter.read_file.return_value = None
    adapter.write_file.return_value = None
    adapter.get_last_modified.return_value = None
    return adapter


@pytest.fixture
def mock_encryption():
    """Create a mock encryption layer."""
    encryption = Mock()
    encryption.encrypt.side_effect = lambda x: f"ENCRYPTED:{x}"
    encryption.decrypt.side_effect = lambda x: x.replace("ENCRYPTED:", "")
    return encryption


@pytest.mark.skipif(not HAS_SYNC, reason="sync dependencies not installed")
class TestSyncResult:
    """Tests for SyncResult model."""
    
    def test_empty_result_is_success(self):
        """Test that empty result is considered success."""
        result = SyncResult()
        assert result.success is True
    
    def test_result_with_conflicts_is_failure(self):
        """Test that result with conflicts is failure."""
        result = SyncResult(conflicts=["conflict1"])
        assert result.success is False
    
    def test_result_with_errors_is_failure(self):
        """Test that result with errors is failure."""
        result = SyncResult(errors=["error1"])
        assert result.success is False


@pytest.mark.skipif(not HAS_SYNC, reason="sync dependencies not installed")
class TestSyncExport:
    """Tests for memory export."""
    
    def test_export_creates_files(self, temp_db, project, mock_adapter, mock_encryption):
        """Test that export creates files for each memory."""
        # Create some memories
        for i in range(3):
            memory = Memory(
                content=f"Memory {i}",
                type=MemoryType.NOTE,
                source=MemorySource.MANUAL,
                project_id=project.id,
                confirmed=True,
            )
            temp_db.create_memory(memory)
        
        manager = SyncManager(
            db=temp_db,
            adapter=mock_adapter,
            encryption=mock_encryption,
            project_id=project.id,
        )
        
        result = manager.export_memories(force=True)
        
        assert result.exported == 3
        assert mock_adapter.write_file.call_count == 3
    
    def test_export_force_overwrites(self, temp_db, project, mock_adapter, mock_encryption):
        """Test that force mode overwrites existing files."""
        memory = Memory(
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Simulate existing remote file
        mock_adapter.read_file.return_value = '{"id": "test", "updated_at": "2020-01-01T00:00:00"}'
        
        manager = SyncManager(
            db=temp_db,
            adapter=mock_adapter,
            encryption=mock_encryption,
            project_id=project.id,
        )
        
        result = manager.export_memories(force=True)
        
        assert result.exported == 1
        assert len(result.conflicts) == 0


@pytest.mark.skipif(not HAS_SYNC, reason="sync dependencies not installed")
class TestSyncConflictDetection:
    """Tests for conflict detection."""
    
    def test_conflict_detected_when_remote_newer(self, temp_db, project, mock_adapter, mock_encryption):
        """Test that conflict is detected when remote is newer."""
        memory = Memory(
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            created_at=datetime.utcnow() - timedelta(days=1),
        )
        temp_db.create_memory(memory)
        
        # Simulate newer remote file
        remote_time = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        mock_adapter.read_file.return_value = json.dumps({
            "id": str(memory.id),
            "updated_at": remote_time,
            "encrypted_data": "ENCRYPTED:test",
        })
        
        manager = SyncManager(
            db=temp_db,
            adapter=mock_adapter,
            encryption=mock_encryption,
            project_id=project.id,
        )
        
        result = manager.export_memories(force=False)
        
        # Should detect conflict
        assert len(result.conflicts) == 1
        assert result.exported == 0


@pytest.mark.skipif(not HAS_SYNC, reason="sync dependencies not installed")
class TestSyncImport:
    """Tests for memory import."""
    
    def test_import_new_memory(self, temp_db, project, mock_adapter, mock_encryption):
        """Test importing a new memory."""
        # Create a mock remote memory
        new_memory = Memory(
            id=uuid4(),
            content="Remote memory",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        
        remote_data = json.dumps({
            "id": str(new_memory.id),
            "project_id": str(new_memory.project_id),
            "updated_at": datetime.utcnow().isoformat(),
            "checksum": None,
            "encrypted_data": new_memory.model_dump_json(),
        })
        
        mock_adapter.list_files.return_value = [f"{new_memory.id}.json"]
        mock_adapter.read_file.return_value = remote_data
        mock_encryption.decrypt.return_value = new_memory.model_dump_json()
        
        manager = SyncManager(
            db=temp_db,
            adapter=mock_adapter,
            encryption=mock_encryption,
            project_id=project.id,
        )
        
        result = manager.import_memories()
        
        assert result.imported == 1
        
        # Verify memory was saved
        saved = temp_db.get_memory(new_memory.id)
        assert saved is not None
        assert saved.content == "Remote memory"


@pytest.mark.skipif(not HAS_SYNC, reason="sync dependencies not installed") 
class TestSyncIntegrity:
    """Tests for data integrity verification."""
    
    def test_checksum_mismatch_raises_error(self, temp_db, project, mock_adapter, mock_encryption):
        """Test that checksum mismatch raises integrity error."""
        import hashlib
        
        new_memory = Memory(
            id=uuid4(),
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        
        json_data = new_memory.model_dump_json()
        wrong_checksum = "definitely_wrong_checksum"
        
        remote_data = json.dumps({
            "id": str(new_memory.id),
            "project_id": str(new_memory.project_id),
            "updated_at": datetime.utcnow().isoformat(),
            "checksum": wrong_checksum,
            "encrypted_data": json_data,
        })
        
        mock_adapter.list_files.return_value = [f"{new_memory.id}.json"]
        mock_adapter.read_file.return_value = remote_data
        mock_encryption.decrypt.return_value = json_data
        
        manager = SyncManager(
            db=temp_db,
            adapter=mock_adapter,
            encryption=mock_encryption,
            project_id=project.id,
        )
        
        result = manager.import_memories()
        
        # Should have an error due to checksum mismatch
        assert len(result.errors) == 1
        assert "checksum" in result.errors[0].lower() or "integrity" in result.errors[0].lower()
