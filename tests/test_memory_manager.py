"""
Tests for the Memory Manager.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.memory_manager import MemoryManager
from memoryforge.core.validation import ValidationError


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = Mock()
    mock.generate.return_value = [0.1] * 1536  # Fake embedding
    return mock


@pytest.fixture
def temp_storage():
    """Create temporary storage directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sqlite_path = Path(tmpdir) / "test.db"
        qdrant_path = Path(tmpdir) / "qdrant"
        qdrant_path.mkdir()
        yield sqlite_path, qdrant_path


@pytest.fixture
def memory_manager(temp_storage, mock_embedding_service):
    """Create a memory manager with mock services."""
    sqlite_path, qdrant_path = temp_storage
    
    db = SQLiteDatabase(sqlite_path)
    
    # Create a test project
    project = Project(name="test", root_path="/test")
    db.create_project(project)
    
    # Use mock for Qdrant to avoid actual vector operations
    mock_qdrant = Mock(spec=QdrantStore)
    mock_qdrant.upsert.return_value = "vector-id"
    mock_qdrant.delete.return_value = True
    
    manager = MemoryManager(
        sqlite_db=db,
        qdrant_store=mock_qdrant,
        embedding_service=mock_embedding_service,
        project_id=project.id,
    )
    
    return manager


class TestMemoryCreation:
    """Tests for creating memories."""
    
    def test_create_memory_unconfirmed(self, memory_manager):
        """Test creating an unconfirmed memory."""
        memory = memory_manager.create_memory(
            content="We use PostgreSQL",
            memory_type=MemoryType.STACK,
        )
        
        assert memory is not None
        assert memory.content == "We use PostgreSQL"
        assert memory.type == MemoryType.STACK
        assert memory.confirmed is False
    
    def test_create_memory_auto_confirm(self, memory_manager, mock_embedding_service):
        """Test creating a memory with auto-confirm."""
        memory = memory_manager.create_memory(
            content="We prefer pytest",
            memory_type=MemoryType.CONVENTION,
            auto_confirm=True,
        )
        
        assert memory.confirmed is True
        mock_embedding_service.generate.assert_called_once()
    
    def test_create_memory_with_metadata(self, memory_manager):
        """Test creating a memory with metadata."""
        memory = memory_manager.create_memory(
            content="Architecture decision",
            memory_type=MemoryType.DECISION,
            metadata={"reason": "performance"},
        )
        
        assert memory.metadata == {"reason": "performance"}
    
    def test_create_memory_validates_content(self, memory_manager):
        """Test that empty content is rejected."""
        # Pydantic raises ValidationError first for empty content
        with pytest.raises(Exception):
            memory_manager.create_memory(
                content="",
                memory_type=MemoryType.NOTE,
            )
    
    def test_create_memory_sanitizes_content(self, memory_manager):
        """Test that content is sanitized."""
        memory = memory_manager.create_memory(
            content="  \n\n  Whitespace test  \n\n  ",
            memory_type=MemoryType.NOTE,
        )
        
        assert memory.content == "Whitespace test"


class TestMemoryConfirmation:
    """Tests for confirming memories."""
    
    def test_confirm_memory(self, memory_manager, mock_embedding_service):
        """Test confirming a pending memory."""
        memory = memory_manager.create_memory(
            content="Pending memory",
            memory_type=MemoryType.NOTE,
        )
        
        assert memory.confirmed is False
        
        success = memory_manager.confirm_memory(memory.id)
        
        assert success is True
        mock_embedding_service.generate.assert_called_once()
    
    def test_confirm_nonexistent_memory(self, memory_manager):
        """Test confirming a memory that doesn't exist."""
        fake_id = uuid4()
        
        success = memory_manager.confirm_memory(fake_id)
        
        assert success is False
    
    def test_confirm_already_confirmed(self, memory_manager):
        """Test confirming an already confirmed memory."""
        memory = memory_manager.create_memory(
            content="Already confirmed",
            memory_type=MemoryType.NOTE,
            auto_confirm=True,
        )
        
        # Should return True and not re-embed
        success = memory_manager.confirm_memory(memory.id)
        
        assert success is True


class TestMemoryDeletion:
    """Tests for deleting memories."""
    
    def test_delete_unconfirmed_memory(self, memory_manager):
        """Test deleting an unconfirmed memory."""
        memory = memory_manager.create_memory(
            content="To delete",
            memory_type=MemoryType.NOTE,
        )
        
        success = memory_manager.delete_memory(memory.id)
        
        assert success is True
        assert memory_manager.get_memory(memory.id) is None
    
    def test_delete_confirmed_memory(self, memory_manager):
        """Test deleting a confirmed memory (removes from Qdrant too)."""
        memory = memory_manager.create_memory(
            content="Confirmed to delete",
            memory_type=MemoryType.NOTE,
            auto_confirm=True,
        )
        
        success = memory_manager.delete_memory(memory.id)
        
        assert success is True
        memory_manager.vector_store.delete.assert_called_once()
    
    def test_delete_nonexistent_memory(self, memory_manager):
        """Test deleting a memory that doesn't exist."""
        fake_id = uuid4()
        
        success = memory_manager.delete_memory(fake_id)
        
        assert success is False


class TestMemoryListing:
    """Tests for listing memories."""
    
    def test_list_memories(self, memory_manager):
        """Test listing all memories."""
        # Create some memories
        for i in range(3):
            memory_manager.create_memory(
                content=f"Memory {i}",
                memory_type=MemoryType.NOTE,
                auto_confirm=True,
            )
        
        memories = memory_manager.list_memories()
        
        assert len(memories) == 3
    
    def test_list_memories_by_type(self, memory_manager):
        """Test filtering memories by type."""
        memory_manager.create_memory(
            content="Stack memory",
            memory_type=MemoryType.STACK,
            auto_confirm=True,
        )
        memory_manager.create_memory(
            content="Note memory",
            memory_type=MemoryType.NOTE,
            auto_confirm=True,
        )
        
        stack_memories = memory_manager.list_memories(memory_type=MemoryType.STACK)
        
        assert len(stack_memories) == 1
        assert stack_memories[0].type == MemoryType.STACK
    
    def test_get_memory_count(self, memory_manager):
        """Test counting memories."""
        for i in range(5):
            memory_manager.create_memory(
                content=f"Memory {i}",
                memory_type=MemoryType.NOTE,
                auto_confirm=True,
            )
        
        count = memory_manager.get_memory_count()
        
        assert count == 5
