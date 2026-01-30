"""
Tests for the SQLite storage layer.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
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


class TestProjectOperations:
    """Tests for project CRUD operations."""
    
    def test_create_project(self, temp_db):
        """Test creating a project."""
        project = Project(
            name="my-project",
            root_path="/path/to/project",
        )
        
        created = temp_db.create_project(project)
        
        assert created.id == project.id
        assert created.name == project.name
        assert created.root_path == project.root_path
    
    def test_get_project_by_id(self, temp_db, project):
        """Test getting a project by ID."""
        retrieved = temp_db.get_project(project.id)
        
        assert retrieved is not None
        assert retrieved.id == project.id
        assert retrieved.name == project.name
    
    def test_get_project_by_name(self, temp_db, project):
        """Test getting a project by name."""
        retrieved = temp_db.get_project_by_name(project.name)
        
        assert retrieved is not None
        assert retrieved.id == project.id
    
    def test_get_nonexistent_project(self, temp_db):
        """Test getting a project that doesn't exist."""
        result = temp_db.get_project(uuid4())
        
        assert result is None
    
    def test_list_projects(self, temp_db):
        """Test listing all projects."""
        # Create multiple projects
        for i in range(3):
            project = Project(
                name=f"project-{i}",
                root_path=f"/path/{i}",
            )
            temp_db.create_project(project)
        
        projects = temp_db.list_projects()
        
        assert len(projects) == 3


class TestMemoryOperations:
    """Tests for memory CRUD operations."""
    
    def test_create_memory(self, temp_db, project):
        """Test creating a memory."""
        memory = Memory(
            content="We use FastAPI for the backend",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
        )
        
        created = temp_db.create_memory(memory)
        
        assert created.id == memory.id
        assert created.content == memory.content
        assert created.type == MemoryType.STACK
        assert created.confirmed is False
    
    def test_get_memory(self, temp_db, project):
        """Test getting a memory by ID."""
        memory = Memory(
            content="Test content",
            type=MemoryType.NOTE,
            source=MemorySource.CHAT,
            project_id=project.id,
        )
        temp_db.create_memory(memory)
        
        retrieved = temp_db.get_memory(memory.id)
        
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content
    
    def test_confirm_memory(self, temp_db, project):
        """Test confirming a memory."""
        memory = Memory(
            content="Pending memory",
            type=MemoryType.DECISION,
            source=MemorySource.MANUAL,
            project_id=project.id,
        )
        temp_db.create_memory(memory)
        
        success = temp_db.confirm_memory(memory.id)
        
        assert success is True
        
        retrieved = temp_db.get_memory(memory.id)
        assert retrieved.confirmed is True
    
    def test_update_memory(self, temp_db, project):
        """Test updating a memory."""
        memory = Memory(
            content="Original content",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
        )
        temp_db.create_memory(memory)
        
        success = temp_db.update_memory(memory.id, "Updated content")
        
        assert success is True
        
        retrieved = temp_db.get_memory(memory.id)
        assert retrieved.content == "Updated content"
    
    def test_delete_memory(self, temp_db, project):
        """Test deleting a memory."""
        memory = Memory(
            content="To be deleted",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
        )
        temp_db.create_memory(memory)
        
        success = temp_db.delete_memory(memory.id)
        
        assert success is True
        assert temp_db.get_memory(memory.id) is None
    
    def test_list_memories(self, temp_db, project):
        """Test listing memories."""
        # Create multiple memories
        for i in range(5):
            memory = Memory(
                content=f"Memory {i}",
                type=MemoryType.NOTE,
                source=MemorySource.MANUAL,
                project_id=project.id,
                confirmed=True,
            )
            temp_db.create_memory(memory)
        
        memories = temp_db.list_memories(
            project_id=project.id,
            confirmed_only=True,
        )
        
        assert len(memories) == 5
    
    def test_list_memories_by_type(self, temp_db, project):
        """Test filtering memories by type."""
        # Create memories of different types
        for mem_type in [MemoryType.STACK, MemoryType.DECISION, MemoryType.NOTE]:
            memory = Memory(
                content=f"{mem_type.value} memory",
                type=mem_type,
                source=MemorySource.MANUAL,
                project_id=project.id,
                confirmed=True,
            )
            temp_db.create_memory(memory)
        
        stack_memories = temp_db.list_memories(
            project_id=project.id,
            memory_type=MemoryType.STACK,
        )
        
        assert len(stack_memories) == 1
        assert stack_memories[0].type == MemoryType.STACK
    
    def test_get_memory_count(self, temp_db, project):
        """Test counting memories."""
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
        
        count = temp_db.get_memory_count(project.id)
        
        assert count == 3


class TestEmbeddingReferences:
    """Tests for embedding reference operations."""
    
    def test_save_embedding_reference(self, temp_db, project):
        """Test saving an embedding reference."""
        memory = Memory(
            content="Test",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
        )
        temp_db.create_memory(memory)
        
        temp_db.save_embedding_reference(memory.id, "vector-123")
        
        vector_id = temp_db.get_embedding_reference(memory.id)
        assert vector_id == "vector-123"
    
    def test_delete_embedding_reference(self, temp_db, project):
        """Test deleting an embedding reference."""
        memory = Memory(
            content="Test",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
        )
        temp_db.create_memory(memory)
        temp_db.save_embedding_reference(memory.id, "vector-456")
        
        success = temp_db.delete_embedding_reference(memory.id)
        
        assert success is True
        assert temp_db.get_embedding_reference(memory.id) is None


class TestRestartRecovery:
    """Tests for restart recovery (data persistence)."""
    
    def test_data_persists_after_reconnect(self):
        """Test that data persists after closing and reopening the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persist.db"
            
            # Create and populate database
            db1 = SQLiteDatabase(db_path)
            project = Project(
                name="persist-test",
                root_path="/test",
            )
            db1.create_project(project)
            
            memory = Memory(
                content="This should persist",
                type=MemoryType.DECISION,
                source=MemorySource.MANUAL,
                project_id=project.id,
                confirmed=True,
            )
            db1.create_memory(memory)
            
            # Reconnect to database
            db2 = SQLiteDatabase(db_path)
            
            # Verify data persisted
            retrieved_project = db2.get_project_by_name("persist-test")
            assert retrieved_project is not None
            
            retrieved_memory = db2.get_memory(memory.id)
            assert retrieved_memory is not None
            assert retrieved_memory.content == "This should persist"
            assert retrieved_memory.confirmed is True
