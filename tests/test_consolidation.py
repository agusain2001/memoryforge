"""
Tests for memory consolidation.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.memory_consolidator import (
    MemoryConsolidator,
    ConsolidationSuggestion,
    ConsolidationResult,
)


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
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = Mock()
    mock.generate.return_value = [0.1] * 384  # Fake embedding
    return mock


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant store."""
    mock = Mock(spec=QdrantStore)
    mock.search.return_value = []
    mock.upsert.return_value = "vector-id"
    mock.delete.return_value = True
    return mock


@pytest.fixture
def consolidator(temp_db, mock_qdrant, mock_embedding_service, project):
    """Create a memory consolidator with mocks."""
    return MemoryConsolidator(
        sqlite_db=temp_db,
        qdrant_store=mock_qdrant,
        embedding_service=mock_embedding_service,
        project_id=project.id,
        threshold=0.90,
    )


class TestSimilaritySearch:
    """Tests for similarity search."""
    
    def test_uses_correct_embedding_method(self, consolidator, mock_embedding_service, temp_db, project):
        """Verify generate() is called, not embed()."""
        # Create two test memories so similarity search has something to compare
        memory1 = Memory(
            content="Test memory content one",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory1)
        
        memory2 = Memory(
            content="Test memory content two",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory2)
        
        # Run similarity search
        consolidator.find_similar_pairs()
        
        # Verify generate was called (needs at least 2 memories to compare)
        mock_embedding_service.generate.assert_called()
        
        # Ensure embed() was NOT called (it shouldn't exist or shouldn't be called)
        if hasattr(mock_embedding_service, 'embed'):
            mock_embedding_service.embed.assert_not_called()
    
    def test_find_similar_pairs_empty_db(self, consolidator):
        """Test find_similar_pairs with empty database."""
        pairs = consolidator.find_similar_pairs()
        assert pairs == []
    
    def test_find_similar_pairs_single_memory(self, consolidator, temp_db, project):
        """Test find_similar_pairs with single memory."""
        memory = Memory(
            content="Single memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        pairs = consolidator.find_similar_pairs()
        assert pairs == []  # Need at least 2 memories


class TestConsolidation:
    """Tests for consolidation operations."""
    
    def test_consolidate_requires_two_memories(self, consolidator):
        """Test that consolidation requires at least 2 memories."""
        with pytest.raises(ValueError, match="at least 2"):
            consolidator.consolidate(
                source_ids=[uuid4()],
                merged_content="Merged",
            )
    
    def test_consolidate_validates_memories_exist(self, consolidator):
        """Test that consolidation validates memory existence."""
        with pytest.raises(ValueError, match="not found"):
            consolidator.consolidate(
                source_ids=[uuid4(), uuid4()],
                merged_content="Merged",
            )
    
    def test_consolidate_archives_sources(self, consolidator, temp_db, project):
        """Test that consolidation archives source memories."""
        # Create two memories
        mem1 = Memory(
            content="Memory 1",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        mem2 = Memory(
            content="Memory 2",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(mem1)
        temp_db.create_memory(mem2)
        
        # Consolidate
        result = consolidator.consolidate(
            source_ids=[mem1.id, mem2.id],
            merged_content="Merged memory content",
        )
        
        assert result.consolidated_memory is not None
        assert result.archived_count == 2
        
        # Verify sources are archived
        archived1 = temp_db.get_memory(mem1.id)
        archived2 = temp_db.get_memory(mem2.id)
        
        assert archived1.is_archived is True
        assert archived2.is_archived is True
        assert archived1.consolidated_into == result.consolidated_memory.id
    
    def test_consolidate_creates_new_memory(self, consolidator, temp_db, project):
        """Test that consolidation creates a new memory."""
        mem1 = Memory(
            content="Memory 1",
            type=MemoryType.DECISION,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        mem2 = Memory(
            content="Memory 2",
            type=MemoryType.DECISION,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(mem1)
        temp_db.create_memory(mem2)
        
        merged_content = "This is the merged content"
        result = consolidator.consolidate(
            source_ids=[mem1.id, mem2.id],
            merged_content=merged_content,
        )
        
        # Verify new memory
        new_memory = temp_db.get_memory(result.consolidated_memory.id)
        assert new_memory is not None
        assert new_memory.content == merged_content
        assert new_memory.confirmed is True


class TestRollback:
    """Tests for rollback operations."""
    
    def test_rollback_restores_archived(self, consolidator, temp_db, project, mock_qdrant):
        """Test that rollback restores archived memories."""
        # Create and consolidate
        mem1 = Memory(
            content="Memory 1",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        mem2 = Memory(
            content="Memory 2",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(mem1)
        temp_db.create_memory(mem2)
        
        result = consolidator.consolidate(
            source_ids=[mem1.id, mem2.id],
            merged_content="Merged",
        )
        
        # Rollback
        restored = consolidator.rollback_consolidation(result.consolidated_memory.id)
        
        assert len(restored) == 2
        
        # Verify memories are restored
        restored1 = temp_db.get_memory(mem1.id)
        restored2 = temp_db.get_memory(mem2.id)
        
        assert restored1.is_archived is False
        assert restored2.is_archived is False
        assert restored1.consolidated_into is None
    
    def test_rollback_deletes_consolidated(self, consolidator, temp_db, project):
        """Test that rollback deletes the consolidated memory."""
        mem1 = Memory(
            content="Memory 1",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        mem2 = Memory(
            content="Memory 2",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(mem1)
        temp_db.create_memory(mem2)
        
        result = consolidator.consolidate(
            source_ids=[mem1.id, mem2.id],
            merged_content="Merged",
        )
        
        consolidated_id = result.consolidated_memory.id
        
        # Rollback
        consolidator.rollback_consolidation(consolidated_id)
        
        # Verify consolidated memory is deleted
        assert temp_db.get_memory(consolidated_id) is None


class TestStaleness:
    """Tests for staleness operations."""
    
    def test_mark_stale(self, consolidator, temp_db, project):
        """Test marking a memory as stale."""
        memory = Memory(
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        success = consolidator.mark_stale(memory.id, "Outdated information")
        
        assert success is True
        
        updated = temp_db.get_memory(memory.id)
        assert updated.is_stale is True
        assert updated.stale_reason == "Outdated information"
    
    def test_clear_stale(self, consolidator, temp_db, project):
        """Test clearing stale flag."""
        memory = Memory(
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        consolidator.mark_stale(memory.id, "Outdated")
        consolidator.clear_stale(memory.id)
        
        updated = temp_db.get_memory(memory.id)
        assert updated.is_stale is False
    
    def test_find_unused_memories(self, consolidator, temp_db, project):
        """Test finding unused memories."""
        # Create old memory
        old_memory = Memory(
            content="Old memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            created_at=datetime.utcnow() - timedelta(days=60),
        )
        temp_db.create_memory(old_memory)
        
        # Create recent memory
        new_memory = Memory(
            content="New memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            created_at=datetime.utcnow(),
        )
        temp_db.create_memory(new_memory)
        
        unused = consolidator.find_unused_memories(days_unused=30)
        
        assert len(unused) == 1
        assert unused[0].id == old_memory.id
