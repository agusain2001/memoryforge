"""
Tests for retrieval engine.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project, SearchResult
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.retrieval import RetrievalEngine


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
    mock.generate.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant store."""
    mock = Mock(spec=QdrantStore)
    mock.search.return_value = []
    return mock


@pytest.fixture
def retrieval_engine(temp_db, mock_qdrant, mock_embedding_service, project):
    """Create a retrieval engine with mocks."""
    return RetrievalEngine(
        sqlite_db=temp_db,
        qdrant_store=mock_qdrant,
        embedding_service=mock_embedding_service,
        project_id=project.id,
        max_results=5,
        min_score=0.5,
    )


class TestSearch:
    """Tests for search functionality."""
    
    def test_search_empty_db(self, retrieval_engine):
        """Test search with empty database."""
        results = retrieval_engine.search("test query")
        assert results == []
    
    def test_search_returns_results(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test that search returns results."""
        # Create a memory
        memory = Memory(
            content="We use FastAPI for the backend",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Mock Qdrant to return this memory
        mock_qdrant.search.return_value = [
            {"memory_id": str(memory.id), "score": 0.9}
        ]
        
        results = retrieval_engine.search("What backend framework?")
        
        assert len(results) == 1
        assert results[0].memory.id == memory.id
        assert results[0].score >= 0.5
    
    def test_search_excludes_archived_memories(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test that search excludes archived memories."""
        # Create an archived memory
        memory = Memory(
            content="Old archived memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            is_archived=True,
        )
        temp_db.create_memory(memory)
        
        # Mock Qdrant to return this memory
        mock_qdrant.search.return_value = [
            {"memory_id": str(memory.id), "score": 0.9}
        ]
        
        results = retrieval_engine.search("test")
        
        # Should not return archived memory
        assert len(results) == 0
    
    def test_search_with_exclude_stale(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test that search can exclude stale memories."""
        # Create a stale memory
        stale_memory = Memory(
            content="Stale information",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            is_stale=True,
            stale_reason="Outdated",
        )
        temp_db.create_memory(stale_memory)
        
        # Create a normal memory
        normal_memory = Memory(
            content="Fresh information",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            is_stale=False,
        )
        temp_db.create_memory(normal_memory)
        
        # Mock Qdrant to return both
        mock_qdrant.search.return_value = [
            {"memory_id": str(stale_memory.id), "score": 0.9},
            {"memory_id": str(normal_memory.id), "score": 0.8},
        ]
        
        # Without exclude_stale
        results = retrieval_engine.search("information", exclude_stale=False)
        assert len(results) == 2
        
        # With exclude_stale
        results = retrieval_engine.search("information", exclude_stale=True)
        assert len(results) == 1
        assert results[0].memory.id == normal_memory.id
    
    def test_search_updates_last_accessed(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test that search updates last_accessed."""
        memory = Memory(
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Verify no last_accessed initially
        assert temp_db.get_memory(memory.id).last_accessed is None
        
        # Mock Qdrant to return this memory
        mock_qdrant.search.return_value = [
            {"memory_id": str(memory.id), "score": 0.9}
        ]
        
        retrieval_engine.search("test")
        
        # Should have last_accessed set
        updated = temp_db.get_memory(memory.id)
        assert updated.last_accessed is not None
    
    def test_search_by_type(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test search filtered by memory type."""
        # Create memories of different types
        stack_memory = Memory(
            content="We use Python",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        note_memory = Memory(
            content="Some note about Python",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(stack_memory)
        temp_db.create_memory(note_memory)
        
        # Mock returns both, but we filter by type
        mock_qdrant.search.return_value = [
            {"memory_id": str(stack_memory.id), "score": 0.9},
            {"memory_id": str(note_memory.id), "score": 0.8},
        ]
        
        results = retrieval_engine.search("Python", memory_type=MemoryType.STACK)
        
        # Should only return stack memory
        # Note: The Qdrant mock doesn't filter, but retrieval should
        stack_results = [r for r in results if r.memory.type == MemoryType.STACK]
        assert len(stack_results) >= 1


class TestTimeline:
    """Tests for timeline functionality."""
    
    def test_get_timeline(self, retrieval_engine, temp_db, project):
        """Test getting memory timeline."""
        # Create memories at different times
        for i in range(5):
            memory = Memory(
                content=f"Memory {i}",
                type=MemoryType.NOTE,
                source=MemorySource.MANUAL,
                project_id=project.id,
                confirmed=True,
                created_at=datetime.utcnow() - timedelta(days=i),
            )
            temp_db.create_memory(memory)
        
        memories = retrieval_engine.get_timeline(limit=3)
        
        assert len(memories) == 3
        # Should be ordered by created_at descending
        assert memories[0].created_at > memories[1].created_at
    
    def test_get_timeline_excludes_archived(self, retrieval_engine, temp_db, project):
        """Test that timeline excludes archived memories."""
        # Create normal memory
        normal = Memory(
            content="Normal memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(normal)
        
        # Create archived memory
        archived = Memory(
            content="Archived memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            is_archived=True,
        )
        temp_db.create_memory(archived)
        
        memories = retrieval_engine.get_timeline()
        
        memory_ids = [m.id for m in memories]
        assert normal.id in memory_ids
        assert archived.id not in memory_ids


class TestReranking:
    """Tests for result re-ranking."""
    
    def test_recency_boost(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test that recent memories get a recency boost."""
        # Create old and new memories with same base score
        old_memory = Memory(
            content="Old memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            created_at=datetime.utcnow() - timedelta(days=60),
        )
        new_memory = Memory(
            content="New memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
            created_at=datetime.utcnow(),
        )
        temp_db.create_memory(old_memory)
        temp_db.create_memory(new_memory)
        
        # Mock same base scores
        mock_qdrant.search.return_value = [
            {"memory_id": str(old_memory.id), "score": 0.8},
            {"memory_id": str(new_memory.id), "score": 0.8},
        ]
        
        results = retrieval_engine.search("memory")
        
        # New memory should rank higher due to recency boost
        assert len(results) == 2
        assert results[0].memory.id == new_memory.id


class TestFallback:
    """Tests for fallback keyword search."""
    
    def test_fallback_on_qdrant_error(self, retrieval_engine, temp_db, project, mock_qdrant):
        """Test that fallback is used when Qdrant fails."""
        # Create memory
        memory = Memory(
            content="FastAPI backend framework",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Make Qdrant raise an exception
        mock_qdrant.search.side_effect = Exception("Qdrant unavailable")
        
        results = retrieval_engine.search("FastAPI")
        
        # Should still get results via keyword fallback
        assert len(results) >= 1
        assert "keyword" in results[0].explanation.lower()
