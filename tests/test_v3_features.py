"""
Tests for MemoryForge v3 features.

Tests cover:
- Graph Memory (memory relationships)
- Confidence Scoring
- Conflict Resolution
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
import tempfile

from memoryforge.models import (
    Memory, MemoryType, MemorySource, Project,
    MemoryRelation, RelationType,
    ConflictLog, ConflictResolution,
)
from memoryforge.storage.sqlite_db import SQLiteDatabase


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    database = SQLiteDatabase(db_path)
    return database


@pytest.fixture
def project_id(db):
    """Create a test project."""
    project = Project(name="test-project", root_path="/test")
    db.create_project(project)
    return project.id


@pytest.fixture
def sample_memories(db, project_id):
    """Create sample memories for testing."""
    memories = []
    for i, mem_type in enumerate([MemoryType.DECISION, MemoryType.CONSTRAINT, MemoryType.STACK]):
        memory = Memory(
            content=f"Test memory {i}: This is a {mem_type.value} memory",
            type=mem_type,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confirmed=True,
            confidence_score=0.8 + (i * 0.05),
        )
        db.create_memory(memory)
        memories.append(memory)
    return memories


# ============================================================================
# Graph Memory Tests
# ============================================================================

class TestGraphMemory:
    """Tests for memory-to-memory relationships."""

    def test_create_memory_relation(self, db, sample_memories):
        """Test creating a memory relation."""
        source = sample_memories[0]
        target = sample_memories[1]
        
        relation = db.create_memory_relation(
            source_memory_id=source.id,
            target_memory_id=target.id,
            relation_type=RelationType.CAUSED_BY,
            created_by="human",
        )
        
        assert relation.source_memory_id == source.id
        assert relation.target_memory_id == target.id
        assert relation.relation_type == RelationType.CAUSED_BY

    def test_get_memory_relations(self, db, sample_memories):
        """Test getting relations for a memory."""
        source = sample_memories[0]
        target = sample_memories[1]
        
        db.create_memory_relation(
            source_memory_id=source.id,
            target_memory_id=target.id,
            relation_type=RelationType.SUPERSEDES,
        )
        
        # Get outgoing
        outgoing = db.get_memory_relations(source.id, direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0].target_memory_id == target.id
        
        # Get incoming
        incoming = db.get_memory_relations(target.id, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].source_memory_id == source.id

    def test_delete_memory_relation(self, db, sample_memories):
        """Test deleting a memory relation."""
        source = sample_memories[0]
        target = sample_memories[1]
        
        relation = db.create_memory_relation(
            source_memory_id=source.id,
            target_memory_id=target.id,
            relation_type=RelationType.RELATES_TO,
        )
        
        # Delete relation
        result = db.delete_memory_relation(relation.id)
        assert result is True
        
        # Verify deletion
        relations = db.get_memory_relations(source.id)
        assert len(relations) == 0

    def test_all_relation_types(self, db, sample_memories):
        """Test all relation types can be created."""
        source = sample_memories[0]
        target = sample_memories[1]
        
        created = []
        for rel_type in RelationType:
            relation = db.create_memory_relation(
                source_memory_id=source.id,
                target_memory_id=target.id,
                relation_type=rel_type,
            )
            created.append(relation)
        
        relations = db.get_memory_relations(source.id, direction="outgoing")
        assert len(relations) == len(RelationType)


class TestGraphBuilder:
    """Tests for the GraphBuilder component."""

    def test_link_memories(self, db, sample_memories):
        """Test linking memories via GraphBuilder."""
        from memoryforge.core.graph_builder import GraphBuilder
        
        builder = GraphBuilder(db)
        source = sample_memories[0]
        target = sample_memories[1]
        
        relation = builder.link_memories(
            source_id=source.id,
            target_id=target.id,
            relation_type=RelationType.DEPENDS_ON,
            created_by="test",
        )
        
        assert relation.source_memory_id == source.id
        assert relation.target_memory_id == target.id
        assert relation.relation_type == RelationType.DEPENDS_ON

    def test_get_graph_view(self, db, sample_memories):
        """Test getting a graph view for a memory."""
        from memoryforge.core.graph_builder import GraphBuilder
        
        builder = GraphBuilder(db)
        central = sample_memories[1]
        source = sample_memories[0]
        target = sample_memories[2]
        
        # Create incoming and outgoing relations
        builder.link_memories(source.id, central.id, RelationType.CAUSED_BY)
        builder.link_memories(central.id, target.id, RelationType.SUPERSEDES)
        
        view = builder.get_graph_view(central.id)
        
        assert view["memory"].id == central.id
        assert len(view["incoming"]) == 1
        assert len(view["outgoing"]) == 1

    def test_link_nonexistent_memory_fails(self, db, sample_memories):
        """Test that linking to nonexistent memory raises error."""
        from memoryforge.core.graph_builder import GraphBuilder
        
        builder = GraphBuilder(db)
        
        with pytest.raises(ValueError):
            builder.link_memories(
                source_id=sample_memories[0].id,
                target_id=uuid4(),  # Nonexistent
                relation_type=RelationType.RELATES_TO,
            )


# ============================================================================
# Confidence Scoring Tests
# ============================================================================

class TestConfidenceScoring:
    """Tests for memory confidence scoring."""

    def test_memory_has_confidence_score(self, db, project_id):
        """Test that memories have confidence_score field."""
        memory = Memory(
            content="Test memory with confidence",
            type=MemoryType.DECISION,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confidence_score=0.75,
        )
        db.create_memory(memory)
        
        retrieved = db.get_memory(memory.id)
        assert retrieved.confidence_score == 0.75

    def test_confidence_default_value(self, db, project_id):
        """Test that confidence defaults to 1.0."""
        memory = Memory(
            content="Test memory default confidence",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project_id,
        )
        db.create_memory(memory)
        
        retrieved = db.get_memory(memory.id)
        assert retrieved.confidence_score == 1.0

    def test_update_confidence_score(self, db, sample_memories):
        """Test updating confidence score."""
        memory = sample_memories[0]
        
        db.update_confidence_score(memory.id, 0.5)
        
        retrieved = db.get_memory(memory.id)
        assert retrieved.confidence_score == 0.5


class TestConfidenceScorer:
    """Tests for the ConfidenceScorer component."""

    def test_calculate_score(self, db, sample_memories):
        """Test confidence score calculation."""
        from memoryforge.core.confidence_scorer import ConfidenceScorer
        
        scorer = ConfidenceScorer(db)
        memory = sample_memories[0]
        
        # Pass the full memory object, not just ID
        score = scorer.calculate_score(memory)
        assert 0.0 <= score <= 1.0

    def test_confirmed_memory_gets_boost(self, db, project_id):
        """Test that confirmed memories get higher scores."""
        from memoryforge.core.confidence_scorer import ConfidenceScorer
        
        # Create unconfirmed memory
        unconfirmed = Memory(
            content="Unconfirmed test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confirmed=False,
        )
        db.create_memory(unconfirmed)
        
        # Create confirmed memory
        confirmed = Memory(
            content="Confirmed test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confirmed=True,
        )
        db.create_memory(confirmed)
        
        scorer = ConfidenceScorer(db)
        
        # Pass full memory objects
        unconfirmed_score = scorer.calculate_score(unconfirmed)
        confirmed_score = scorer.calculate_score(confirmed)
        
        assert confirmed_score > unconfirmed_score

    def test_get_low_confidence(self, db, project_id):
        """Test getting low confidence memories."""
        from memoryforge.core.confidence_scorer import ConfidenceScorer
        
        # Create memories with varying confidence
        low_conf = Memory(
            content="Low confidence memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confidence_score=0.3,
        )
        db.create_memory(low_conf)
        
        high_conf = Memory(
            content="High confidence memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confidence_score=0.9,
        )
        db.create_memory(high_conf)
        
        scorer = ConfidenceScorer(db)
        low_memories = scorer.get_low_confidence(project_id, threshold=0.5)
        
        assert len(low_memories) == 1
        assert low_memories[0].id == low_conf.id


# ============================================================================
# Conflict Resolution Tests
# ============================================================================

class TestConflictLogging:
    """Tests for conflict logging."""

    def test_log_conflict(self, db, sample_memories):
        """Test logging a sync conflict."""
        memory = sample_memories[0]
        
        conflict = db.log_conflict(
            memory_id=memory.id,
            local_content="Local version",
            remote_content="Remote version",
            resolution=ConflictResolution.LOCAL_WINS,
            resolved_by="user",
        )
        
        assert conflict.memory_id == memory.id
        assert conflict.resolution == ConflictResolution.LOCAL_WINS

    def test_get_conflicts_for_memory(self, db, sample_memories):
        """Test getting conflicts for a specific memory."""
        memory = sample_memories[0]
        
        db.log_conflict(
            memory_id=memory.id,
            local_content="Local",
            remote_content="Remote",
            resolution=ConflictResolution.MERGED,
        )
        
        conflicts = db.get_conflict_history(memory.id)
        assert len(conflicts) == 1
        assert conflicts[0].resolution == ConflictResolution.MERGED

    def test_all_conflict_resolutions(self, db, sample_memories):
        """Test all conflict resolution types."""
        memory = sample_memories[0]
        
        for resolution in ConflictResolution:
            db.log_conflict(
                memory_id=memory.id,
                local_content="Local",
                remote_content="Remote",
                resolution=resolution,
            )
        
        conflicts = db.get_conflict_history(memory.id)
        assert len(conflicts) == len(ConflictResolution)


class TestConflictResolver:
    """Tests for the ConflictResolver component."""

    def test_resolve_last_write_wins(self, db, sample_memories):
        """Test last-write-wins conflict resolution."""
        from memoryforge.core.conflict_resolver import ConflictResolver, SyncConflict
        
        resolver = ConflictResolver(db)
        memory = sample_memories[0]
        
        # Create a conflict where remote is newer
        conflict = SyncConflict(
            memory_id=memory.id,
            local_memory=memory,
            remote_content="Updated remotely",
            remote_updated_at=datetime.utcnow() + timedelta(hours=1),  # Remote is newer
        )
        
        # Last-write-wins should pick remote (more recent)
        result = resolver.resolve_last_write_wins(conflict)
        assert result.resolution == ConflictResolution.REMOTE_WINS

    def test_resolve_manual(self, db, sample_memories):
        """Test manual conflict resolution."""
        from memoryforge.core.conflict_resolver import ConflictResolver, SyncConflict
        
        resolver = ConflictResolver(db)
        memory = sample_memories[0]
        
        conflict = SyncConflict(
            memory_id=memory.id,
            local_memory=memory,
            remote_content="Remote version",
            remote_updated_at=datetime.utcnow(),
        )
        
        result = resolver.resolve_manual(
            conflict,
            merged_content="Manually merged content",
            resolved_by="test_user",
        )
        
        # Check conflict was logged
        assert result.resolution == ConflictResolution.MANUAL
        
        conflicts = resolver.list_conflicts(memory.id)
        assert len(conflicts) == 1
        assert conflicts[0].resolution == ConflictResolution.MANUAL


# ============================================================================
# Schema Version Tests
# ============================================================================

class TestSchemaVersion:
    """Tests for v3 schema changes."""

    def test_schema_version_is_3(self):
        """Test that schema version is 3."""
        from memoryforge.storage.sqlite_db import SCHEMA_VERSION
        assert SCHEMA_VERSION == 3

    def test_memory_relations_table_exists(self, db):
        """Test that memory_relations table exists."""
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_relations'"
            )
            result = cursor.fetchone()
            assert result is not None

    def test_conflict_log_table_exists(self, db):
        """Test that conflict_log table exists."""
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conflict_log'"
            )
            result = cursor.fetchone()
            assert result is not None

    def test_confidence_score_column_exists(self, db, project_id):
        """Test that confidence_score column exists in memories."""
        memory = Memory(
            content="Test for confidence column",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project_id,
            confidence_score=0.5,
        )
        db.create_memory(memory)
        
        retrieved = db.get_memory(memory.id)
        assert hasattr(retrieved, 'confidence_score')
        assert retrieved.confidence_score == 0.5
