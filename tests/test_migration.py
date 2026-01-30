"""
Tests for database migration.
"""

import gc
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.migrate import Migrator, MigrationError, MigrationVerificationError


def create_v1_database(db_path: Path) -> None:
    """Create a v1-style database at the given path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        
        # Create v1 tables (without v2 columns)
        cursor.execute("""
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                confirmed INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE embeddings (
                memory_id TEXT PRIMARY KEY,
                vector_id TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)
        
        # Create a project
        project_id = str(uuid4())
        cursor.execute(
            "INSERT INTO projects (id, name, root_path, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "test-project", "/test", datetime.utcnow().isoformat())
        )
        
        # Create some memories
        for i in range(5):
            cursor.execute(
                """INSERT INTO memories 
                   (id, project_id, content, type, source, created_at, confirmed, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()),
                    project_id,
                    f"Memory {i}",
                    "note",
                    "manual",
                    datetime.utcnow().isoformat(),
                    1,
                    "{}",
                )
            )
        
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def v1_database(tmp_path):
    """Create a v1-style database (without v2 columns) with unique tmp path."""
    # Create a mock config
    config = Mock()
    config.sqlite_path = tmp_path / "test.db"
    config.qdrant_path = tmp_path / "qdrant"
    
    # Create the database
    create_v1_database(config.sqlite_path)
    
    yield config
    
    # Cleanup - force garbage collection to release file handles
    gc.collect()


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config for testing (v2 database)."""
    config = Mock()
    config.sqlite_path = tmp_path / "test_backup.db"
    config.qdrant_path = tmp_path / "qdrant"
    yield config
    gc.collect()


class TestMigrationV1ToV2:
    """Tests for v1 to v2 migration."""
    
    def test_migration_detects_v1_database(self, v1_database):
        """Test that migrator correctly detects v1 database."""
        migrator = Migrator(v1_database)
        version = migrator._get_schema_version()
        assert version == 1
    
    def test_migration_creates_backup(self, v1_database):
        """Test that migration creates a backup file."""
        migrator = Migrator(v1_database)
        
        backup_path = migrator.backup_database()
        
        assert backup_path.exists()
        assert "backup" in backup_path.name
    
    def test_migration_preserves_all_memories(self, v1_database):
        """Test that migration preserves all memories."""
        migrator = Migrator(v1_database)
        
        # Count memories before migration
        with sqlite3.connect(v1_database.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            pre_count = cursor.fetchone()[0]
        
        # Run migration
        success, error = migrator.run_migration(verify=True)
        
        assert success is True
        assert error is None
        
        # Count memories after migration
        with sqlite3.connect(v1_database.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            post_count = cursor.fetchone()[0]
        
        assert post_count == pre_count
    
    def test_migration_adds_v2_columns(self, v1_database):
        """Test that migration adds v2 columns."""
        migrator = Migrator(v1_database)
        
        success, error = migrator.run_migration()
        
        assert success is True
        
        # Check for v2 columns
        with sqlite3.connect(v1_database.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}
        
        assert "is_stale" in columns
        assert "stale_reason" in columns
        assert "last_accessed" in columns
        assert "is_archived" in columns
        assert "consolidated_into" in columns
    
    def test_migration_creates_v2_tables(self, v1_database):
        """Test that migration creates v2 tables."""
        migrator = Migrator(v1_database)
        
        success, error = migrator.run_migration()
        
        assert success is True
        
        # Check for v2 tables
        with sqlite3.connect(v1_database.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
        
        assert "memory_versions" in tables
        assert "memory_links" in tables
        assert "schema_version" in tables
    
    def test_migration_updates_version(self, v1_database):
        """Test that migration updates schema version."""
        migrator = Migrator(v1_database)
        
        success, error = migrator.run_migration()
        
        assert success is True
        
        version = migrator._get_schema_version()
        assert version == 3  # Latest version
    
    def test_migration_idempotent(self, v1_database):
        """Test that running migration twice is safe."""
        migrator = Migrator(v1_database)
        
        # First migration
        success1, _ = migrator.run_migration()
        assert success1 is True
        
        # Second migration (should be no-op)
        success2, _ = migrator.run_migration()
        assert success2 is True
        
        # Still at v3 (latest)
        assert migrator._get_schema_version() == 3


class TestMigrationRollback:
    """Tests for migration rollback."""
    
    def test_rollback_restores_backup(self, v1_database):
        """Test that rollback restores from backup."""
        migrator = Migrator(v1_database)
        
        # Create backup
        backup_path = migrator.backup_database()
        
        # Corrupt the original (simulate failed migration)
        with sqlite3.connect(v1_database.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE memories")
            conn.commit()
        
        # Restore
        migrator.restore_backup(backup_path)
        
        # Verify restoration
        with sqlite3.connect(v1_database.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
        
        assert count == 5  # Original 5 memories


class TestBackupManagement:
    """Tests for backup cleanup."""
    
    def test_cleanup_old_backups(self, temp_config):
        """Test cleanup of old backups."""
        # Initialize database
        db = SQLiteDatabase(temp_config.sqlite_path)
        
        migrator = Migrator(temp_config)
        
        # Create multiple backups
        for i in range(7):
            backup_path = temp_config.sqlite_path.parent / f"memoryforge_v1_backup_{i:03d}.sqlite"
            shutil.copy(temp_config.sqlite_path, backup_path)
        
        # Should have 7 backups
        assert len(migrator.list_backups()) == 7
        
        # Cleanup keeping only 5
        deleted = migrator.cleanup_old_backups(keep_count=5)
        
        assert deleted == 2
        assert len(migrator.list_backups()) == 5
    
    def test_rollback_warning(self, temp_config):
        """Test rollback warning about data loss."""
        import time
        
        # Initialize database
        db = SQLiteDatabase(temp_config.sqlite_path)
        
        # Create a project
        project = Project(name="test", root_path="/test")
        db.create_project(project)
        
        migrator = Migrator(temp_config)
        
        # Create backup
        backup_path = migrator.backup_database()
        
        # Wait a moment to ensure timestamp difference
        time.sleep(1.1)
        
        # Add new memories after backup
        for i in range(3):
            memory = Memory(
                content=f"New memory {i}",
                type=MemoryType.NOTE,
                source=MemorySource.MANUAL,
                project_id=project.id,
                confirmed=True,
            )
            db.create_memory(memory)
        
        # Get warning - may be None if timing is exact, but should warn about new data
        warning = migrator.get_rollback_warning()
        
        # Warning should exist and mention potential data loss
        assert warning is not None, "Expected rollback warning for memories created after backup"
        assert "memories" in warning.lower() or "3" in warning
