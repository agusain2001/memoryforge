"""
Shared pytest fixtures for MemoryForge integration tests.
"""

import pytest
from pathlib import Path
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.models import Memory, MemoryType, MemorySource, Project


@pytest.fixture
def temp_db(tmp_path):
    """Initialized SQLiteDatabase at a temp path."""
    db_path = tmp_path / "db" / "memoryforge.db"
    return SQLiteDatabase(db_path)


@pytest.fixture
def temp_project(temp_db):
    """A Project saved to temp_db."""
    project = Project(name="test-project", root_path="/tmp/test")
    temp_db.create_project(project)
    return project


@pytest.fixture
def sample_memory(temp_db, temp_project):
    """A confirmed Memory saved to temp_db."""
    memory = Memory(
        content="We use FastAPI for the backend and PostgreSQL for storage",
        type=MemoryType.STACK,
        source=MemorySource.MANUAL,
        project_id=temp_project.id,
        confirmed=True,
    )
    temp_db.create_memory(memory)
    return memory


@pytest.fixture
def sync_manager(temp_db, temp_project, tmp_path, encryption_key):
    """A SyncManager wired to a temp directory."""
    from memoryforge.sync.encryption import EncryptionLayer
    from memoryforge.sync.local_file_adapter import LocalFileAdapter
    from memoryforge.sync.manager import SyncManager

    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()

    encryption = EncryptionLayer(encryption_key)
    adapter = LocalFileAdapter(sync_dir)
    return SyncManager(
        db=temp_db,
        adapter=adapter,
        encryption=encryption,
        project_id=temp_project.id,
    )


@pytest.fixture
def encryption_key():
    """A fresh Fernet key."""
    pytest.importorskip("cryptography")
    from memoryforge.sync.encryption import EncryptionLayer
    return EncryptionLayer.generate_key()
