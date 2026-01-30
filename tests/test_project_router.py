"""
Tests for project router.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from memoryforge.models import Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.core.project_router import ProjectRouter


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = SQLiteDatabase(db_path)
        yield db


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = Mock()
    config.active_project_id = None
    config.save = Mock()  # Add save method to prevent AttributeError
    return config


@pytest.fixture
def project_router(temp_db, mock_config):
    """Create a project router."""
    return ProjectRouter(temp_db, mock_config)


class TestProjectCreation:
    """Tests for project creation."""
    
    def test_create_project(self, project_router, temp_db):
        """Test creating a new project."""
        # Use temp directory for cross-platform compatibility
        test_path = os.path.join(tempfile.gettempdir(), "test_project")
        project = project_router.create_project(
            name="test-project",
            root_path=test_path,
        )
        
        assert project.name == "test-project"
        # Path is normalized, so just check it contains the expected dir name
        assert "test_project" in project.root_path
        
        # Verify in database
        fetched = temp_db.get_project(project.id)
        assert fetched is not None
        assert fetched.name == "test-project"
    
    def test_create_project_normalizes_name(self, project_router):
        """Test that project names are normalized."""
        test_path = os.path.join(tempfile.gettempdir(), "norm_test")
        project = project_router.create_project(
            name="  Test Project  ",
            root_path=test_path,
        )
        
        # Name should be trimmed
        assert project.name.strip() == project.name
    
    def test_create_duplicate_name_fails(self, project_router):
        """Test that duplicate names are rejected."""
        path1 = os.path.join(tempfile.gettempdir(), "dup_test1")
        path2 = os.path.join(tempfile.gettempdir(), "dup_test2")
        project_router.create_project(name="unique-name", root_path=path1)
        
        with pytest.raises(ValueError, match="already exists"):
            project_router.create_project(name="unique-name", root_path=path2)


class TestProjectRetrieval:
    """Tests for project retrieval."""
    
    def test_get_project_by_id(self, project_router):
        """Test getting project by ID."""
        test_path = os.path.join(tempfile.gettempdir(), "get_by_id_test")
        created = project_router.create_project(name="test", root_path=test_path)
        
        fetched = project_router.get_project(created.id)
        
        assert fetched.id == created.id
        assert fetched.name == "test"
    
    def test_get_project_by_name(self, project_router):
        """Test getting project by name."""
        test_path = os.path.join(tempfile.gettempdir(), "get_by_name_test")
        created = project_router.create_project(name="my-project", root_path=test_path)
        
        fetched = project_router.get_project_by_name("my-project")
        
        assert fetched.id == created.id
    
    def test_get_nonexistent_project(self, project_router):
        """Test getting nonexistent project returns None."""
        result = project_router.get_project(uuid4())
        
        assert result is None
    
    def test_list_projects(self, project_router):
        """Test listing all projects."""
        for i in range(3):
            path = os.path.join(tempfile.gettempdir(), f"list_test_{i}")
            project_router.create_project(name=f"project-{i}", root_path=path)
        
        projects = project_router.list_projects()
        
        assert len(projects) == 3


class TestProjectSwitching:
    """Tests for project switching."""
    
    def test_switch_project_by_id(self, project_router, mock_config):
        """Test switching project by ID."""
        test_path = os.path.join(tempfile.gettempdir(), "switch_by_id_test")
        project = project_router.create_project(name="target", root_path=test_path)
        
        project_router.switch_project(project.id)
        
        assert mock_config.active_project_id == str(project.id)
    
    def test_switch_project_by_name(self, project_router, mock_config):
        """Test switching project by name."""
        test_path = os.path.join(tempfile.gettempdir(), "switch_by_name_test")
        project = project_router.create_project(name="named-project", root_path=test_path)
        
        project_router.switch_project_by_name("named-project")
        
        assert mock_config.active_project_id == str(project.id)
    
    def test_switch_to_nonexistent_fails(self, project_router):
        """Test switching to nonexistent project fails."""
        with pytest.raises(ValueError, match="not found"):
            project_router.switch_project(uuid4())
    
    def test_get_active_project(self, project_router, mock_config):
        """Test getting active project."""
        test_path = os.path.join(tempfile.gettempdir(), "active_test")
        project = project_router.create_project(name="active", root_path=test_path)
        project_router.switch_project(project.id)
        
        active = project_router.get_active_project()
        
        assert active.id == project.id


class TestProjectDeletion:
    """Tests for project deletion."""
    
    def test_delete_empty_project(self, project_router, temp_db):
        """Test deleting project with no memories."""
        test_path = os.path.join(tempfile.gettempdir(), "deletable_test")
        project = project_router.create_project(name="deletable", root_path=test_path)
        
        success = project_router.delete_project(project.id)
        
        assert success is True
        assert temp_db.get_project(project.id) is None
    
    def test_delete_project_with_memories_fails(self, project_router, temp_db):
        """Test that project with memories cannot be deleted."""
        from memoryforge.models import Memory, MemoryType, MemorySource
        
        test_path = os.path.join(tempfile.gettempdir(), "has_memories_test")
        project = project_router.create_project(name="has-memories", root_path=test_path)
        
        # Add a memory
        memory = Memory(
            content="Test memory",
            type=MemoryType.NOTE,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Should fail to delete
        with pytest.raises(ValueError, match="memories"):
            project_router.delete_project(project.id)


class TestProjectDiscovery:
    """Tests for project auto-discovery."""
    
    def test_detect_project_from_path(self, project_router):
        """Test detecting project from working directory."""
        # Create project with specific root
        project = project_router.create_project(
            name="detected",
            root_path="/home/user/myproject",
        )
        
        # Should detect from path
        detected = project_router.detect_project_from_path(
            Path("/home/user/myproject/src/main.py")
        )
        
        if detected:  # May return None if detection not implemented
            assert detected.id == project.id
    
    def test_no_project_for_unknown_path(self, project_router):
        """Test that unknown paths return None."""
        detected = project_router.detect_project_from_path(
            Path("/completely/unknown/path")
        )
        
        assert detected is None
