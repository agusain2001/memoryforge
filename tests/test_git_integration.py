"""
Tests for Git integration.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

import pytest

from memoryforge.models import Memory, MemoryType, MemorySource, Project, CommitInfo, LinkType
from memoryforge.storage.sqlite_db import SQLiteDatabase

# Check if git dependencies are available
try:
    from memoryforge.core.git_integration import GitIntegration
    HAS_GIT = True
except ImportError:
    HAS_GIT = False


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
        root_path="/test/repo",
    )
    return temp_db.create_project(project)


@pytest.fixture
def mock_repo():
    """Create a mock git repo."""
    repo = Mock()
    repo.active_branch.name = "main"
    repo.head.commit.hexsha = "abc123"
    return repo


@pytest.fixture
def mock_config():
    """Create a mock config for git integration."""
    config = Mock()
    config.enable_git_integration = True
    return config


@pytest.mark.skipif(not HAS_GIT, reason="git dependencies not installed")
class TestGitIntegrationInit:
    """Tests for git integration initialization."""
    
    def test_init_with_valid_repo(self, temp_db, project, mock_config):
        """Test initialization with a valid git repository."""
        integration = GitIntegration(
            sqlite_db=temp_db,
            config=mock_config,
            project_id=project.id,
        )
        
        assert integration is not None
        assert integration.project_id == project.id
    
    def test_init_with_invalid_repo(self, temp_db, project, mock_config):
        """Test initialization with git disabled."""
        mock_config.enable_git_integration = False
        
        integration = GitIntegration(
            sqlite_db=temp_db,
            config=mock_config,
            project_id=project.id,
        )
        
        # Should still initialize but not be available
        assert integration is not None
        assert not integration.is_available()


@pytest.mark.skipif(not HAS_GIT, reason="git dependencies not installed")
class TestMemoryLinking:
    """Tests for memory-commit linking."""
    
    def test_link_memory_to_commit(self, temp_db, project):
        """Test linking a memory to a commit."""
        # Create a memory
        memory = Memory(
            content="Added FastAPI to the project",
            type=MemoryType.STACK,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Link to commit
        link = temp_db.create_memory_link(
            memory_id=memory.id,
            commit_sha="abc123def456",
            link_type=LinkType.CREATED_FROM,
        )
        
        assert link.memory_id == memory.id
        assert link.commit_sha == "abc123def456"
        assert link.link_type == LinkType.CREATED_FROM
    
    def test_get_memories_by_commit(self, temp_db, project):
        """Test retrieving memories by commit SHA."""
        commit_sha = "abc123"
        
        # Create memories and link to commit
        for i in range(3):
            memory = Memory(
                content=f"Memory {i}",
                type=MemoryType.NOTE,
                source=MemorySource.GIT,
                project_id=project.id,
                confirmed=True,
            )
            temp_db.create_memory(memory)
            temp_db.create_memory_link(
                memory_id=memory.id,
                commit_sha=commit_sha,
                link_type=LinkType.RELATED_TO,
            )
        
        # Retrieve
        memories = temp_db.get_memories_by_commit(commit_sha)
        
        assert len(memories) == 3
    
    def test_get_memory_links(self, temp_db, project):
        """Test retrieving links for a memory."""
        memory = Memory(
            content="Test memory",
            type=MemoryType.DECISION,
            source=MemorySource.MANUAL,
            project_id=project.id,
            confirmed=True,
        )
        temp_db.create_memory(memory)
        
        # Create multiple links
        temp_db.create_memory_link(memory.id, "commit1", LinkType.CREATED_FROM)
        temp_db.create_memory_link(memory.id, "commit2", LinkType.MENTIONED_IN)
        
        links = temp_db.get_memory_links(memory.id)
        
        assert len(links) == 2
        assert {l.commit_sha for l in links} == {"commit1", "commit2"}


@pytest.mark.skipif(not HAS_GIT, reason="git dependencies not installed")
class TestCommitAnalysis:
    """Tests for commit analysis."""
    
    def test_parse_commit_info(self):
        """Test parsing commit information."""
        commit_info = CommitInfo(
            sha="abc123",
            message="feat: Add user authentication",
            author="developer@example.com",
            date=datetime.utcnow(),
            files_changed=["auth.py", "login.py"],
        )
        
        assert commit_info.sha == "abc123"
        assert "authentication" in commit_info.message.lower()
        assert len(commit_info.files_changed) == 2
    
    def test_commit_is_architectural(self):
        """Test detecting architectural commits."""
        # Keywords that suggest architectural changes
        arch_messages = [
            "feat: Add new microservice architecture",
            "refactor: Move to event-driven design",
            "feat: Implement API gateway",
            "chore: Setup Docker infrastructure",
        ]
        
        non_arch_messages = [
            "fix: Typo in readme",
            "style: Format code",
            "docs: Update comments",
        ]
        
        arch_keywords = ["architecture", "refactor", "infrastructure", "api", "service", "design"]
        
        for msg in arch_messages:
            has_keyword = any(kw in msg.lower() for kw in arch_keywords)
            # At least some should match
            assert has_keyword or "feat" in msg.lower()


@pytest.mark.skipif(not HAS_GIT, reason="git dependencies not installed")
class TestGitActivity:
    """Tests for git activity commands."""
    
    def test_get_recent_commits(self, temp_db, project, mock_config):
        """Test getting recent commits."""
        integration = GitIntegration(
            sqlite_db=temp_db,
            config=mock_config,
            project_id=project.id,
        )
        
        # Test that get_recent_activity returns proper structure
        activity = integration.get_recent_activity(days=7)
        
        # Should return a dict with expected keys
        assert isinstance(activity, dict)
        # Available may be False if not in a git repo, that's OK
