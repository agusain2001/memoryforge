"""
Project Router for MemoryForge v2.

Manages project switching, lifecycle, and context. Active project is stored
in config only (not in DB) as it's runtime state.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from memoryforge.config import Config
from memoryforge.models import Project
from memoryforge.storage.sqlite_db import SQLiteDatabase

logger = logging.getLogger(__name__)


class ProjectRouter:
    """Manages project switching and lifecycle."""
    
    def __init__(self, sqlite_db: SQLiteDatabase, config: Config):
        """
        Initialize the project router.
        
        Args:
            sqlite_db: SQLite database instance
            config: MemoryForge configuration
        """
        self.db = sqlite_db
        self.config = config
    
    def create_project(
        self,
        name: str,
        root_path: str,
        set_active: bool = True,
    ) -> Project:
        """
        Create a new project.
        
        Args:
            name: Project name (must be unique)
            root_path: Path to project root directory
            set_active: Whether to set this as the active project
            
        Returns:
            The created Project
            
        Raises:
            ValueError: If project with same name exists
        """
        # Check for existing project with same name
        existing = self.db.get_project_by_name(name)
        if existing:
            raise ValueError(f"Project '{name}' already exists")
        
        # Create the project
        project = Project(
            name=name,
            root_path=str(Path(root_path).resolve()),
        )
        
        self.db.create_project(project)
        logger.info(f"Created project '{name}' with ID {project.id}")
        
        # Set as active if requested
        if set_active:
            self.switch_project(project.id)
        
        return project
    
    def switch_project(self, project_id: UUID) -> bool:
        """
        Switch to a different active project.
        
        Active project is stored in config only (not in DB) as it's
        runtime state that shouldn't be shared.
        
        Args:
            project_id: ID of the project to switch to
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If project doesn't exist
        """
        project = self.db.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        # Update config (config-only, not DB)
        self.config.active_project_id = str(project_id)
        self.config.save()
        
        logger.info(f"Switched to project '{project.name}' (ID: {project_id})")
        return True
    
    def switch_project_by_name(self, name: str) -> bool:
        """
        Switch to a project by name.
        
        Args:
            name: Name of the project to switch to
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If project doesn't exist
        """
        project = self.db.get_project_by_name(name)
        if not project:
            raise ValueError(f"Project '{name}' not found")
        
        return self.switch_project(project.id)
    
    def get_active_project(self) -> Optional[Project]:
        """
        Get the currently active project from config.
        
        Returns:
            The active Project, or None if no project is active
        """
        if not self.config.active_project_id:
            return None
        
        try:
            project_id = UUID(self.config.active_project_id)
            return self.db.get_project(project_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid active_project_id in config: {self.config.active_project_id}")
            return None
    
    def get_active_project_id(self) -> Optional[UUID]:
        """
        Get the active project ID.
        
        Returns:
            The active project UUID, or None if no project is active
        """
        if not self.config.active_project_id:
            return None
        
        try:
            return UUID(self.config.active_project_id)
        except (ValueError, TypeError):
            return None
    
    def list_projects(self) -> list[Project]:
        """
        List all projects.
        
        Returns:
            List of all projects
        """
        return self.db.list_projects()
    
    def get_project(self, project_id: UUID) -> Optional[Project]:
        """
        Get a project by ID.
        
        Args:
            project_id: ID of the project
            
        Returns:
            The Project, or None if not found
        """
        return self.db.get_project(project_id)
    
    def get_project_by_name(self, name: str) -> Optional[Project]:
        """
        Get a project by name.
        
        Args:
            name: Name of the project
            
        Returns:
            The Project, or None if not found
        """
        return self.db.get_project_by_name(name)
    
    def delete_project(self, project_id: UUID) -> bool:
        """
        Delete a project (blocked if has memories).
        
        Projects with existing memories cannot be deleted to prevent
        data loss. Delete all memories first.
        
        Args:
            project_id: ID of the project to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            ValueError: If project has memories or doesn't exist
        """
        project = self.db.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        if not self.db.can_delete_project(project_id):
            memory_count = self.db.get_memory_count(project_id, confirmed_only=False)
            raise ValueError(
                f"Cannot delete project '{project.name}' - it has {memory_count} memories. "
                "Delete all memories first."
            )
        
        # If deleting the active project, clear it
        if self.config.active_project_id == str(project_id):
            self.config.active_project_id = None
            self.config.save()
        
        success = self.db.delete_project(project_id)
        if success:
            logger.info(f"Deleted project '{project.name}' (ID: {project_id})")
        
        return success
    
    def get_project_status(self, project_id: Optional[UUID] = None) -> dict:
        """
        Get status information for a project.
        
        Args:
            project_id: ID of project (defaults to active project)
            
        Returns:
            Dict with project status info
        """
        if project_id is None:
            project_id = self.get_active_project_id()
        
        if not project_id:
            return {
                "active": False,
                "message": "No active project",
            }
        
        project = self.db.get_project(project_id)
        if not project:
            return {
                "active": False,
                "message": f"Project {project_id} not found",
            }
        
        memory_count = self.db.get_memory_count(project_id, confirmed_only=True)
        total_count = self.db.get_memory_count(project_id, confirmed_only=False)
        pending_count = total_count - memory_count
        
        return {
            "active": True,
            "project_id": str(project.id),
            "project_name": project.name,
            "root_path": project.root_path,
            "created_at": project.created_at.isoformat(),
            "memory_count": memory_count,
            "pending_count": pending_count,
            "is_active_project": str(project_id) == self.config.active_project_id,
        }
    
    def ensure_active_project(self) -> UUID:
        """
        Ensure there is an active project, creating default if needed.
        
        Returns:
            The active project ID
            
        Raises:
            RuntimeError: If no project can be determined
        """
        project_id = self.get_active_project_id()
        if project_id:
            return project_id
        
        # Check if any projects exist
        projects = self.list_projects()
        if projects:
            # Use the first project
            self.switch_project(projects[0].id)
            return projects[0].id
        
        # No projects exist - this shouldn't happen after init
        raise RuntimeError(
            "No projects found. Run 'memoryforge init' to create a project."
        )
