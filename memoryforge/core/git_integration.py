"""
Git Integration Service for MemoryForge v2.

Links memories to git commits and provides sync functionality.
All git operations are read-only - human memories take precedence.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

from memoryforge.config import Config
from memoryforge.models import Memory, MemoryType, MemorySource, LinkType
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.core.git_scanner import GitScanner, CommitInfo, GitNotAvailableError

logger = logging.getLogger(__name__)


class GitIntegration:
    """
    Manages git ↔ memory integration.
    
    Git-derived links are best-effort and not authoritative.
    Human-written memories always take precedence.
    """
    
    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        config: Config,
        project_id: UUID,
    ):
        """
        Initialize git integration.
        
        Args:
            sqlite_db: SQLite database instance
            config: MemoryForge configuration
            project_id: Current project ID
        """
        self.db = sqlite_db
        self.config = config
        self.project_id = project_id
        self._scanner: Optional[GitScanner] = None
    
    def _get_scanner(self) -> Optional[GitScanner]:
        """Get or create the git scanner."""
        if self._scanner is not None:
            return self._scanner
        
        if not self.config.enable_git_integration:
            return None
        
        try:
            # Use project root as repo path
            project = self.db.get_project(self.project_id)
            if not project:
                return None
            
            repo_path = Path(project.root_path)
            self._scanner = GitScanner(repo_path)
            return self._scanner
        except GitNotAvailableError as e:
            logger.warning(f"Git not available: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if git integration is available and enabled."""
        return self._get_scanner() is not None
    
    def get_status(self) -> dict:
        """
        Get git integration status.
        
        Returns:
            Dict with status information
        """
        if not self.config.enable_git_integration:
            return {
                "enabled": False,
                "message": "Git integration is disabled in config",
            }
        
        scanner = self._get_scanner()
        if not scanner:
            return {
                "enabled": True,
                "available": False,
                "message": "Git repository not found or git not installed",
            }
        
        try:
            repo_info = scanner.get_repo_info()
            return {
                "enabled": True,
                "available": True,
                "branch": repo_info["branch"],
                "commit": repo_info["commit"],
                "is_clean": repo_info["is_clean"],
                "remote": repo_info["remote"],
            }
        except Exception as e:
            return {
                "enabled": True,
                "available": False,
                "message": f"Error reading git: {e}",
            }
    
    def link_memory_to_commit(
        self,
        memory_id: UUID,
        commit_sha: str,
        link_type: LinkType = LinkType.MENTIONED_IN,
    ) -> bool:
        """
        Link a memory to a git commit.
        
        Args:
            memory_id: Memory ID to link
            commit_sha: Commit SHA (full or short)
            link_type: Type of link
            
        Returns:
            True if linked successfully
        """
        scanner = self._get_scanner()
        if not scanner:
            return False
        
        # Validate commit exists
        commit = scanner.get_commit(commit_sha)
        if not commit:
            logger.warning(f"Commit not found: {commit_sha}")
            return False
        
        # Create the link
        try:
            self.db.create_memory_link(
                memory_id=memory_id,
                commit_sha=commit.sha,  # Use full SHA
                link_type=link_type,
            )
            logger.info(f"Linked memory {memory_id} to commit {commit.short_sha}")
            return True
        except Exception as e:
            logger.error(f"Failed to link memory: {e}")
            return False
    
    def get_memories_for_commit(self, commit_sha: str) -> List[Memory]:
        """
        Get all memories linked to a commit.
        
        Args:
            commit_sha: Commit SHA (full or short)
            
        Returns:
            List of linked memories
        """
        return self.db.get_memories_by_commit(commit_sha)
    
    def get_commit_info(self, commit_sha: str) -> Optional[CommitInfo]:
        """
        Get information about a commit.
        
        Args:
            commit_sha: Commit SHA
            
        Returns:
            CommitInfo or None
        """
        scanner = self._get_scanner()
        if not scanner:
            return None
        
        return scanner.get_commit(commit_sha)
    
    def find_relevant_commits(
        self,
        memory: Memory,
        limit: int = 5,
    ) -> List[Tuple[CommitInfo, float]]:
        """
        Find commits potentially related to a memory.
        
        Uses keyword matching as a heuristic. This is best-effort
        and may not find all relevant commits.
        
        Args:
            memory: Memory to find commits for
            limit: Maximum commits to return
            
        Returns:
            List of (CommitInfo, relevance_score) tuples
        """
        scanner = self._get_scanner()
        if not scanner:
            return []
        
        # Extract keywords from memory content
        words = set(memory.content.lower().split())
        
        # Filter to significant words (>3 chars, not common)
        common_words = {"the", "and", "for", "are", "but", "not", "you", "all", 
                       "can", "had", "her", "was", "one", "our", "out", "has",
                       "have", "been", "were", "will", "with", "this", "that",
                       "from", "they", "what", "when", "make", "like", "time",
                       "just", "know", "take", "into", "year", "your", "good",
                       "some", "could", "them", "than", "look", "only", "come",
                       "over", "such", "also", "back", "after", "work", "first",
                       "well", "most", "must"}
        keywords = [w for w in words if len(w) > 3 and w not in common_words]
        
        if not keywords:
            return []
        
        # Get recent commits
        commits = scanner.get_recent_commits(100)
        
        # Score each commit by keyword matches
        scored = []
        for commit in commits:
            message_lower = commit.message.lower()
            matches = sum(1 for kw in keywords if kw in message_lower)
            if matches > 0:
                score = matches / len(keywords)
                scored.append((commit, score))
        
        # Sort by score and return top results
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
    
    def sync_architectural_commits(self) -> List[dict]:
        """
        Scan for architectural commits and suggest memory creation.
        
        This is a discovery tool - it suggests commits that might
        deserve a memory but doesn't create them automatically.
        
        Returns:
            List of commit suggestions with reason
        """
        scanner = self._get_scanner()
        if not scanner:
            return []
        
        # Find architectural commits
        arch_commits = scanner.find_architectural_commits(limit=50)
        
        suggestions = []
        for commit in arch_commits:
            # Check if already linked to a memory
            linked_memories = self.db.get_memories_by_commit(commit.sha)
            
            # Determine why this commit was flagged
            reason = self._get_architectural_reason(commit)
            
            suggestions.append({
                "commit": {
                    "sha": commit.sha,
                    "short_sha": commit.short_sha,
                    "message": commit.first_line,
                    "author": commit.author,
                    "date": commit.date.isoformat(),
                    "files_changed": len(commit.files_changed),
                },
                "reason": reason,
                "has_memory": len(linked_memories) > 0,
                "memory_count": len(linked_memories),
            })
        
        return suggestions
    
    def _get_architectural_reason(self, commit: CommitInfo) -> str:
        """Determine why a commit was flagged as architectural."""
        message_lower = commit.message.lower()
        
        if "refactor" in message_lower:
            return "Refactoring change"
        elif "migrate" in message_lower:
            return "Migration"
        elif "architecture" in message_lower:
            return "Architecture change"
        elif "design" in message_lower:
            return "Design change"
        elif "breaking" in message_lower:
            return "Breaking change"
        elif "rewrite" in message_lower:
            return "Rewrite"
        elif "restructure" in message_lower:
            return "Restructuring"
        elif "upgrade" in message_lower:
            return "Upgrade"
        elif "deprecate" in message_lower:
            return "Deprecation"
        elif "remove" in message_lower:
            return "Removal"
        else:
            return "Architectural keyword detected"
    
    def get_recent_activity(self, days: int = 7) -> dict:
        """
        Get recent git activity summary.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with activity summary
        """
        scanner = self._get_scanner()
        if not scanner:
            return {"available": False}
        
        from datetime import datetime, timedelta
        
        commits = scanner.get_recent_commits(100)
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_commits = [c for c in commits if c.date.replace(tzinfo=None) > cutoff]
        
        # Count files changed
        all_files = set()
        for commit in recent_commits:
            all_files.update(commit.files_changed)
        
        return {
            "available": True,
            "days": days,
            "commit_count": len(recent_commits),
            "files_changed": len(all_files),
            "authors": list(set(c.author for c in recent_commits)),
        }
