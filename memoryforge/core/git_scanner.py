"""
Git Scanner for MemoryForge v2.

Read-only git repository scanner for linking memories to commits.
Git-derived links are best-effort and not authoritative - human-written
memories always take precedence.

Requires: gitpython (optional dependency)
"""

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default keywords for finding architectural commits
DEFAULT_ARCHITECTURAL_KEYWORDS = [
    "refactor",
    "migrate",
    "architecture",
    "design",
    "breaking",
    "rewrite",
    "restructure",
    "upgrade",
    "deprecate",
    "remove",
]


@dataclass
class CommitInfo:
    """Information about a git commit."""
    
    sha: str
    message: str
    author: str
    date: datetime
    files_changed: List[str]
    
    @property
    def short_sha(self) -> str:
        """Get short SHA (first 7 chars)."""
        return self.sha[:7]
    
    @property
    def first_line(self) -> str:
        """Get first line of commit message."""
        return self.message.split("\n")[0]


class GitNotAvailableError(Exception):
    """Raised when git is not available or not a git repository."""
    pass


class GitScanner:
    """
    Read-only git repository scanner.
    
    Uses subprocess calls to git for maximum compatibility.
    Does NOT modify the repository in any way.
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize the git scanner.
        
        Args:
            repo_path: Path to the git repository root
            
        Raises:
            GitNotAvailableError: If git is not installed or path is not a repo
        """
        self.repo_path = Path(repo_path).resolve()
        self._validate_git_available()
        self._validate_git_repo()
    
    def _validate_git_available(self) -> None:
        """Check if git command is available."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise GitNotAvailableError("git command failed")
        except FileNotFoundError:
            raise GitNotAvailableError("git is not installed")
        except subprocess.TimeoutExpired:
            raise GitNotAvailableError("git command timed out")
    
    def _validate_git_repo(self) -> None:
        """Check if the path is a valid git repository."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            raise GitNotAvailableError(f"Not a git repository: {self.repo_path}")
    
    def _run_git(self, *args: str) -> str:
        """
        Run a git command and return output.
        
        Args:
            *args: Git command arguments
            
        Returns:
            Command output (stdout)
            
        Raises:
            GitNotAvailableError: If command fails
        """
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(f"git command failed: git {' '.join(args)}")
                logger.debug(f"stderr: {result.stderr}")
                return ""
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"git command timed out: git {' '.join(args)}")
            return ""
        except Exception as e:
            logger.error(f"git command error: {e}")
            return ""
    
    def get_current_branch(self) -> Optional[str]:
        """Get the current branch name."""
        output = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return output.strip() if output else None
    
    def get_current_commit(self) -> Optional[str]:
        """Get the current HEAD commit SHA."""
        output = self._run_git("rev-parse", "HEAD")
        return output.strip() if output else None
    
    def get_recent_commits(self, limit: int = 50) -> List[CommitInfo]:
        """
        Get recent commits.
        
        Args:
            limit: Maximum number of commits to return
            
        Returns:
            List of CommitInfo objects, newest first
        """
        # Format: SHA|author|date|subject
        format_str = "%H|%an|%aI|%s"
        output = self._run_git(
            "log",
            f"--format={format_str}",
            f"-n{limit}",
            "--no-merges",
        )
        
        if not output:
            return []
        
        commits = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            
            sha, author, date_str, message = parts
            
            try:
                # Parse ISO format date
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                date = datetime.utcnow()
            
            # Get files changed for this commit
            files_output = self._run_git(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                sha,
            )
            files_changed = [f for f in files_output.strip().split("\n") if f]
            
            commits.append(CommitInfo(
                sha=sha,
                message=message,
                author=author,
                date=date,
                files_changed=files_changed,
            ))
        
        return commits
    
    def get_commit(self, sha: str) -> Optional[CommitInfo]:
        """
        Get details for a specific commit.
        
        Args:
            sha: Full or short commit SHA
            
        Returns:
            CommitInfo or None if not found
        """
        # Format: SHA|author|date|subject
        format_str = "%H|%an|%aI|%B"
        output = self._run_git(
            "log",
            f"--format={format_str}",
            "-n1",
            sha,
        )
        
        if not output:
            return None
        
        parts = output.strip().split("|", 3)
        if len(parts) < 4:
            return None
        
        full_sha, author, date_str, message = parts
        
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            date = datetime.utcnow()
        
        # Get files changed
        files_output = self._run_git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            full_sha,
        )
        files_changed = [f for f in files_output.strip().split("\n") if f]
        
        return CommitInfo(
            sha=full_sha,
            message=message.strip(),
            author=author,
            date=date,
            files_changed=files_changed,
        )
    
    def find_architectural_commits(
        self,
        keywords: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[CommitInfo]:
        """
        Find commits mentioning architecture-related keywords.
        
        This is a best-effort heuristic and may not catch all
        architectural changes.
        
        Args:
            keywords: Keywords to search for (default: common architecture terms)
            limit: Maximum commits to scan
            
        Returns:
            Commits matching any of the keywords
        """
        if keywords is None:
            keywords = DEFAULT_ARCHITECTURAL_KEYWORDS
        
        all_commits = self.get_recent_commits(limit)
        
        matching = []
        for commit in all_commits:
            message_lower = commit.message.lower()
            if any(kw.lower() in message_lower for kw in keywords):
                matching.append(commit)
        
        return matching
    
    def find_commits_affecting_file(
        self,
        file_path: str,
        limit: int = 20,
    ) -> List[CommitInfo]:
        """
        Find commits that modified a specific file.
        
        Args:
            file_path: Path to the file (relative to repo root)
            limit: Maximum commits to return
            
        Returns:
            Commits that modified the file
        """
        format_str = "%H|%an|%aI|%s"
        output = self._run_git(
            "log",
            f"--format={format_str}",
            f"-n{limit}",
            "--follow",
            "--",
            file_path,
        )
        
        if not output:
            return []
        
        commits = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            
            sha, author, date_str, message = parts
            
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                date = datetime.utcnow()
            
            commits.append(CommitInfo(
                sha=sha,
                message=message,
                author=author,
                date=date,
                files_changed=[file_path],  # We know it affected this file
            ))
        
        return commits
    
    def get_file_at_commit(self, file_path: str, sha: str) -> Optional[str]:
        """
        Get file contents at a specific commit.
        
        Args:
            file_path: Path to the file (relative to repo root)
            sha: Commit SHA
            
        Returns:
            File contents or None if file didn't exist
        """
        output = self._run_git("show", f"{sha}:{file_path}")
        return output if output else None
    
    def get_diff_stats(self, since_sha: str) -> dict:
        """
        Get diff statistics since a commit.
        
        Args:
            since_sha: Commit SHA to diff from
            
        Returns:
            Dict with files_changed, insertions, deletions
        """
        output = self._run_git(
            "diff",
            "--stat",
            "--numstat",
            f"{since_sha}..HEAD",
        )
        
        if not output:
            return {"files_changed": 0, "insertions": 0, "deletions": 0}
        
        insertions = 0
        deletions = 0
        files = set()
        
        for line in output.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    ins = int(parts[0]) if parts[0] != "-" else 0
                    dels = int(parts[1]) if parts[1] != "-" else 0
                    insertions += ins
                    deletions += dels
                    files.add(parts[2])
                except ValueError:
                    pass
        
        return {
            "files_changed": len(files),
            "insertions": insertions,
            "deletions": deletions,
        }
    
    def is_clean(self) -> bool:
        """Check if the working tree is clean (no uncommitted changes)."""
        output = self._run_git("status", "--porcelain")
        return len(output.strip()) == 0
    
    def get_repo_info(self) -> dict:
        """
        Get general repository information.
        
        Returns:
            Dict with repo metadata
        """
        branch = self.get_current_branch()
        commit = self.get_current_commit()
        is_clean = self.is_clean()
        
        # Get remote URL (origin)
        remote = self._run_git("remote", "get-url", "origin").strip()
        
        return {
            "path": str(self.repo_path),
            "branch": branch,
            "commit": commit[:7] if commit else None,
            "is_clean": is_clean,
            "remote": remote if remote else None,
        }
