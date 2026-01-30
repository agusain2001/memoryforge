"""
Local File Sync Adapter.

Syncs memories to a local directory (which can be a git repo or shared drive).
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from memoryforge.sync.adapter import SyncAdapterProtocol


class LocalFileAdapter:
    """Implementation of SyncAdapter for local filesystem."""
    
    def __init__(self, sync_path: Path):
        """
        Initialize local file adapter.
        
        Args:
            sync_path: Path to the sync directory
        """
        self.sync_path = Path(sync_path).resolve()
        
    def initialize(self) -> None:
        """Create the sync directory if it doesn't exist."""
        self.sync_path.mkdir(parents=True, exist_ok=True)
        
    def list_files(self) -> List[str]:
        """List all .json files in the sync directory."""
        if not self.sync_path.exists():
            return []
        
        return [
            f.name for f in self.sync_path.glob("*.json")
            if f.is_file()
        ]
        
    def read_file(self, filename: str) -> Optional[str]:
        """Read content of a file."""
        file_path = self.sync_path / filename
        if not file_path.exists():
            return None
            
        return file_path.read_text(encoding="utf-8")
        
    def write_file(self, filename: str, content: str) -> None:
        """Write content to a file."""
        self.initialize()  # Ensure dir exists
        file_path = self.sync_path / filename
        file_path.write_text(content, encoding="utf-8")
        
    def delete_file(self, filename: str) -> None:
        """Delete a file."""
        file_path = self.sync_path / filename
        if file_path.exists():
            file_path.unlink()
            
    def get_last_modified(self, filename: str) -> Optional[datetime]:
        """Get last modified timestamp."""
        file_path = self.sync_path / filename
        if not file_path.exists():
            return None
            
        timestamp = file_path.stat().st_mtime
        return datetime.fromtimestamp(timestamp)
