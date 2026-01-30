"""
Sync Adapter Protocol for MemoryForge.

Defines the interface for sync backends (Local File, S3, etc.).
"""

from typing import Protocol, List, Optional
from datetime import datetime


class SyncAdapterProtocol(Protocol):
    """Interface for sync backends."""
    
    def initialize(self) -> None:
        """Initialize the backend (create directories, buckets, etc.)."""
        ...
        
    def list_files(self) -> List[str]:
        """List all available memory files in the backend."""
        ...
        
    def read_file(self, filename: str) -> Optional[str]:
        """
        Read content of a file.
        
        Args:
            filename: Name of the file to read
            
        Returns:
            File content as string, or None if not found
        """
        ...
        
    def write_file(self, filename: str, content: str) -> None:
        """
        Write content to a file.
        
        Args:
            filename: Name of the file
            content: Content to write
        """
        ...
        
    def delete_file(self, filename: str) -> None:
        """Delete a file from the backend."""
        ...
        
    def get_last_modified(self, filename: str) -> Optional[datetime]:
        """Get last modified timestamp of a file."""
        ...
