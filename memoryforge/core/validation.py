"""
Validation layer for MemoryForge.

Validates memory entries before storage to prevent junk data.
"""

import logging
from typing import Optional

from memoryforge.models import Memory, MemoryCreate, MemoryType, MemorySource

logger = logging.getLogger(__name__)

# Validation constraints
MAX_CONTENT_LENGTH = 10240  # 10KB
MIN_CONTENT_LENGTH = 1
ALLOWED_MEMORY_TYPES = set(MemoryType)
ALLOWED_MEMORY_SOURCES = set(MemorySource)


class ValidationError(Exception):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)


class ValidationLayer:
    """Validates memory entries before storage."""
    
    @staticmethod
    def validate_memory_create(data: MemoryCreate) -> None:
        """
        Validate a memory creation request.
        
        Raises:
            ValidationError: If validation fails
        """
        # Validate content length
        if len(data.content) < MIN_CONTENT_LENGTH:
            raise ValidationError(
                f"Content must be at least {MIN_CONTENT_LENGTH} character(s)",
                field="content",
            )
        
        if len(data.content) > MAX_CONTENT_LENGTH:
            raise ValidationError(
                f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters",
                field="content",
            )
        
        # Validate memory type
        if data.type not in ALLOWED_MEMORY_TYPES:
            raise ValidationError(
                f"Invalid memory type: {data.type}. Allowed: {[t.value for t in ALLOWED_MEMORY_TYPES]}",
                field="type",
            )
        
        # Validate memory source
        if data.source not in ALLOWED_MEMORY_SOURCES:
            raise ValidationError(
                f"Invalid memory source: {data.source}. Allowed: {[s.value for s in ALLOWED_MEMORY_SOURCES]}",
                field="source",
            )
        
        # Validate content is not just whitespace
        if not data.content.strip():
            raise ValidationError(
                "Content cannot be empty or only whitespace",
                field="content",
            )
        
        logger.debug(f"Validation passed for memory of type {data.type.value}")
    
    @staticmethod
    def validate_memory(memory: Memory) -> None:
        """
        Validate a full memory object.
        
        Raises:
            ValidationError: If validation fails
        """
        # Validate content
        if len(memory.content) < MIN_CONTENT_LENGTH:
            raise ValidationError(
                f"Content must be at least {MIN_CONTENT_LENGTH} character(s)",
                field="content",
            )
        
        if len(memory.content) > MAX_CONTENT_LENGTH:
            raise ValidationError(
                f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters",
                field="content",
            )
        
        # Validate type
        if memory.type not in ALLOWED_MEMORY_TYPES:
            raise ValidationError(
                f"Invalid memory type: {memory.type}",
                field="type",
            )
        
        # Validate source
        if memory.source not in ALLOWED_MEMORY_SOURCES:
            raise ValidationError(
                f"Invalid memory source: {memory.source}",
                field="source",
            )
        
        # Validate project_id exists
        if not memory.project_id:
            raise ValidationError(
                "Memory must have a project_id",
                field="project_id",
            )
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """
        Sanitize memory content.
        
        - Strips leading/trailing whitespace
        - Normalizes line endings
        - Removes null characters
        """
        # Remove null characters
        content = content.replace("\x00", "")
        
        # Normalize line endings
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        return content
    
    @staticmethod
    def validate_search_query(query: str) -> None:
        """
        Validate a search query.
        
        Raises:
            ValidationError: If validation fails
        """
        if not query or not query.strip():
            raise ValidationError(
                "Search query cannot be empty",
                field="query",
            )
        
        if len(query) > MAX_CONTENT_LENGTH:
            raise ValidationError(
                f"Search query exceeds maximum length of {MAX_CONTENT_LENGTH} characters",
                field="query",
            )
