"""
Data models for MemoryForge.

These Pydantic models define the core data structures used throughout
the application, ensuring type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories that can be stored."""
    
    STACK = "stack"           # Tech stack (languages, frameworks, libraries)
    DECISION = "decision"     # Architecture decisions with reasons
    CONSTRAINT = "constraint" # Performance, infra, deadline constraints
    CONVENTION = "convention" # Code conventions (naming, structure, testing)
    NOTE = "note"             # General notes and observations


class MemorySource(str, Enum):
    """Source of how the memory was captured."""
    
    CHAT = "chat"                 # From AI conversation
    MANUAL = "manual"             # Explicitly added by user
    FILE_REFERENCE = "file_reference"  # Referenced from a file
    GIT = "git"                   # v2: Derived from git commit


class LinkType(str, Enum):
    """Types of memory-to-commit links."""
    
    CREATED_FROM = "created_from"   # Memory created from this commit
    MENTIONED_IN = "mentioned_in"   # Memory mentioned in commit message
    RELATED_TO = "related_to"       # Memory related to commit changes


class Memory(BaseModel):
    """Core memory entry stored in the system."""
    
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1, max_length=10240)
    type: MemoryType
    source: MemorySource
    project_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    confirmed: bool = False
    metadata: dict = Field(default_factory=dict)
    
    # v2: Staleness tracking
    is_stale: bool = False
    stale_reason: Optional[str] = None
    last_accessed: Optional[datetime] = None  # Updated on RETRIEVAL only
    
    # v2: Consolidation tracking
    is_archived: bool = False  # True if consolidated into another memory
    consolidated_into: Optional[UUID] = None  # Target memory ID
    
    class Config:
        from_attributes = True


class MemoryCreate(BaseModel):
    """Schema for creating a new memory."""
    
    content: str = Field(..., min_length=1, max_length=10240)
    type: MemoryType
    source: MemorySource = MemorySource.MANUAL
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result with relevance scoring and explanation."""
    
    memory: Memory
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str  # Why this result was selected
    
    class Config:
        from_attributes = True


class Project(BaseModel):
    """Project configuration and metadata."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    root_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class EmbeddingRecord(BaseModel):
    """Links a memory to its vector embedding in Qdrant."""
    
    memory_id: UUID
    vector_id: str  # Qdrant point ID
    
    class Config:
        from_attributes = True


# ============================================================================
# v2 Models
# ============================================================================

class MemoryVersion(BaseModel):
    """Tracks memory content history (for consolidation rollback)."""
    
    id: UUID = Field(default_factory=uuid4)
    memory_id: UUID
    content: str
    version: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class MemoryLink(BaseModel):
    """Links memories to git commits."""
    
    id: UUID = Field(default_factory=uuid4)
    memory_id: UUID
    commit_sha: str
    link_type: LinkType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class ConsolidationSuggestion(BaseModel):
    """Suggested consolidation for review."""
    
    source_memories: list[Memory]
    similarity_score: float
    suggested_content: str


class CommitInfo(BaseModel):
    """Git commit information."""
    
    sha: str
    message: str
    author: str
    date: datetime
    files_changed: list[str] = Field(default_factory=list)
