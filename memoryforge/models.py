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
    
    # v3: Confidence scoring
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
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


# ============================================================================
# v3 Models
# ============================================================================

class RelationType(str, Enum):
    """Types of memory-to-memory relationships."""
    
    CAUSED_BY = "caused_by"       # This memory was caused by another
    SUPERSEDES = "supersedes"     # This memory replaces another
    RELATES_TO = "relates_to"     # General relationship
    BLOCKS = "blocks"             # This memory blocks another
    DEPENDS_ON = "depends_on"     # This memory depends on another


class MemoryRelation(BaseModel):
    """Links between memories (v3: Graph Memory)."""
    
    id: UUID = Field(default_factory=uuid4)
    source_memory_id: UUID
    target_memory_id: UUID
    relation_type: RelationType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None  # 'human' or 'git-derived'
    
    class Config:
        from_attributes = True


class ConflictResolution(str, Enum):
    """How a sync conflict was resolved."""
    
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MANUAL = "manual"
    MERGED = "merged"


class ConflictLog(BaseModel):
    """Logs sync conflicts and their resolutions (v3)."""
    
    id: UUID = Field(default_factory=uuid4)
    memory_id: UUID
    local_content: Optional[str] = None
    remote_content: Optional[str] = None
    resolution: ConflictResolution
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_by: Optional[str] = None
    
    class Config:
        from_attributes = True


class CrossProjectSuggestion(BaseModel):
    """Suggestion from cross-project reasoning (v3)."""
    
    source_project_id: UUID
    source_project_name: str
    source_memory: Memory
    similarity_score: float
    suggestion: str  # Why this might be relevant
