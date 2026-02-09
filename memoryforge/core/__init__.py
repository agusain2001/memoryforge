"""Core engine components for MemoryForge."""

from memoryforge.core.memory_manager import MemoryManager
from memoryforge.core.retrieval import RetrievalEngine
from memoryforge.core.embedding_service import EmbeddingService
from memoryforge.core.local_embedding_service import LocalEmbeddingService
from memoryforge.core.embedding_factory import create_embedding_service, get_embedding_dimension
from memoryforge.core.validation import ValidationLayer
# v3 components
from memoryforge.core.graph_builder import GraphBuilder
from memoryforge.core.confidence_scorer import ConfidenceScorer
from memoryforge.core.conflict_resolver import ConflictResolver, SyncConflict

__all__ = [
    "MemoryManager",
    "RetrievalEngine", 
    "EmbeddingService",
    "LocalEmbeddingService",
    "create_embedding_service",
    "get_embedding_dimension",
    "ValidationLayer",
    # v3
    "GraphBuilder",
    "ConfidenceScorer",
    "ConflictResolver",
    "SyncConflict",
]

