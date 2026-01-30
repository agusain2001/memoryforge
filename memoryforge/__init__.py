"""
MemoryForge - Local-first memory layer for AI coding assistants.

A simple, practical memory system that remembers your codebase context,
architecture decisions, and preferences across AI conversations.
"""

from memoryforge.models import Memory, MemoryType, MemorySource, SearchResult, Project
from memoryforge.core.memory_manager import MemoryManager
from memoryforge.config import Config

__version__ = "0.9.0"
__all__ = [
    "Memory",
    "MemoryType", 
    "MemorySource",
    "SearchResult",
    "Project",
    "MemoryManager",
    "Config",
]
