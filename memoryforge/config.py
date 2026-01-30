"""
Configuration management for MemoryForge.

Handles loading, validating, and persisting configuration from YAML files.
Default location: ~/.memoryforge/config.yaml
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def get_default_storage_path() -> Path:
    """Get the default storage path for MemoryForge."""
    return Path.home() / ".memoryforge"


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""
    OPENAI = "openai"
    LOCAL = "local"  # sentence-transformers


class Config(BaseSettings):
    """MemoryForge configuration settings."""
    
    # Project settings
    project_name: str = Field(default="default")
    project_root: str = Field(default=".")
    
    # Storage settings
    storage_path: Path = Field(default_factory=get_default_storage_path)
    
    # Embedding settings
    embedding_provider: EmbeddingProvider = Field(default=EmbeddingProvider.LOCAL)
    
    # OpenAI settings (used when embedding_provider = "openai")
    openai_api_key: str = Field(default="")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    
    # Local embedding settings (used when embedding_provider = "local")
    local_embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Retrieval settings
    max_results: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Server settings
    mcp_host: str = Field(default="localhost")
    mcp_port: int = Field(default=3000)
    
    # v2: Multi-project (active project stored here, NOT in DB)
    active_project_id: Optional[str] = None
    
    # v2: Git integration
    enable_git_integration: bool = False
    
    # v2: Consolidation settings
    consolidation_threshold: float = Field(default=0.90, ge=0.7, le=0.99)
    
    # v2.1: Sync settings
    sync_key: Optional[str] = None
    sync_path: Optional[Path] = None
    sync_backend: str = "local"
    
    class Config:
        env_prefix = "MEMORYFORGE_"
        env_file = ".env"
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = get_default_storage_path() / "config.yaml"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        
        return cls()
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = self.storage_path / "config.yaml"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "project_name": self.project_name,
            "project_root": self.project_root,
            "storage_path": str(self.storage_path),
            "embedding_provider": self.embedding_provider.value,
            "openai_api_key": self.openai_api_key,
            "openai_embedding_model": self.openai_embedding_model,
            "local_embedding_model": self.local_embedding_model,
            "max_results": self.max_results,
            "min_score": self.min_score,
            "mcp_host": self.mcp_host,
            "mcp_port": self.mcp_port,
            # v2 fields
            "active_project_id": self.active_project_id,
            "enable_git_integration": self.enable_git_integration,
            "consolidation_threshold": self.consolidation_threshold,
            # v2.1 fields
            "sync_key": self.sync_key,
            "sync_path": str(self.sync_path) if self.sync_path else None,
            "sync_backend": self.sync_backend,
        }
        
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @property
    def sqlite_path(self) -> Path:
        """Get the SQLite database path."""
        return self.storage_path / "sqlite" / "memoryforge.db"
    
    @property
    def qdrant_path(self) -> Path:
        """Get the Qdrant storage path."""
        return self.storage_path / "qdrant"
    
    @property
    def logs_path(self) -> Path:
        """Get the logs directory path."""
        return self.storage_path / "logs"
    
    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
