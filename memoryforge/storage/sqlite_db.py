"""
SQLite database manager for MemoryForge.

This is the source of truth for all memory data. 
Qdrant vectors are derived from this data.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
from uuid import UUID

from memoryforge.models import Memory, MemoryType, MemorySource, Project, MemoryVersion, MemoryLink, LinkType

# Current schema version
SCHEMA_VERSION = 2


class SQLiteDatabase:
    """SQLite database manager for memories and projects."""
    
    def __init__(self, db_path: Path):
        """Initialize the database connection."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    confirmed INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            
            # Embeddings table (links memory to Qdrant vector)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    memory_id TEXT PRIMARY KEY,
                    vector_id TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_project_id 
                ON memories(project_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_confirmed 
                ON memories(confirmed)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type 
                ON memories(type)
            """)
            
            # ========== v2 Schema Additions ==========
            
            # Schema version tracking (for reversible migrations)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT
                )
            """)
            
            # Memory versions (for consolidation history and rollback)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_versions (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # Memory links to git commits
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # v2 indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_versions_memory 
                ON memory_versions(memory_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_links_commit 
                ON memory_links(commit_sha)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_links_memory 
                ON memory_links(memory_id)
            """)
            
            # Add v2 columns to memories table if not exist
            self._add_v2_columns(conn)
    
    # ========== Project Operations ==========
    
    def create_project(self, project: Project) -> Project:
        """Create a new project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO projects (id, name, root_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(project.id),
                    project.name,
                    project.root_path,
                    project.created_at.isoformat(),
                ),
            )
        return project
    
    def get_project(self, project_id: UUID) -> Optional[Project]:
        """Get a project by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM projects WHERE id = ?",
                (str(project_id),),
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Project(
                id=UUID(row["id"]),
                name=row["name"],
                root_path=row["root_path"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
    
    def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get a project by name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM projects WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Project(
                id=UUID(row["id"]),
                name=row["name"],
                root_path=row["root_path"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
    
    def list_projects(self) -> list[Project]:
        """List all projects."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
            rows = cursor.fetchall()
            
            return [
                Project(
                    id=UUID(row["id"]),
                    name=row["name"],
                    root_path=row["root_path"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]
    
    # ========== Memory Operations ==========
    
    def create_memory(self, memory: Memory) -> Memory:
        """Create a new memory (unconfirmed by default)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories 
                (id, project_id, content, type, source, created_at, updated_at, confirmed, metadata,
                 is_stale, stale_reason, last_accessed, is_archived, consolidated_into)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(memory.id),
                    str(memory.project_id),
                    memory.content,
                    memory.type.value,
                    memory.source.value,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat() if memory.updated_at else None,
                    1 if memory.confirmed else 0,
                    str(memory.metadata),
                    1 if memory.is_stale else 0,
                    memory.stale_reason,
                    memory.last_accessed.isoformat() if memory.last_accessed else None,
                    1 if memory.is_archived else 0,
                    str(memory.consolidated_into) if memory.consolidated_into else None,
                ),
            )
        return memory
    
    def save_memory(self, memory: Memory) -> Memory:
        """Save a memory (alias for create_memory, used by sync)."""
        return self.create_memory(memory)
    
    def get_memory(self, memory_id: UUID) -> Optional[Memory]:
        """Get a memory by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE id = ?",
                (str(memory_id),),
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_memory(row)
    
    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        # Get row keys for safe access to v2 columns
        row_keys = row.keys()
        
        return Memory(
            id=UUID(row["id"]),
            project_id=UUID(row["project_id"]),
            content=row["content"],
            type=MemoryType(row["type"]),
            source=MemorySource(row["source"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            confirmed=bool(row["confirmed"]),
            metadata=eval(row["metadata"]) if row["metadata"] else {},
            # v2 fields with safe fallbacks
            is_stale=bool(row["is_stale"]) if "is_stale" in row_keys and row["is_stale"] else False,
            stale_reason=row["stale_reason"] if "stale_reason" in row_keys else None,
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if "last_accessed" in row_keys and row["last_accessed"] else None,
            is_archived=bool(row["is_archived"]) if "is_archived" in row_keys and row["is_archived"] else False,
            consolidated_into=UUID(row["consolidated_into"]) if "consolidated_into" in row_keys and row["consolidated_into"] else None,
        )
    
    def list_memories(
        self,
        project_id: UUID,
        confirmed_only: bool = True,
        memory_type: Optional[MemoryType] = None,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories for a project with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM memories WHERE project_id = ?"
            params: list = [str(project_id)]
            
            if confirmed_only:
                query += " AND confirmed = 1"
            
            if memory_type:
                query += " AND type = ?"
                params.append(memory_type.value)
                
            if not include_archived:
                # Only filter out archived if not explicitly requested
                # If table has is_archived column (checked via exception or assumption)
                # Since we added it via migration, it should exist.
                query += " AND (is_archived = 0 OR is_archived IS NULL)"
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    def confirm_memory(self, memory_id: UUID) -> bool:
        """Confirm a memory (makes it eligible for indexing and retrieval)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET confirmed = 1, updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    def update_memory(self, memory_id: UUID, content: str) -> bool:
        """Update memory content."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET content = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    def delete_memory(self, memory_id: UUID) -> bool:
        """Delete a memory and its embedding reference."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete embedding reference first
            cursor.execute(
                "DELETE FROM embeddings WHERE memory_id = ?",
                (str(memory_id),),
            )
            
            # Delete memory
            cursor.execute(
                "DELETE FROM memories WHERE id = ?",
                (str(memory_id),),
            )
            
            return cursor.rowcount > 0
    
    def get_confirmed_memory_ids(self, project_id: UUID) -> list[UUID]:
        """Get all confirmed memory IDs for a project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM memories 
                WHERE project_id = ? AND confirmed = 1
                """,
                (str(project_id),),
            )
            rows = cursor.fetchall()
            return [UUID(row["id"]) for row in rows]
    
    # ========== Embedding Operations ==========
    
    def save_embedding_reference(self, memory_id: UUID, vector_id: str) -> None:
        """Save a reference linking a memory to its Qdrant vector."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (memory_id, vector_id)
                VALUES (?, ?)
                """,
                (str(memory_id), vector_id),
            )
    
    def get_embedding_reference(self, memory_id: UUID) -> Optional[str]:
        """Get the Qdrant vector ID for a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector_id FROM embeddings WHERE memory_id = ?",
                (str(memory_id),),
            )
            row = cursor.fetchone()
            return row["vector_id"] if row else None
    
    def delete_embedding_reference(self, memory_id: UUID) -> bool:
        """Delete the embedding reference for a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM embeddings WHERE memory_id = ?",
                (str(memory_id),),
            )
            return cursor.rowcount > 0
    
    def get_memory_count(self, project_id: UUID, confirmed_only: bool = True) -> int:
        """Get the count of memories for a project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) as count FROM memories WHERE project_id = ?"
            params = [str(project_id)]
            
            if confirmed_only:
                query += " AND confirmed = 1"
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row["count"] if row else 0
    
    # ========== v2 Operations ==========
    
    def _add_v2_columns(self, conn: sqlite3.Connection) -> None:
        """Add v2 columns to memories table if they don't exist."""
        cursor = conn.cursor()
        
        # Get existing columns
        cursor.execute("PRAGMA table_info(memories)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # v2 columns to add
        v2_columns = [
            ("is_stale", "INTEGER DEFAULT 0"),
            ("stale_reason", "TEXT"),
            ("last_accessed", "TEXT"),
            ("is_archived", "INTEGER DEFAULT 0"),
            ("consolidated_into", "TEXT"),
        ]
        
        for col_name, col_type in v2_columns:
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
    
    def can_delete_project(self, project_id: UUID) -> bool:
        """Check if a project can be deleted (no memories exist)."""
        return self.get_memory_count(project_id, confirmed_only=False) == 0
    
    def delete_project(self, project_id: UUID) -> bool:
        """Delete a project (only if no memories exist)."""
        if not self.can_delete_project(project_id):
            return False
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM projects WHERE id = ?",
                (str(project_id),),
            )
            return cursor.rowcount > 0
    
    # ========== Memory Version Operations ==========
    
    def save_memory_version(self, memory_id: UUID, content: str, version: int) -> MemoryVersion:
        """Save a memory content version for history/rollback."""
        from uuid import uuid4
        
        version_record = MemoryVersion(
            id=uuid4(),
            memory_id=memory_id,
            content=content,
            version=version,
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_versions (id, memory_id, content, version, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(version_record.id),
                    str(version_record.memory_id),
                    version_record.content,
                    version_record.version,
                    version_record.created_at.isoformat(),
                ),
            )
        return version_record
    
    def get_memory_versions(self, memory_id: UUID) -> list[MemoryVersion]:
        """Get all versions of a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM memory_versions 
                WHERE memory_id = ? 
                ORDER BY version DESC
                """,
                (str(memory_id),),
            )
            rows = cursor.fetchall()
            
            return [
                MemoryVersion(
                    id=UUID(row["id"]),
                    memory_id=UUID(row["memory_id"]),
                    content=row["content"],
                    version=row["version"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]
    
    def get_next_version_number(self, memory_id: UUID) -> int:
        """Get the next version number for a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(version) as max_v FROM memory_versions WHERE memory_id = ?",
                (str(memory_id),),
            )
            row = cursor.fetchone()
            return (row["max_v"] or 0) + 1
    
    # ========== Memory Link Operations ==========
    
    def create_memory_link(
        self,
        memory_id: UUID,
        commit_sha: str,
        link_type: LinkType,
    ) -> MemoryLink:
        """Create a link between a memory and a git commit."""
        from uuid import uuid4
        
        link = MemoryLink(
            id=uuid4(),
            memory_id=memory_id,
            commit_sha=commit_sha,
            link_type=link_type,
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_links (id, memory_id, commit_sha, link_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(link.id),
                    str(link.memory_id),
                    link.commit_sha,
                    link.link_type.value,
                    link.created_at.isoformat(),
                ),
            )
        return link
    
    def get_memories_by_commit(self, commit_sha: str) -> list[Memory]:
        """Get all memories linked to a commit."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.* FROM memories m
                INNER JOIN memory_links ml ON m.id = ml.memory_id
                WHERE ml.commit_sha = ?
                """,
                (commit_sha,),
            )
            rows = cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
    
    def get_memory_links(self, memory_id: UUID) -> list[MemoryLink]:
        """Get all git links for a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memory_links WHERE memory_id = ?",
                (str(memory_id),),
            )
            rows = cursor.fetchall()
            
            return [
                MemoryLink(
                    id=UUID(row["id"]),
                    memory_id=UUID(row["memory_id"]),
                    commit_sha=row["commit_sha"],
                    link_type=LinkType(row["link_type"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]
            
    def get_recent_memories(self, project_id: UUID, limit: int = 20) -> list[Memory]:
        """Get most recent memories for a project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE project_id = ? AND is_archived = 0
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (str(project_id), limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
    
    # ========== Staleness Operations ==========
    
    def mark_stale(self, memory_id: UUID, reason: str) -> bool:
        """Mark a memory as stale."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET is_stale = 1, stale_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (reason, datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    def clear_stale(self, memory_id: UUID) -> bool:
        """Clear the stale flag from a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET is_stale = 0, stale_reason = NULL, updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    def get_stale_memories(self, project_id: UUID) -> list[Memory]:
        """Get all stale memories for a project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE project_id = ? AND is_stale = 1
                ORDER BY created_at DESC
                """,
                (str(project_id),),
            )
            rows = cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
    
    def update_last_accessed(self, memory_id: UUID) -> bool:
        """Update the last_accessed timestamp (called on retrieval only)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET last_accessed = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    # ========== Consolidation Operations ==========
    
    def archive_memory(self, memory_id: UUID, consolidated_into: UUID) -> bool:
        """Archive a memory (mark as consolidated into another)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET is_archived = 1, consolidated_into = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(consolidated_into), datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    def restore_archived_memory(self, memory_id: UUID) -> bool:
        """Restore an archived memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories 
                SET is_archived = 0, consolidated_into = NULL, updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), str(memory_id)),
            )
            return cursor.rowcount > 0
    
    def get_archived_memories(self, consolidated_into: UUID) -> list[Memory]:
        """Get memories that were consolidated into a specific memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE consolidated_into = ?
                ORDER BY created_at DESC
                """,
                (str(consolidated_into),),
            )
            rows = cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
    
    def get_all_archived_memories(self, project_id: UUID) -> list[Memory]:
        """Get all archived memories for a project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE project_id = ? AND is_archived = 1
                ORDER BY created_at DESC
                """,
                (str(project_id),),
            )
            rows = cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
    
    # ========== Schema Version Operations ==========
    
    def get_schema_version(self) -> int:
        """Get the current schema version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT MAX(version) as v FROM schema_version")
                row = cursor.fetchone()
                return row["v"] if row and row["v"] else 1
            except sqlite3.OperationalError:
                return 1  # Table doesn't exist, assume v1
    
    def set_schema_version(self, version: int, description: str) -> None:
        """Record a schema version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO schema_version (version, applied_at, description)
                VALUES (?, ?, ?)
                """,
                (version, datetime.utcnow().isoformat(), description),
            )
