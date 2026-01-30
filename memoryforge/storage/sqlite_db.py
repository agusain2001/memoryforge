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

from memoryforge.models import Memory, MemoryType, MemorySource, Project


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
                (id, project_id, content, type, source, created_at, updated_at, confirmed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
        return memory
    
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
        )
    
    def list_memories(
        self,
        project_id: UUID,
        confirmed_only: bool = True,
        memory_type: Optional[MemoryType] = None,
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
