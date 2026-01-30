"""
Migration Tool for MemoryForge v2.

Safely migrates v1 databases to v2 schema with rollback support.
Always backs up database before migrating.

Migrations handled:
- Adding v2 columns (is_stale, last_accessed, etc.)
- Creating new tables (schema_version, memory_versions, memory_links)
- Initializing schema version if missing
"""

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from memoryforge.config import Config
from memoryforge.storage.sqlite_db import SQLiteDatabase

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration fails."""
    pass


class Migrator:
    """
    Handles database migrations safely.
    
    Principles:
    1. Always backup first
    2. Use transactions
    3. Verify success or rollback
    """
    
    def __init__(self, config: Config):
        """Initialize migrator."""
        self.config = config
        self.db_path = config.sqlite_path
    
    def backup_database(self) -> Path:
        """
        Create a backup of the current database.
        
        Returns:
            Path to the backup file
        """
        if not self.db_path.exists():
            raise MigrationError("Database file not found")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"memoryforge_v1_backup_{timestamp}.sqlite"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Created database backup: {backup_path}")
            return backup_path
        except Exception as e:
            raise MigrationError(f"Failed to create backup: {e}")
    
    def restore_backup(self, backup_path: Path) -> None:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
        """
        if not backup_path.exists():
            raise MigrationError(f"Backup file not found: {backup_path}")
        
        try:
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Restored database from: {backup_path}")
        except Exception as e:
            raise MigrationError(f"Failed to restore backup: {e}")
    
    def run_migration(self) -> bool:
        """
        Run migration from v1 to v2.
        
        Returns:
            True if migration successful (or already done)
        """
        if not self.db_path.exists():
            self._init_new_db()
            return True
        
        # 1. Check current version
        current_version = self._get_schema_version()
        if current_version >= 2:
            logger.info("Database is already at v2 or higher")
            return True
        
        logger.info(f"Migrating database from v{current_version} to v2...")
        
        # 2. Backup
        backup_path = self.backup_database()
        
        # 3. Migrate
        try:
            self._perform_migration()
            logger.info("Migration to v2 successful")
            return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            logger.info("Restoring backup...")
            self.restore_backup(backup_path)
            return False
    
    def _init_new_db(self) -> None:
        """Initialize a new database (implicitly v2 via SQLiteDatabase class)."""
        logger.info("Initializing new database (v2)")
        db = SQLiteDatabase(self.db_path)
        # SQLiteDatabase init already creates full v2 schema
        # We just need to verify schema_version is set
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (2, ?)",
                (datetime.utcnow().isoformat(),)
            )
    
    def _get_schema_version(self) -> int:
        """Get current schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if schema_version table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                )
                if not cursor.fetchone():
                    return 1  # No version table implies v1
                
                cursor.execute("SELECT MAX(version) FROM schema_version")
                row = cursor.fetchone()
                return row[0] if row and row[0] else 1
        except Exception:
            return 1
    
    def _perform_migration(self) -> None:
        """Execute migration SQL commands."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Use SQLiteDatabase internal method if available, or manual SQL
            # Since we modified SQLiteDatabase to handle initialization, we can replicate specific steps here
            
            # 1. Create schema_version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. Add v2 columns to memories table
            # Check existing columns to avoid errors
            cursor.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            
            v2_columns = {
                "is_stale": "BOOLEAN DEFAULT 0",
                "stale_reason": "TEXT",
                "last_accessed": "TIMESTAMP",
                "is_archived": "BOOLEAN DEFAULT 0",
                "consolidated_into": "TEXT"
            }
            
            for col_name, col_def in v2_columns.items():
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
            
            # 3. Create memory_versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_versions (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # 4. Create memory_links table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # 5. Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_sha ON memory_links(commit_sha)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_versions_mid ON memory_versions(memory_id)")
            
            # 6. Update version
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (2, ?)",
                (datetime.utcnow().isoformat(),)
            )
            
            conn.commit()
