"""
Migration Tool for MemoryForge v2.

Safely migrates v1 databases to v2 schema with rollback support.
Always backs up database before migrating.

Migrations handled:
- Adding v2 columns (is_stale, last_accessed, etc.)
- Creating new tables (schema_version, memory_versions, memory_links)
- Initializing schema version if missing

v2.1: Added migration verification and backup retention management.
"""

import glob
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from memoryforge.config import Config
from memoryforge.storage.sqlite_db import SQLiteDatabase

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration fails."""
    pass


class MigrationVerificationError(MigrationError):
    """Raised when migration verification fails."""
    pass


class Migrator:
    """
    Handles database migrations safely.
    
    Principles:
    1. Always backup first
    2. Use transactions
    3. Verify success or rollback
    4. Clean up old backups
    """
    
    MAX_BACKUPS = 5  # Keep only the last 5 backups
    
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
    
    def list_backups(self) -> List[Path]:
        """
        List all backup files, sorted by modification time (newest first).
        
        Returns:
            List of backup file paths
        """
        pattern = str(self.db_path.parent / "memoryforge_*_backup_*.sqlite")
        backups = [Path(p) for p in glob.glob(pattern)]
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: Optional[int] = None) -> int:
        """
        Remove old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of backups to keep (default: MAX_BACKUPS)
            
        Returns:
            Number of backups deleted
        """
        keep_count = keep_count or self.MAX_BACKUPS
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        to_delete = backups[keep_count:]
        deleted = 0
        
        for backup in to_delete:
            try:
                backup.unlink()
                logger.info(f"Deleted old backup: {backup}")
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete backup {backup}: {e}")
        
        return deleted
    
    def run_migration(self, verify: bool = True, target_version: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Run migration to target version (multi-version support).
        
        Supports incremental migrations: v1 → v2 → v3 → ...
        
        Args:
            verify: If True, verify migration success
            target_version: Target schema version (default: latest)
        
        Returns:
            Tuple of (success, error_message)
        """
        LATEST_VERSION = 3  # Current latest schema version
        target = target_version or LATEST_VERSION
        
        if not self.db_path.exists():
            self._init_new_db(target)
            return True, None
        
        # 1. Check current version
        current_version = self._get_schema_version()
        
        if current_version >= target:
            logger.info(f"Database is already at v{current_version} (target: v{target})")
            return True, None
        
        logger.info(f"Migrating database from v{current_version} to v{target}...")
        
        # 2. Get pre-migration counts for verification
        pre_counts = self._get_table_counts() if verify else {}
        
        # 3. Backup
        backup_path = self.backup_database()
        
        # 4. Run incremental migrations
        try:
            for from_version in range(current_version, target):
                to_version = from_version + 1
                logger.info(f"  Running migration v{from_version} → v{to_version}...")
                self._perform_migration_step(from_version, to_version)
            
            # 5. Verify migration
            if verify:
                self._verify_migration(pre_counts)
            
            # 6. Cleanup old backups
            deleted = self.cleanup_old_backups()
            if deleted:
                logger.info(f"Cleaned up {deleted} old backup(s)")
            
            logger.info(f"Migration to v{target} successful")
            return True, None
            
        except MigrationVerificationError as e:
            logger.error(f"Migration verification failed: {e}")
            logger.info("Restoring backup...")
            self.restore_backup(backup_path)
            return False, str(e)
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            logger.info("Restoring backup...")
            self.restore_backup(backup_path)
            return False, str(e)
    
    def _perform_migration_step(self, from_version: int, to_version: int) -> None:
        """Perform a single migration step."""
        migration_method = getattr(self, f"_migrate_v{from_version}_to_v{to_version}", None)
        
        if migration_method is None:
            # Fallback to legacy method for v1→v2
            if from_version == 1 and to_version == 2:
                self._perform_migration()
            else:
                raise MigrationError(f"No migration path from v{from_version} to v{to_version}")
        else:
            migration_method()
    
    def _migrate_v2_to_v3(self) -> None:
        """
        Migrate from v2 to v3.
        
        v3 additions:
        - sync_history table for tracking sync operations
        - memory_tags table for performance
        - Performance indexes
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sync history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    memories_affected INTEGER,
                    conflicts INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'success'
                )
            """)
            
            # Create memory tags for faster filtering
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_tags (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # Add v3 indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sync_history_timestamp ON sync_history(timestamp)")
            
            # Update schema version
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (3, ?)",
                (datetime.utcnow().isoformat(),)
            )
            
            conn.commit()
    
    def _get_table_counts(self) -> dict:
        """Get row counts for all tables."""
        counts = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"Could not get table counts: {e}")
        
        return counts
    
    def _verify_migration(self, pre_counts: dict) -> None:
        """
        Verify migration preserved all data.
        
        Args:
            pre_counts: Pre-migration table counts
            
        Raises:
            MigrationVerificationError: If verification fails
        """
        post_counts = self._get_table_counts()
        
        # Key tables that must preserve data
        critical_tables = ["memories", "projects"]
        
        for table in critical_tables:
            pre = pre_counts.get(table, 0)
            post = post_counts.get(table, 0)
            
            if post < pre:
                raise MigrationVerificationError(
                    f"Data loss detected in '{table}': had {pre} rows, now have {post}"
                )
        
        logger.info(f"Migration verified: {post_counts}")
    
    def _init_new_db(self, target_version: int = 2) -> None:
        """Initialize a new database at target version."""
        logger.info(f"Initializing new database (v{target_version})")
        db = SQLiteDatabase(self.db_path)
        # SQLiteDatabase init already creates full v2 schema
        # Run any additional migrations if target > 2
        if target_version >= 2:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (2, ?)",
                    (datetime.utcnow().isoformat(),)
                )
        
        if target_version >= 3:
            self._migrate_v2_to_v3()
    
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
    
    def get_rollback_warning(self) -> Optional[str]:
        """
        Get warning about potential data loss from rollback.
        
        Returns:
            Warning message if there's risk, None otherwise
        """
        backups = self.list_backups()
        if not backups:
            return None
        
        latest_backup = backups[0]
        backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
        
        # Count memories created after backup
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE created_at > ?",
                    (backup_time.isoformat(),)
                )
                new_memories = cursor.fetchone()[0]
                
                if new_memories > 0:
                    return (
                        f"WARNING: Rolling back will delete {new_memories} memories "
                        f"created after {backup_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
        except Exception:
            pass
        
        return None
