"""Test script to debug migration."""
from pathlib import Path
import tempfile
import sqlite3
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock
from memoryforge.migrate import Migrator

# Create temp dir and v1 database
tmpdir = tempfile.mkdtemp()
db_path = Path(tmpdir) / 'test.db'

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.execute('CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, root_path TEXT NOT NULL, created_at TEXT NOT NULL)')
cur.execute('CREATE TABLE memories (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, content TEXT NOT NULL, type TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL, confirmed INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL DEFAULT "{}")')
cur.execute('CREATE TABLE embeddings (memory_id TEXT PRIMARY KEY, vector_id TEXT NOT NULL)')
project_id = str(uuid4())
cur.execute('INSERT INTO projects VALUES (?,?,?,?)', (project_id, 'test', '/test', datetime.utcnow().isoformat()))
cur.execute('INSERT INTO memories (id, project_id, content, type, source, created_at, confirmed, metadata) VALUES (?,?,?,?,?,?,?,?)', 
    (str(uuid4()), project_id, 'Test memory', 'note', 'manual', datetime.utcnow().isoformat(), 1, '{}'))
conn.commit()
conn.close()

# Run migration
config = Mock()
config.sqlite_path = db_path
migrator = Migrator(config)
print('Schema version before:', migrator._get_schema_version())
success, error = migrator.run_migration()
print('Success:', success)
print('Error:', error)
print('Schema version after:', migrator._get_schema_version())
