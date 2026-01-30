# MemoryForge v1.0.0

> **Local-first memory layer for AI coding assistants**

MemoryForge provides persistent, semantic memory for AI pair programmers. Store decisions, constraints, and context that survives across sessions.

---

## Features

### Core Memory System
- **Semantic Search**: Vector-based retrieval using Qdrant for finding relevant memories
- **Memory Types**: Stack, Decision, Constraint, Convention, Note
- **Confirmation Flow**: Memories require explicit confirmation before becoming permanent
- **SQLite Source of Truth**: All data persisted locally, vectors are derived

### v2 Features
- **Multi-Project Support**: Separate memory indexes per project with automatic routing
- **Memory Consolidation**: Merge similar memories (threshold: 0.90) with rollback support
- **Staleness Tracking**: Mark outdated memories, filter from search
- **Git Integration**: Link memories to commits, analyze architectural changes
- **Version History**: Track memory revisions for rollback

### v2.1 Features  
- **Team Sync**: Encrypted sync to cloud storage (Git, Dropbox, S3)
- **Conflict Detection**: Timestamp-based conflict detection with checksums
- **MCP Server**: Model Context Protocol for IDE integration
- **Multi-Version Migration**: Safe incremental database upgrades (v1→v2→v3)

---

## Installation

```bash
# Basic installation
pip install memoryforge

# With local embeddings (recommended)
pip install memoryforge[local]

# With OpenAI embeddings
pip install memoryforge[openai]

# With team sync
pip install memoryforge[sync]

# Everything
pip install memoryforge[all]
```

### Requirements
- Python 3.11+
- ~500MB disk for local embedding model (if using local provider)

---

## Quick Start

```bash
# Initialize MemoryForge for your project
cd your-project
memoryforge init --name "my-project"

# Store a memory
memoryforge add "Using FastAPI for backend, PostgreSQL for persistence" --type stack

# Search memories
memoryforge search "what database are we using?"

# List recent memories
memoryforge list --limit 5
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MemoryForge v2                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   CLI       │    │  MCP Server │    │   Python    │     │
│  │  Interface  │    │   (stdio)   │    │    API      │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                   │            │
│         └──────────────────┼───────────────────┘            │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Memory Manager                       │  │
│  │  • Create/Confirm/Delete memories                     │  │
│  │  • Validation layer                                   │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                 │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐         │
│  │  SQLite    │   │   Qdrant   │   │ Embedding  │         │
│  │  (Source   │◄──│  (Derived  │◄──│  Service   │         │
│  │  of Truth) │   │   Vectors) │   │            │         │
│  └────────────┘   └────────────┘   └────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Data Flow:
1. Memories created → stored in SQLite (confirmed=false)
2. On confirmation → embedding generated → stored in Qdrant
3. On search → query embedded → Qdrant finds similar → SQLite returns full data
4. SQLite is authoritative; Qdrant can be rebuilt from SQLite
```

---

## Configuration

Configuration is stored in `~/.memoryforge/config.yaml`:

```yaml
# Embedding provider: "local" or "openai"
embedding_provider: local

# Local embedding model (if using local provider)
local_embedding_model: all-MiniLM-L6-v2

# OpenAI settings (if using openai provider)
openai_api_key: sk-...
openai_embedding_model: text-embedding-3-small

# Search settings
max_results: 5
min_score: 0.5

# Active project
active_project_id: <uuid>
```

> [!WARNING]
> **Changing `embedding_provider`** requires re-embedding all memories.
> The embedding dimensions differ between providers, so you must:
> 1. Export memories: `memoryforge export memories.json`
> 2. Delete Qdrant collections: `rm -rf ~/.memoryforge/qdrant`
> 3. Re-import with new embeddings: `memoryforge import memories.json --reindex`

---

## CLI Reference

### Project Management
```bash
memoryforge init --name "project"       # Initialize new project
memoryforge projects                     # List all projects
memoryforge switch <name|id>            # Switch active project
```

### Memory Operations
```bash
memoryforge add "<content>" --type <type>  # Add memory (unconfirmed)
memoryforge confirm <id>                    # Confirm memory
memoryforge list [--type <type>] [--limit N]
memoryforge search "<query>" [--limit N]
memoryforge delete <id>
```

### Consolidation (v2)
```bash
memoryforge consolidate suggest         # Find similar memories
memoryforge consolidate apply <id1> <id2> --content "merged"
memoryforge consolidate rollback <id>   # Undo consolidation
```

### Staleness (v2)
```bash
memoryforge stale mark <id> --reason "outdated"
memoryforge stale clear <id>
memoryforge stale list
```

### Git Integration (v2)
```bash
memoryforge git link <memory_id> <commit_sha>
memoryforge git activity                 # Show recent commits
memoryforge git sync                     # Find architectural commits
```

### Team Sync (v2.1)
```bash
memoryforge sync init --backend git --path ./sync
memoryforge sync push                    # Export to sync location
memoryforge sync pull                    # Import from sync location
memoryforge sync status
```

### Migration
```bash
memoryforge migrate                     # Upgrade database schema
memoryforge migrate --verify            # Upgrade with verification
```

---

## MCP Server

MemoryForge provides an MCP server for IDE integration:

```bash
memoryforge serve
```

### Cursor/VSCode Configuration

Add to your MCP settings:

```json
{
  "mcpServers": {
    "memoryforge": {
      "command": "memoryforge",
      "args": ["serve"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Store a new memory |
| `search_memory` | Semantic search |
| `list_memory` | List recent memories |
| `delete_memory` | Delete a memory |
| `memory_timeline` | Get chronological view |
| `list_projects` | List all projects |
| `switch_project` | Switch active project |
| `project_status` | Get current project info |

---

## Troubleshooting

### Common Errors

**"Dimension mismatch" error**
```
QdrantClient error: vector dimension mismatch
```
**Cause**: Embedding provider was changed without rebuilding vectors.
**Fix**: Delete Qdrant data and re-index all memories:
```bash
rm -rf ~/.memoryforge/qdrant
memoryforge reindex
```

**"Memory not found" after confirmation**
```
Memory <id> not found in search results
```
**Cause**: Qdrant indexing failed silently.
**Fix**: Check Qdrant is running, then re-confirm the memory:
```bash
memoryforge confirm <id> --force
```

**"Database locked" on Windows**
```
sqlite3.OperationalError: database is locked
```
**Cause**: Multiple processes accessing the database.
**Fix**: Ensure only one CLI/MCP instance is running.

**Team sync conflicts**
```
SyncConflictError: Conflict detected for memory <id>
```
**Cause**: Same memory edited by multiple team members.
**Fix**: Use `--force` to overwrite, or manually resolve:
```bash
memoryforge sync push --force  # Overwrite remote with local
memoryforge sync pull --force  # Overwrite local with remote
```

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Memory creation | <50ms | SQLite write |
| Memory confirmation | 100-500ms | Embedding + Qdrant upsert |
| Semantic search | 50-200ms | Depends on collection size |
| Max memories per project | ~100,000 | Limited by Qdrant memory |
| SQLite database size | ~1KB/memory | Excludes embeddings |
| Qdrant collection size | ~2KB/memory | 384-dim embeddings |

### Embedding Provider Comparison

| Provider | Dimension | Speed | Quality | Cost |
|----------|-----------|-------|---------|------|
| Local (MiniLM) | 384 | Fast | Good | Free |
| OpenAI (3-small) | 1536 | Network-bound | Better | ~$0.02/1K |
| OpenAI (3-large) | 3072 | Network-bound | Best | ~$0.13/1K |

---

## Development

```bash
# Clone repository
git clone https://github.com/memoryforge/core.git
cd core

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy memoryforge/

# Linting
ruff check memoryforge/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
