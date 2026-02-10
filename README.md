# MemoryForge

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-8%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.1-purple)

> **Local-first memory layer for AI coding assistants**

MemoryForge provides persistent, semantic memory for AI pair programmers. Store decisions, constraints, and context that survives across sessions.

---

## Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Vector-based retrieval using Qdrant |
| **Memory Types** | Stack, Decision, Constraint, Convention, Note |
| **Multi-Project** | Separate memory indexes per project |
| **Consolidation** | Merge similar memories with rollback support |
| **Staleness Tracking** | Mark outdated memories, filter from search |
| **Git Integration** | Link memories to commits |
| **Team Sync** | Encrypted sync to shared storage |
| **Graph Memory** | Memory-to-memory relationships (v3) |
| **Confidence Scoring** | Trust scores for prioritized retrieval (v3) |
| **Conflict Resolution** | Sync conflict detection and logging (v3) |
| **MCP Server** | IDE integration via Model Context Protocol |

---

## Installation

### Stable Release (Recommended)

```bash
# Basic installation
pip install memoryforge

# With local embeddings (free, no API key needed)
pip install memoryforge[local]

# With OpenAI embeddings
pip install memoryforge[openai]

# With team sync
pip install memoryforge[sync]

# Everything
pip install memoryforge[all]
```

### Development Version (Latest)

```bash
# From GitHub
pip install git+https://github.com/agusain2001/memoryforge.git

# With extras
pip install "memoryforge[local] @ git+https://github.com/agusain2001/memoryforge.git"
```

**Requirements:** Python 3.11+

---

## Quick Start

```bash
# Initialize MemoryForge
cd your-project
memoryforge init --name "my-project"

# Add a memory
memoryforge add "Using FastAPI for backend, PostgreSQL for data" --type stack

# Search memories
memoryforge search "what database?"

# List recent memories
memoryforge list --limit 5
```

---

## CLI Reference

### Core Commands

```bash
memoryforge init [OPTIONS]              # Initialize MemoryForge
  -n, --name TEXT                       # Project name
  -p, --provider [local|openai]         # Embedding provider (default: local)
  -k, --api-key TEXT                    # OpenAI API key (if using openai)

memoryforge add CONTENT [OPTIONS]       # Add a memory
  -t, --type [stack|decision|constraint|convention|note]
  --confirm / --no-confirm              # Immediately confirm

memoryforge list [OPTIONS]              # List memories
  -t, --type TYPE                       # Filter by type
  -a, --all                             # Include unconfirmed
  -l, --limit INTEGER                   # Max results

memoryforge search QUERY [OPTIONS]      # Semantic search
  -t, --type TYPE                       # Filter by type
  -l, --limit INTEGER                   # Max results

memoryforge confirm MEMORY_ID           # Confirm a pending memory
memoryforge delete MEMORY_ID            # Delete a memory
memoryforge timeline [--limit N]        # Chronological view
memoryforge status                      # Show project status
memoryforge reindex [--force]           # Rebuild vector index
memoryforge serve                       # Start MCP server
```

### Project Management

```bash
memoryforge project create --name NAME [--path PATH]
memoryforge project list
memoryforge project switch NAME_OR_ID
memoryforge project delete NAME_OR_ID
```

### Memory Consolidation

```bash
memoryforge consolidate suggest [--limit N]
memoryforge consolidate apply ID1 ID2 ... --content "merged content"
memoryforge consolidate rollback CONSOLIDATED_ID
memoryforge consolidate stats
```

### Staleness Management

```bash
memoryforge stale list
memoryforge stale mark MEMORY_ID --reason "outdated"
memoryforge stale clear MEMORY_ID
memoryforge stale unused [--days N]
```

### Git Integration

```bash
memoryforge git status
memoryforge git sync [--limit N]        # Find architectural commits
memoryforge git link MEMORY_ID COMMIT_SHA
memoryforge git activity [--days N]
```

### Team Sync

```bash
memoryforge sync init --path ./shared-folder
memoryforge sync push [--force]
memoryforge sync pull
memoryforge sync status
```

### Database Migration

```bash
memoryforge migrate [--rollback] [--backup-file FILE]
```

### Graph Memory (v3)

```bash
memoryforge graph view MEMORY_ID           # View memory relationships
memoryforge graph link SRC_ID TGT_ID       # Create relationship
  -t, --type [caused_by|supersedes|relates_to|blocks|depends_on]
```

### Conflict Management (v3)

```bash
memoryforge conflicts list [--memory-id ID]  # List sync conflicts
memoryforge conflicts show MEMORY_ID         # Detailed conflict history
```

### Confidence Scoring (v3)

```bash
memoryforge confidence show MEMORY_ID      # View confidence breakdown
memoryforge confidence update MEMORY_ID    # Recalculate score
memoryforge confidence low [--threshold N] # List low-confidence memories
memoryforge confidence refresh             # Batch update all scores
```

### Memory Sharing (v3)

```bash
memoryforge share memory MEMORY_ID          # Share a memory with team
  --with TEAM                               # Share target (default: team)
  -n, --note TEXT                           # Add a note for recipients
memoryforge share list                      # List shared memories
memoryforge share import FILENAME           # Import shared memory
```

---

## Configuration

Configuration stored in `~/.memoryforge/config.yaml`:

```yaml
embedding_provider: local              # "local" or "openai"
local_embedding_model: all-MiniLM-L6-v2
openai_api_key: sk-...                 # Required if using openai
openai_embedding_model: text-embedding-3-small
max_results: 5
min_score: 0.5
active_project_id: <uuid>
```

> **Note:** Changing `embedding_provider` requires re-indexing: `memoryforge reindex --force`

---

## MCP Server

MemoryForge integrates with AI IDEs via the Model Context Protocol:

```bash
memoryforge serve
```

### IDE Configuration (Cursor/VSCode)

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

### Available Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Store a new memory |
| `search_memory` | Semantic search |
| `list_memory` | List recent memories |
| `delete_memory` | Delete a memory |
| `memory_timeline` | Chronological view |
| `list_projects` | List all projects |
| `switch_project` | Switch active project |
| `project_status` | Current project info |

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `vector dimension mismatch` | Embedding provider changed | `rm -rf ~/.memoryforge/qdrant && memoryforge reindex` |
| `Memory not found in search` | Qdrant indexing failed | `memoryforge confirm <id> --force` |
| `database is locked` | Multiple processes | Ensure single CLI/MCP instance |
| `SyncConflictError` | Concurrent team edits | `memoryforge sync push --force` |

---

## Development

```bash
git clone https://github.com/agusain2001/memoryforge.git
cd memoryforge
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License - see [LICENSE](LICENSE)
