# MemoryForge v2

**Intelligent, team-aware memory for AI coding assistants.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()

---

## What Is This?

MemoryForge is a **local-first memory layer** for AI coding assistants like Claude, GPT, and Cursor.
It enables your AI to "remember" project details across sessions and conversations.

**Key Features:**
- ✅ **Tech Stack & Decisions**: Remembers architecture, constraints, and why decisions were made.
- ✅ **Multi-Project Support (v2)**: Isolates memories per project.
- ✅ **Git Integration (v2)**: Links memories to commits and scans for architectural changes.
- ✅ **Memory Consolidation (v2)**: Merges similar memories to keep context clean.
- ✅ **Team Sync (v2.1)**: Secure, encrypted synchronization for teams.
- ✅ **Local privacy**: All data stored locally in SQLite and Qdrant.

---

## Quick Start

### Installation

```bash
# Install with local embeddings (FREE, recommended)
pip install -e ".[local]"

# Install with Team Sync support
pip install -e ".[local,sync]"
```

### Initialize

```bash
cd your-project

# Initialize MemoryForge for this project
memoryforge init
```

### Basic Usage

```bash
# Add a memory
memoryforge add "We use FastAPI with Pydantic v2" --type stack

# Search memories
memoryforge search "What backend framework?"

# View recent memory timeline
memoryforge timeline

# List memories
memoryforge list
```

### Project Management (v2)

```bash
# Check status
memoryforge status

# Switch projects
memoryforge project switch <project_id>

# list projects
memoryforge project list
```

### Team Sync (v2.1)

Securely share memories using any shared folder (Dropbox, Git repo, Network drive).

```bash
# 1. Initialize Sync (generates encryption key)
memoryforge sync init --path /path/to/shared/folder

# 2. Push local memories
memoryforge sync push

# 3. Pull team updates
memoryforge sync pull
```

**Security**: All shared data is encrypted with AES-256 (Fernet). Share the generated key securely with your team.

### Git Integration (v2)

Enable in `config.yaml`: `enable_git_integration: true`

```bash
# View git integration status
memoryforge git status

# Scan for architectural commits
memoryforge git sync

# Link a memory to a specific commit
memoryforge git link <memory_id> <commit_sha>
```

### Memory Consolidation (v2)

Keep your memory bank clean by merging duplicates.

```bash
# Suggest consolidations
memoryforge consolidate suggest

# Apply consolidation
memoryforge consolidate apply <id1> <id2> --content "Merged content..."

# Undo consolidation
memoryforge consolidate rollback <consolidated_id>
```

### Maintenance

```bash
# Find stale/unused memories
memoryforge stale unused --days 30

# Migrate database schema (v1 -> v2)
memoryforge migrate
```

---

## Configuration

Config file: `~/.memoryforge/config.yaml` or project-local `.memoryforge/config.yaml`.

```yaml
# v2 Configuration
active_project_id: "..."  # Managed automatically
consolidation_threshold: 0.90
enable_git_integration: true

# v2.1 Sync Settings
sync_path: "/path/to/shared"
sync_backend: "local"

# Embedding settings
embedding_provider: "local"
local_embedding_model: "all-MiniLM-L6-v2"
```

---

## Architecture

MemoryForge uses a dual-storage approach:
1. **SQLite**: Relational data, project metadata, version history.
2. **Qdrant**: Vector storage for semantic search.

Embeddings are generated locally (via `sentence-transformers`) or via OpenAI API.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,local,sync]"

# Run tests
python -m pytest tests/ -v

# Type checking
mypy memoryforge/
```

## License

MIT License - see [LICENSE](LICENSE)
