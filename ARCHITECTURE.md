# MemoryForge Architecture

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            MemoryForge v1.0                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │     CLI      │    │  MCP Server  │    │  Python API  │                 │
│  │   (click)    │    │   (stdio)    │    │   (direct)   │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Core Services Layer                            │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐  │  │
│  │  │   Memory    │ │  Retrieval  │ │   Project   │ │     Git      │  │  │
│  │  │   Manager   │ │   Engine    │ │   Router    │ │ Integration  │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └──────────────┘  │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │  │
│  │  │ Validation  │ │Consolidation│ │   Sync      │                   │  │
│  │  │   Layer     │ │   Engine    │ │  Manager    │                   │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                             │                                              │
│         ┌───────────────────┼───────────────────┐                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │   SQLite    │     │   Qdrant    │     │  Embedding  │                  │
│  │  (Primary)  │◄────│  (Vectors)  │◄────│   Factory   │                  │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                  │
│                                                  │                          │
│                              ┌───────────────────┼───────────────────┐     │
│                              ▼                                       ▼     │
│                       ┌─────────────┐                         ┌──────────┐ │
│                       │    Local    │                         │  OpenAI  │ │
│                       │ (MiniLM-L6) │                         │   API    │ │
│                       └─────────────┘                         └──────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
memoryforge/
├── __init__.py              # Package exports
├── __main__.py              # Entry point
├── cli.py                   # CLI commands (click)
├── config.py                # Configuration management
├── models.py                # Pydantic data models
├── migrate.py               # Database migration logic
│
├── core/                    # Core business logic
│   ├── memory_manager.py    # CRUD operations for memories
│   ├── retrieval.py         # Semantic search engine
│   ├── validation.py        # Input validation rules
│   ├── project_router.py    # Multi-project management
│   ├── memory_consolidator.py # Memory merging
│   ├── embedding_factory.py # Embedding provider selector
│   ├── embedding_service.py # OpenAI embeddings
│   ├── local_embedding_service.py  # sentence-transformers
│   ├── git_integration.py   # Git-memory links
│   └── git_scanner.py       # Architectural commit detection
│
├── storage/                 # Persistence layer
│   ├── sqlite_db.py         # SQLite operations
│   └── qdrant_store.py      # Vector store operations
│
├── sync/                    # Team sync (v2.1)
│   ├── adapter.py           # Sync backend interface
│   ├── encryption.py        # AES-256 encryption
│   ├── local_file_adapter.py # File-based sync
│   └── manager.py           # Sync orchestration
│
└── mcp/                     # MCP integration
    ├── __init__.py
    └── server.py            # MCP server implementation
```

---

## Data Flow

### Memory Creation

```
User Input
    │
    ▼
┌─────────────┐
│  Validation │  ← Type, content length, format checks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SQLite    │  ← Memory stored (confirmed=false)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Confirm   │  ← User confirms memory is correct
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │  ← Generate vector (local or OpenAI)
│   Factory   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Qdrant    │  ← Store vector with memory ID
└─────────────┘
```

### Semantic Search

```
Search Query
    │
    ▼
┌─────────────┐
│  Embedding  │  ← Query → Vector
│   Factory   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Qdrant    │  ← ANN search, return top-k IDs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SQLite    │  ← Fetch full memory data
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │  ← Score, rank, filter stale
│   Engine    │
└──────┬──────┘
       │
       ▼
Search Results
```

---

## Core Components

### Memory Manager
**File:** `core/memory_manager.py`

Handles CRUD operations for memories:
- Create (unconfirmed) → Confirm → Generate embedding
- Update with version history
- Delete (removes from SQLite and Qdrant)

### Retrieval Engine
**File:** `core/retrieval.py`

Semantic search with:
- Query embedding generation
- Qdrant similarity search
- Staleness filtering
- Score-based ranking

### Project Router
**File:** `core/project_router.py`

Multi-project support:
- Project creation/switching
- Path-based auto-detection
- Separate Qdrant collections per project

### Memory Consolidator
**File:** `core/memory_consolidator.py`

Memory merging:
- Similarity detection (threshold: 0.90)
- Archive originals with rollback support
- Stats tracking

### Embedding Factory
**File:** `core/embedding_factory.py`

Provider abstraction:
- Local: `sentence-transformers` (all-MiniLM-L6-v2, 384 dims)
- OpenAI: text-embedding-3-small (1536 dims)

---

## Storage Layer

### SQLite (Primary Store)
**File:** `storage/sqlite_db.py`

Source of truth for all data:

| Table | Description |
|-------|-------------|
| `projects` | Project metadata |
| `memories` | Memory content, type, flags |
| `memory_versions` | Content history for rollback |
| `memory_links` | Git commit associations |
| `embeddings` | Memory-to-vector-ID mapping |

### Qdrant (Derived Store)
**File:** `storage/qdrant_store.py`

Vector storage (can be rebuilt from SQLite):
- One collection per project
- Dimension matches embedding provider
- Supports filtered search

---

## Data Models

**File:** `models.py`

```python
class MemoryType(Enum):
    STACK = "stack"           # Languages, frameworks, libraries
    DECISION = "decision"     # Architecture decisions
    CONSTRAINT = "constraint" # Performance, deadlines
    CONVENTION = "convention" # Code conventions
    NOTE = "note"             # General notes

class Memory(BaseModel):
    id: UUID
    content: str              # Max 10KB
    type: MemoryType
    project_id: UUID
    confirmed: bool
    is_stale: bool
    is_archived: bool
    created_at: datetime
    last_accessed: datetime   # Updated on retrieval
```

---

## MCP Integration

**File:** `mcp/server.py`

Exposes tools via Model Context Protocol:

| Tool | Operation |
|------|-----------|
| `store_memory` | Create + optional confirm |
| `search_memory` | Semantic search |
| `list_memory` | Recent memories |
| `delete_memory` | Remove by ID |
| `memory_timeline` | Chronological view |
| `list_projects` | All projects |
| `switch_project` | Change active project |
| `project_status` | Current project info |

---

## Team Sync

**Files:** `sync/manager.py`, `sync/encryption.py`

Encrypted sync to shared storage:

```
Local SQLite
    │
    ▼
┌─────────────┐
│  Export     │  ← Serialize memories to JSON
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Encryption  │  ← AES-256-GCM
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Adapter   │  ← Write to shared path
└─────────────┘
```

Conflict detection via timestamps and checksums.

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Memory creation | <50ms | SQLite write |
| Confirmation | 100-500ms | Embedding + Qdrant |
| Semantic search | 50-200ms | Depends on collection size |
| Max memories | ~100K | Per project, limited by RAM |

### Embedding Providers

| Provider | Dimensions | Speed | Cost |
|----------|------------|-------|------|
| Local (MiniLM) | 384 | Fast | Free |
| OpenAI (3-small) | 1536 | Network | ~$0.02/1K |

---

## Key Design Decisions

1. **SQLite as Source of Truth**
   - All data restorable from SQLite
   - Qdrant is derived (can rebuild via `reindex`)

2. **Two-Phase Confirmation**
   - Memories created as unconfirmed
   - Embedding only on confirmation (saves cost/compute)

3. **Local-First Architecture**
   - All data stored locally by default
   - Sync is opt-in, encrypted

4. **Provider Abstraction**
   - Switch embedding providers without code changes
   - Dimension mismatch requires reindex
