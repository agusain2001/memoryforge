# MemoryForge

**Simple, practical memory for AI coding assistants**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()

---

## What Is This?

MemoryForge is a **local-first memory layer** for AI coding assistants like Claude, GPT, and Cursor.

It remembers your:
- ✅ Tech stack and frameworks
- ✅ Architecture decisions and why you made them
- ✅ Code conventions and preferences
- ✅ Constraints and past learnings

**No more re-explaining your project every conversation.**

---

## Quick Start

### Installation

```bash
# Install with local embeddings (FREE, recommended)
pip install -e ".[local]"

# OR with OpenAI embeddings (paid, higher quality)
pip install -e ".[openai]"

# OR install both
pip install -e ".[all]"
```

### Initialize

```bash
cd your-project

# Use local embeddings (default, FREE)
memoryforge init

# Or use OpenAI embeddings
memoryforge init --provider openai
```

### Add Memories

```bash
memoryforge add "We use FastAPI with Pydantic v2" --type stack
memoryforge add "No Redis - too complex for deployment" --type decision
```

### Search Memories

```bash
memoryforge search "What backend framework?"
```

### Start MCP Server

```bash
memoryforge serve
```

Add to your Claude Desktop config:
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

---

## Embedding Providers

| Provider | Cost | Quality | Setup |
|----------|------|---------|-------|
| **local** (default) | 🟢 FREE | Good | `pip install memoryforge[local]` |
| **openai** | 💰 $0.02/1M tokens | Best | API key required |

### Local Embeddings (sentence-transformers)
- **100% free** - no API costs
- Works **offline**
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- First load downloads ~90MB

### OpenAI Embeddings
- Higher quality for nuanced queries
- Model: `text-embedding-3-small` (1536 dimensions)
- Requires API key

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `memoryforge init` | Initialize for current project |
| `memoryforge add "content" --type TYPE` | Add a memory |
| `memoryforge list` | List all memories |
| `memoryforge search "query"` | Semantic search |
| `memoryforge delete ID` | Delete a memory |
| `memoryforge confirm ID` | Confirm pending memory |
| `memoryforge serve` | Start MCP server |

### Memory Types

- `stack` - Tech stack (languages, frameworks)
- `decision` - Architecture decisions
- `constraint` - Limitations and constraints
- `convention` - Code conventions
- `note` - General notes

---

## Configuration

Config file: `~/.memoryforge/config.yaml`

```yaml
project_name: "my-project"
embedding_provider: "local"  # or "openai"

# Local embedding settings
local_embedding_model: "all-MiniLM-L6-v2"

# OpenAI settings (only needed if provider = openai)
openai_api_key: "sk-..."
openai_embedding_model: "text-embedding-3-small"

# Retrieval settings
max_results: 5
min_score: 0.5
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,local]"

# Run tests
python -m pytest tests/ -v

# Type checking
mypy memoryforge/

# Linting
ruff check memoryforge/
```

---

## License

MIT License - see [LICENSE](LICENSE)
