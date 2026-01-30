"""
MCP Server for MemoryForge.

Provides MCP-compliant tools for AI assistants to interact with memory:
- store_memory: Create and optionally confirm a memory
- search_memory: Semantic search for relevant memories
- list_memory: List all stored memories
- delete_memory: Remove a memory by ID
"""

import asyncio
import logging
from typing import Any, Optional
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

from memoryforge.config import Config, EmbeddingProvider
from memoryforge.models import MemoryType, MemorySource
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.embedding_factory import create_embedding_service, get_embedding_dimension
from memoryforge.core.memory_manager import MemoryManager
from memoryforge.core.retrieval import RetrievalEngine
from memoryforge.core.validation import ValidationError

logger = logging.getLogger(__name__)


def create_mcp_server(config: Config, project_id: UUID) -> Server:
    """
    Create and configure the MCP server.
    
    Args:
        config: MemoryForge configuration
        project_id: Active project ID
        
    Returns:
        Configured MCP Server instance
    """
    # Get embedding dimension based on provider
    embedding_dim = get_embedding_dimension(
        config.embedding_provider,
        config.local_embedding_model if config.embedding_provider == EmbeddingProvider.LOCAL else config.openai_embedding_model
    )
    
    # Initialize storage and services
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    qdrant_store = QdrantStore(config.qdrant_path, embedding_dimension=embedding_dim)
    embedding_service = create_embedding_service(config)
    
    # Initialize core components
    memory_manager = MemoryManager(
        sqlite_db=sqlite_db,
        qdrant_store=qdrant_store,
        embedding_service=embedding_service,
        project_id=project_id,
    )
    
    retrieval_engine = RetrievalEngine(
        sqlite_db=sqlite_db,
        qdrant_store=qdrant_store,
        embedding_service=embedding_service,
        project_id=project_id,
        max_results=config.max_results,
        min_score=config.min_score,
    )
    
    # Create MCP server
    server = Server("memoryforge")
    
    # Define available tools
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="store_memory",
                description=(
                    "Store a new memory about the project. "
                    "Memories can be: tech stack, architecture decisions, constraints, "
                    "code conventions, or general notes. "
                    "Set confirm=true to make it immediately available for retrieval."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store (e.g., 'We use FastAPI with Pydantic v2')",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["stack", "decision", "constraint", "convention", "note"],
                            "description": "Type of memory",
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "If true, immediately confirm and index the memory",
                            "default": False,
                        },
                    },
                    "required": ["content", "type"],
                },
            ),
            Tool(
                name="search_memory",
                description=(
                    "Search for relevant memories using semantic similarity. "
                    "Returns the most relevant memories with explanations of why they were selected."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'What backend framework do we use?')",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["stack", "decision", "constraint", "convention", "note"],
                            "description": "Optional: filter by memory type",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5, max: 10)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_memory",
                description=(
                    "List all stored memories for the current project. "
                    "Useful for seeing what context is available."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["stack", "decision", "constraint", "convention", "note"],
                            "description": "Optional: filter by memory type",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 20)",
                            "default": 20,
                        },
                    },
                },
            ),
            Tool(
                name="delete_memory",
                description="Delete a memory by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The UUID of the memory to delete",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "store_memory":
                return await _handle_store_memory(memory_manager, arguments)
            elif name == "search_memory":
                return await _handle_search_memory(retrieval_engine, arguments)
            elif name == "list_memory":
                return await _handle_list_memory(memory_manager, arguments)
            elif name == "delete_memory":
                return await _handle_delete_memory(memory_manager, arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e.message}")]
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    return server


async def _handle_store_memory(
    memory_manager: MemoryManager,
    arguments: dict[str, Any],
) -> list[TextContent]:
    """Handle store_memory tool call."""
    content = arguments.get("content", "")
    memory_type_str = arguments.get("type", "note")
    confirm = arguments.get("confirm", False)
    
    # Parse memory type
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        return [TextContent(
            type="text",
            text=f"Invalid memory type: {memory_type_str}. Use: stack, decision, constraint, convention, or note",
        )]
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    memory = await loop.run_in_executor(
        None,
        lambda: memory_manager.create_memory(
            content=content,
            memory_type=memory_type,
            source=MemorySource.CHAT,
            auto_confirm=confirm,
        ),
    )
    
    status = "confirmed and indexed" if confirm else "stored (pending confirmation)"
    return [TextContent(
        type="text",
        text=f"Memory {status}.\nID: {memory.id}\nType: {memory.type.value}\nContent: {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}",
    )]


async def _handle_search_memory(
    retrieval_engine: RetrievalEngine,
    arguments: dict[str, Any],
) -> list[TextContent]:
    """Handle search_memory tool call."""
    query = arguments.get("query", "")
    memory_type_str = arguments.get("type")
    limit = min(arguments.get("limit", 5), 10)
    
    # Parse memory type if provided
    memory_type = None
    if memory_type_str:
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            pass
    
    # Run search in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: retrieval_engine.search(
            query=query,
            memory_type=memory_type,
            limit=limit,
        ),
    )
    
    if not results:
        return [TextContent(type="text", text="No relevant memories found.")]
    
    # Format results
    lines = [f"Found {len(results)} relevant memories:\n"]
    for i, result in enumerate(results, 1):
        memory = result.memory
        lines.append(f"{i}. {result.explanation}")
        lines.append(f"   Content: {memory.content[:200]}{'...' if len(memory.content) > 200 else ''}")
        lines.append("")
    
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_list_memory(
    memory_manager: MemoryManager,
    arguments: dict[str, Any],
) -> list[TextContent]:
    """Handle list_memory tool call."""
    memory_type_str = arguments.get("type")
    limit = min(arguments.get("limit", 20), 100)
    
    # Parse memory type if provided
    memory_type = None
    if memory_type_str:
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            pass
    
    # Get memories
    loop = asyncio.get_event_loop()
    memories = await loop.run_in_executor(
        None,
        lambda: memory_manager.list_memories(
            confirmed_only=True,
            memory_type=memory_type,
            limit=limit,
        ),
    )
    
    if not memories:
        return [TextContent(type="text", text="No memories stored yet.")]
    
    # Format list
    lines = [f"Stored memories ({len(memories)}):\n"]
    for memory in memories:
        type_label = f"[{memory.type.value}]"
        date_str = memory.created_at.strftime("%b %d")
        lines.append(f"• {type_label} {memory.content[:80]}{'...' if len(memory.content) > 80 else ''}")
        lines.append(f"  ID: {memory.id} | {date_str}")
        lines.append("")
    
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_delete_memory(
    memory_manager: MemoryManager,
    arguments: dict[str, Any],
) -> list[TextContent]:
    """Handle delete_memory tool call."""
    memory_id_str = arguments.get("memory_id", "")
    
    try:
        memory_id = UUID(memory_id_str)
    except ValueError:
        return [TextContent(type="text", text=f"Invalid memory ID: {memory_id_str}")]
    
    # Delete memory
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(
        None,
        lambda: memory_manager.delete_memory(memory_id),
    )
    
    if success:
        return [TextContent(type="text", text=f"Memory {memory_id} deleted.")]
    else:
        return [TextContent(type="text", text=f"Memory {memory_id} not found.")]


async def run_mcp_server(config: Config, project_id: UUID) -> None:
    """Run the MCP server with stdio transport."""
    server = create_mcp_server(config, project_id)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
