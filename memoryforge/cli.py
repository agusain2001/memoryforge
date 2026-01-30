"""
CLI for MemoryForge.

Commands:
- init: Initialize MemoryForge in current directory
- add: Add a memory manually
- list: List stored memories
- delete: Delete a memory by ID
- confirm: Confirm a pending memory
- search: Search for memories
- serve: Start the MCP server
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from memoryforge.config import Config, EmbeddingProvider
from memoryforge.models import MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore
from memoryforge.core.embedding_factory import create_embedding_service, get_embedding_dimension
from memoryforge.core.memory_manager import MemoryManager
from memoryforge.core.retrieval import RetrievalEngine
from memoryforge.mcp.server import run_mcp_server

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("memoryforge")


def get_project_id(config: Config) -> Optional[UUID]:
    """Get the active project ID from the database."""
    try:
        db = SQLiteDatabase(config.sqlite_path)
        project = db.get_project_by_name(config.project_name)
        return project.id if project else None
    except Exception:
        return None


def ensure_initialized(config: Config) -> tuple[SQLiteDatabase, QdrantStore, UUID]:
    """Ensure MemoryForge is initialized and return storage components."""
    if not config.sqlite_path.exists():
        console.print("[red]MemoryForge not initialized. Run 'memoryforge init' first.[/red]")
        sys.exit(1)
    
    db = SQLiteDatabase(config.sqlite_path)
    project = db.get_project_by_name(config.project_name)
    
    if not project:
        console.print("[red]Project not found. Run 'memoryforge init' first.[/red]")
        sys.exit(1)
    
    # Get embedding dimension based on provider
    embedding_dim = get_embedding_dimension(
        config.embedding_provider,
        config.local_embedding_model if config.embedding_provider == EmbeddingProvider.LOCAL else config.openai_embedding_model
    )
    
    qdrant = QdrantStore(config.qdrant_path, embedding_dimension=embedding_dim)
    
    return db, qdrant, project.id


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.pass_context
def main(ctx: click.Context, config: Optional[str]) -> None:
    """MemoryForge - Local-first memory for AI coding assistants."""
    ctx.ensure_object(dict)
    
    if config:
        ctx.obj["config"] = Config.load(Path(config))
    else:
        ctx.obj["config"] = Config.load()


@main.command()
@click.option("--name", "-n", default=None, help="Project name")
@click.option("--api-key", "-k", default=None, help="OpenAI API key (optional)")
@click.option(
    "--provider", "-p",
    type=click.Choice(["local", "openai"]),
    default="local",
    help="Embedding provider (default: local for free embeddings)",
)
@click.pass_context
def init(ctx: click.Context, name: Optional[str], api_key: Optional[str], provider: str) -> None:
    """Initialize MemoryForge for the current project."""
    config: Config = ctx.obj["config"]
    
    console.print(Panel.fit(
        "[bold blue]MemoryForge v0.9.0[/bold blue]\n"
        "Local-first memory for AI coding assistants",
        border_style="blue",
    ))
    
    # Get project name
    if not name:
        cwd = Path.cwd()
        default_name = cwd.name
        name = Prompt.ask("Project name", default=default_name)
    
    # Set embedding provider
    config.embedding_provider = EmbeddingProvider(provider)
    
    # Get API key if using OpenAI
    if provider == "openai":
        if not api_key:
            if config.openai_api_key:
                api_key = config.openai_api_key
                console.print(f"[dim]Using existing OpenAI API key[/dim]")
            else:
                api_key = Prompt.ask("OpenAI API key")
        config.openai_api_key = api_key
    else:
        console.print(f"[green]Using local embeddings (free, no API key needed)[/green]")
        console.print(f"[dim]Model: {config.local_embedding_model}[/dim]")
    
    # Update config
    config.project_name = name
    config.project_root = str(Path.cwd())
    
    # Create directories
    config.ensure_directories()
    
    # Initialize database
    db = SQLiteDatabase(config.sqlite_path)
    
    # Check if project already exists
    existing = db.get_project_by_name(name)
    if existing:
        console.print(f"[yellow]Project '{name}' already exists.[/yellow]")
        if not Confirm.ask("Reinitialize?"):
            return
    else:
        # Create project
        project = Project(
            name=name,
            root_path=str(Path.cwd()),
        )
        db.create_project(project)
    
    # Initialize Qdrant with correct dimension
    embedding_dim = get_embedding_dimension(
        config.embedding_provider,
        config.local_embedding_model if config.embedding_provider == EmbeddingProvider.LOCAL else config.openai_embedding_model
    )
    QdrantStore(config.qdrant_path, embedding_dimension=embedding_dim)
    
    # Save config
    config.save()
    
    console.print(f"\n[green]✓ MemoryForge initialized for '{name}'[/green]")
    console.print(f"[dim]Embedding provider: {config.embedding_provider.value}[/dim]")
    console.print(f"[dim]Config: {config.storage_path / 'config.yaml'}[/dim]")
    console.print(f"[dim]Database: {config.sqlite_path}[/dim]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Add memories: [cyan]memoryforge add 'We use FastAPI'[/cyan]")
    console.print("  2. Start server: [cyan]memoryforge serve[/cyan]")


@main.command()
@click.argument("content")
@click.option(
    "--type", "-t",
    type=click.Choice(["stack", "decision", "constraint", "convention", "note"]),
    default="note",
    help="Memory type",
)
@click.option("--confirm/--no-confirm", default=True, help="Immediately confirm the memory")
@click.pass_context
def add(ctx: click.Context, content: str, type: str, confirm: bool) -> None:
    """Add a new memory."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    # Initialize embedding service
    try:
        embedding_service = create_embedding_service(config)
    except (ValueError, ImportError) as e:
        console.print(f"[red]Error: {e}[/red]")
        if config.embedding_provider == EmbeddingProvider.LOCAL:
            console.print("[dim]Install local embeddings: pip install memoryforge[local][/dim]")
        else:
            console.print("[dim]Set your OpenAI API key with 'memoryforge init --provider openai'[/dim]")
        sys.exit(1)
    
    # Create memory manager
    memory_manager = MemoryManager(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
    )
    
    # Create memory
    memory_type = MemoryType(type)
    
    with console.status("[bold green]Generating embedding..."):
        memory = memory_manager.create_memory(
            content=content,
            memory_type=memory_type,
            source=MemorySource.MANUAL,
            auto_confirm=confirm,
        )
    
    status = "[green]confirmed[/green]" if confirm else "[yellow]pending[/yellow]"
    console.print(f"\n[bold]Memory added ({status})[/bold]")
    console.print(f"  ID: {memory.id}")
    console.print(f"  Type: {memory.type.value}")
    console.print(f"  Content: {content[:100]}{'...' if len(content) > 100 else ''}")


@main.command("list")
@click.option(
    "--type", "-t",
    type=click.Choice(["stack", "decision", "constraint", "convention", "note"]),
    default=None,
    help="Filter by memory type",
)
@click.option("--all", "-a", "show_all", is_flag=True, help="Include unconfirmed memories")
@click.option("--limit", "-l", default=20, help="Maximum number of results")
@click.pass_context
def list_memories(ctx: click.Context, type: Optional[str], show_all: bool, limit: int) -> None:
    """List stored memories."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    # Parse memory type
    memory_type = MemoryType(type) if type else None
    
    # Get memories
    memories = db.list_memories(
        project_id=project_id,
        confirmed_only=not show_all,
        memory_type=memory_type,
        limit=limit,
    )
    
    if not memories:
        console.print("[dim]No memories stored yet.[/dim]")
        return
    
    # Create table
    table = Table(title=f"Memories ({len(memories)})")
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Content", width=50)
    table.add_column("ID", style="dim", width=36)
    table.add_column("Status", width=10)
    
    for memory in memories:
        status = "✓" if memory.confirmed else "⏳"
        content = memory.content[:47] + "..." if len(memory.content) > 50 else memory.content
        table.add_row(
            memory.type.value,
            content.replace("\n", " "),
            str(memory.id),
            status,
        )
    
    console.print(table)


@main.command()
@click.argument("memory_id")
@click.pass_context
def delete(ctx: click.Context, memory_id: str) -> None:
    """Delete a memory by ID."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    try:
        uuid = UUID(memory_id)
    except ValueError:
        console.print(f"[red]Invalid memory ID: {memory_id}[/red]")
        sys.exit(1)
    
    # Check if memory exists
    memory = db.get_memory(uuid)
    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        sys.exit(1)
    
    # Confirm deletion
    console.print(f"Memory: {memory.content[:100]}...")
    if not Confirm.ask("Delete this memory?"):
        return
    
    # Delete from Qdrant if confirmed
    if memory.confirmed:
        qdrant.delete(uuid)
    
    # Delete from SQLite
    db.delete_memory(uuid)
    
    console.print(f"[green]✓ Memory deleted[/green]")


@main.command("confirm")
@click.argument("memory_id")
@click.pass_context
def confirm_memory(ctx: click.Context, memory_id: str) -> None:
    """Confirm a pending memory."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    try:
        uuid = UUID(memory_id)
    except ValueError:
        console.print(f"[red]Invalid memory ID: {memory_id}[/red]")
        sys.exit(1)
    
    # Check if memory exists
    memory = db.get_memory(uuid)
    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        sys.exit(1)
    
    if memory.confirmed:
        console.print("[yellow]Memory is already confirmed.[/yellow]")
        return
    
    # Initialize embedding service
    try:
        embedding_service = create_embedding_service(config)
    except (ValueError, ImportError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    # Create memory manager
    memory_manager = MemoryManager(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
    )
    
    # Confirm memory
    with console.status("[bold green]Generating embedding..."):
        success = memory_manager.confirm_memory(uuid)
    
    if success:
        console.print(f"[green]✓ Memory confirmed and indexed[/green]")
    else:
        console.print(f"[red]Failed to confirm memory[/red]")


@main.command()
@click.argument("query")
@click.option(
    "--type", "-t",
    type=click.Choice(["stack", "decision", "constraint", "convention", "note"]),
    default=None,
    help="Filter by memory type",
)
@click.option("--limit", "-l", default=5, help="Maximum number of results")
@click.pass_context
def search(ctx: click.Context, query: str, type: Optional[str], limit: int) -> None:
    """Search for relevant memories."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    # Initialize embedding service
    try:
        embedding_service = create_embedding_service(config)
    except (ValueError, ImportError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    # Create retrieval engine
    retrieval = RetrievalEngine(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
        max_results=config.max_results,
        min_score=config.min_score,
    )
    
    # Parse memory type
    memory_type = MemoryType(type) if type else None
    
    # Search
    with console.status("[bold green]Searching..."):
        results = retrieval.search(
            query=query,
            memory_type=memory_type,
            limit=limit,
        )
    
    if not results:
        console.print("[dim]No relevant memories found.[/dim]")
        return
    
    console.print(f"\n[bold]Found {len(results)} relevant memories:[/bold]\n")
    
    for i, result in enumerate(results, 1):
        memory = result.memory
        console.print(f"[cyan]{i}. {result.explanation}[/cyan]")
        console.print(f"   {memory.content[:200]}{'...' if len(memory.content) > 200 else ''}")
        console.print()


@main.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the MCP server."""
    config: Config = ctx.obj["config"]
    
    # Ensure initialized
    db, _, project_id = ensure_initialized(config)
    
    console.print("[bold blue]Starting MemoryForge MCP Server...[/bold blue]")
    console.print(f"[dim]Project: {config.project_name}[/dim]")
    console.print(f"[dim]Embedding: {config.embedding_provider.value}[/dim]")
    console.print(f"[dim]Storage: {config.storage_path}[/dim]")
    console.print()
    console.print("[green]Server running. Press Ctrl+C to stop.[/green]")
    
    # Run the async server
    try:
        asyncio.run(run_mcp_server(config, project_id))
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    main()
