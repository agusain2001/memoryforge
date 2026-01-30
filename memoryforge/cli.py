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

v2 Commands:
- project create: Create a new project
- project switch: Switch to a different project
- project list: List all projects
- project delete: Delete a project
- status: Show current project status
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
from memoryforge.core.project_router import ProjectRouter
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
    
    # v2: Use active_project_id from config, fallback to project_name
    project = None
    if config.active_project_id:
        try:
            project = db.get_project(UUID(config.active_project_id))
        except (ValueError, TypeError):
            pass
    
    if not project:
        project = db.get_project_by_name(config.project_name)
    
    if not project:
        console.print("[red]Project not found. Run 'memoryforge init' first.[/red]")
        sys.exit(1)
    
    # Get embedding dimension based on provider
    embedding_dim = get_embedding_dimension(
        config.embedding_provider,
        config.local_embedding_model if config.embedding_provider == EmbeddingProvider.LOCAL else config.openai_embedding_model
    )
    
    # v2: Per-project Qdrant collection
    qdrant = QdrantStore(
        config.qdrant_path,
        project_id=project.id,
        embedding_dimension=embedding_dim,
    )
    
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
    
    # Use ProjectRouter for v2
    router = ProjectRouter(db, config)
    
    # Check if project already exists
    existing = db.get_project_by_name(name)
    if existing:
        console.print(f"[yellow]Project '{name}' already exists.[/yellow]")
        if not Confirm.ask("Reinitialize?"):
            return
        project = existing
        # Set as active project
        config.active_project_id = str(project.id)
    else:
        # Create project using router
        project = router.create_project(name, str(Path.cwd()), set_active=True)
    
    # Initialize Qdrant with correct dimension and project scope
    embedding_dim = get_embedding_dimension(
        config.embedding_provider,
        config.local_embedding_model if config.embedding_provider == EmbeddingProvider.LOCAL else config.openai_embedding_model
    )
    QdrantStore(config.qdrant_path, project_id=project.id, embedding_dimension=embedding_dim)
    
    # Save config
    config.save()
    
    console.print(f"\n[green]✓ MemoryForge initialized for '{name}'[/green]")
    console.print(f"[dim]Project ID: {project.id}[/dim]")
    console.print(f"[dim]Embedding provider: {config.embedding_provider.value}[/dim]")
    console.print(f"[dim]Config: {config.storage_path / 'config.yaml'}[/dim]")
    console.print(f"[dim]Database: {config.sqlite_path}[/dim]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Add memories: [cyan]memoryforge add 'We use FastAPI'[/cyan]")
    console.print("  2. Start server: [cyan]memoryforge serve[/cyan]")
    console.print("  3. View status: [cyan]memoryforge status[/cyan]")


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
@click.option("--limit", "-l", default=20, help="Number of memories to show")
@click.pass_context
def timeline(ctx: click.Context, limit: int) -> None:
    """Show memory timeline (chronological view)."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    memories = db.get_recent_memories(project_id, limit=limit)
    
    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        return
        
    console.print(f"\n[bold]Memory Timeline ({len(memories)} most recent):[/bold]\n")
    
    for memory in memories:
        date_str = memory.created_at.strftime("%Y-%m-%d %H:%M")
        
        # Color code types
        type_color = "white"
        if memory.type == MemoryType.DECISION:
            type_color = "red"
        elif memory.type == MemoryType.STACK:
            type_color = "cyan"
        elif memory.type == MemoryType.CONSTRAINT:
            type_color = "yellow"
            
        console.print(f"[{type_color}][{date_str}] {memory.type.value.upper()}[/{type_color}]")
        console.print(f"[dim]ID: {memory.id}[/dim]")
        
        # Truncate content for display
        content = memory.content
        if len(content) > 100:
            content = content[:100] + "..."
        console.print(f"  {content}")
        console.print("")


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


# ============================================================================
# v2 Commands: Project Management
# ============================================================================

@main.group()
@click.pass_context
def project(ctx: click.Context) -> None:
    """Manage projects (v2)."""
    pass


@project.command("create")
@click.argument("name")
@click.option("--path", "-p", default=".", help="Project root path")
@click.pass_context
def project_create(ctx: click.Context, name: str, path: str) -> None:
    """Create a new project."""
    config: Config = ctx.obj["config"]
    
    # Ensure storage exists
    config.ensure_directories()
    db = SQLiteDatabase(config.sqlite_path)
    router = ProjectRouter(db, config)
    
    try:
        project = router.create_project(name, path, set_active=True)
        console.print(f"[green]✓ Created project '{name}'[/green]")
        console.print(f"  ID: {project.id}")
        console.print(f"  Path: {project.root_path}")
        console.print(f"  [dim]Now active[/dim]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@project.command("switch")
@click.argument("name_or_id")
@click.pass_context
def project_switch(ctx: click.Context, name_or_id: str) -> None:
    """Switch to a different project."""
    config: Config = ctx.obj["config"]
    
    if not config.sqlite_path.exists():
        console.print("[red]MemoryForge not initialized. Run 'memoryforge init' first.[/red]")
        sys.exit(1)
    
    db = SQLiteDatabase(config.sqlite_path)
    router = ProjectRouter(db, config)
    
    try:
        # Try as UUID first
        try:
            project_id = UUID(name_or_id)
            router.switch_project(project_id)
            project = router.get_project(project_id)
        except ValueError:
            # Try as name
            router.switch_project_by_name(name_or_id)
            project = router.get_project_by_name(name_or_id)
        
        console.print(f"[green]✓ Switched to project '{project.name}'[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@project.command("list")
@click.pass_context
def project_list(ctx: click.Context) -> None:
    """List all projects."""
    config: Config = ctx.obj["config"]
    
    if not config.sqlite_path.exists():
        console.print("[dim]No projects found. Run 'memoryforge init' first.[/dim]")
        return
    
    db = SQLiteDatabase(config.sqlite_path)
    router = ProjectRouter(db, config)
    
    projects = router.list_projects()
    
    if not projects:
        console.print("[dim]No projects found.[/dim]")
        return
    
    active_id = config.active_project_id
    
    table = Table(title="Projects")
    table.add_column("Active", width=6)
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Memories", justify="right")
    table.add_column("ID", style="dim")
    
    for proj in projects:
        is_active = "→" if str(proj.id) == active_id else ""
        memory_count = db.get_memory_count(proj.id, confirmed_only=True)
        table.add_row(
            is_active,
            proj.name,
            proj.root_path[:40] + "..." if len(proj.root_path) > 40 else proj.root_path,
            str(memory_count),
            str(proj.id)[:8],
        )
    
    console.print(table)


@project.command("delete")
@click.argument("name_or_id")
@click.pass_context
def project_delete(ctx: click.Context, name_or_id: str) -> None:
    """Delete a project (blocked if has memories)."""
    config: Config = ctx.obj["config"]
    
    if not config.sqlite_path.exists():
        console.print("[red]MemoryForge not initialized.[/red]")
        sys.exit(1)
    
    db = SQLiteDatabase(config.sqlite_path)
    router = ProjectRouter(db, config)
    
    # Find project
    try:
        try:
            project_id = UUID(name_or_id)
            project = router.get_project(project_id)
        except ValueError:
            project = router.get_project_by_name(name_or_id)
        
        if not project:
            console.print(f"[red]Project not found: {name_or_id}[/red]")
            sys.exit(1)
        
        # Show info and confirm
        memory_count = db.get_memory_count(project.id, confirmed_only=False)
        console.print(f"Project: {project.name}")
        console.print(f"Memories: {memory_count}")
        
        if memory_count > 0:
            console.print(f"[red]Cannot delete project with {memory_count} memories.[/red]")
            console.print("[dim]Delete all memories first.[/dim]")
            sys.exit(1)
        
        if not Confirm.ask("Delete this project?"):
            return
        
        router.delete_project(project.id)
        console.print(f"[green]✓ Deleted project '{project.name}'[/green]")
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current project status (v2)."""
    config: Config = ctx.obj["config"]
    
    if not config.sqlite_path.exists():
        console.print("[yellow]MemoryForge not initialized.[/yellow]")
        console.print("Run: [cyan]memoryforge init[/cyan]")
        return
    
    db = SQLiteDatabase(config.sqlite_path)
    router = ProjectRouter(db, config)
    
    status_info = router.get_project_status()
    
    if not status_info.get("active"):
        console.print(f"[yellow]{status_info.get('message', 'No active project')}[/yellow]")
        return
    
    console.print(Panel.fit(
        f"[bold]{status_info['project_name']}[/bold]\n"
        f"[dim]ID: {status_info['project_id'][:8]}...[/dim]",
        title="Active Project",
        border_style="blue",
    ))
    
    console.print(f"\n[bold]Memories:[/bold] {status_info['memory_count']} confirmed")
    if status_info['pending_count'] > 0:
        console.print(f"[yellow]Pending:[/yellow] {status_info['pending_count']}")
    
    console.print(f"\n[bold]Path:[/bold] {status_info['root_path']}")
    console.print(f"[bold]Created:[/bold] {status_info['created_at'][:10]}")
    console.print(f"[bold]Embedding:[/bold] {config.embedding_provider.value}")


# ============================================================================
# v2 Commands: Git Integration (Phase 2)
# ============================================================================

@main.group()
@click.pass_context
def git(ctx: click.Context) -> None:
    """Git integration commands (v2)."""
    pass


@git.command("status")
@click.pass_context
def git_status(ctx: click.Context) -> None:
    """Show git integration status."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    from memoryforge.core.git_integration import GitIntegration
    
    git_int = GitIntegration(db, config, project_id)
    status = git_int.get_status()
    
    if not status.get("enabled"):
        console.print("[yellow]Git integration is disabled.[/yellow]")
        console.print("Enable with: [cyan]Set enable_git_integration: true in config.yaml[/cyan]")
        return
    
    if not status.get("available"):
        console.print(f"[red]Git not available: {status.get('message', 'Unknown error')}[/red]")
        return
    
    console.print(Panel.fit(
        f"[bold]Branch:[/bold] {status.get('branch', 'unknown')}\n"
        f"[bold]Commit:[/bold] {status.get('commit', 'unknown')}\n"
        f"[bold]Clean:[/bold] {'✓' if status.get('is_clean') else '✗ (uncommitted changes)'}",
        title="Git Status",
        border_style="green",
    ))
    
    if status.get("remote"):
        console.print(f"[dim]Remote: {status['remote']}[/dim]")


@git.command("sync")
@click.option("--limit", "-l", default=20, help="Maximum commits to scan")
@click.pass_context
def git_sync(ctx: click.Context, limit: int) -> None:
    """Scan for architectural commits and suggest memory links."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    from memoryforge.core.git_integration import GitIntegration
    
    git_int = GitIntegration(db, config, project_id)
    
    if not git_int.is_available():
        status = git_int.get_status()
        if not status.get("enabled"):
            console.print("[yellow]Git integration is disabled.[/yellow]")
            console.print("Enable in config.yaml: [cyan]enable_git_integration: true[/cyan]")
        else:
            console.print(f"[red]Git not available: {status.get('message')}[/red]")
        return
    
    with console.status("[bold green]Scanning commits..."):
        suggestions = git_int.sync_architectural_commits()
    
    if not suggestions:
        console.print("[dim]No architectural commits found.[/dim]")
        return
    
    # Show commits without memories
    unlinked = [s for s in suggestions if not s["has_memory"]]
    linked = [s for s in suggestions if s["has_memory"]]
    
    if unlinked:
        console.print(f"\n[bold yellow]Commits without memories ({len(unlinked)}):[/bold yellow]\n")
        
        table = Table()
        table.add_column("SHA", style="cyan", width=8)
        table.add_column("Message", width=40)
        table.add_column("Reason", style="yellow", width=20)
        table.add_column("Files", justify="right", width=6)
        
        for s in unlinked[:limit]:
            table.add_row(
                s["commit"]["short_sha"],
                s["commit"]["message"][:38] + "..." if len(s["commit"]["message"]) > 40 else s["commit"]["message"],
                s["reason"],
                str(s["commit"]["files_changed"]),
            )
        
        console.print(table)
        console.print(f"\n[dim]Tip: Create a memory for these commits with:[/dim]")
        console.print("[cyan]memoryforge add 'Description' --type decision[/cyan]")
        console.print("[cyan]memoryforge git link <memory_id> <commit_sha>[/cyan]")
    
    if linked:
        console.print(f"\n[bold green]Commits with memories ({len(linked)}):[/bold green]")
        for s in linked[:5]:
            console.print(f"  ✓ {s['commit']['short_sha']} - {s['commit']['message'][:40]} ({s['memory_count']} memories)")


@git.command("link")
@click.argument("memory_id")
@click.argument("commit_sha")
@click.pass_context
def git_link(ctx: click.Context, memory_id: str, commit_sha: str) -> None:
    """Link a memory to a git commit."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    from memoryforge.core.git_integration import GitIntegration
    from memoryforge.models import LinkType
    
    try:
        uuid = UUID(memory_id)
    except ValueError:
        console.print(f"[red]Invalid memory ID: {memory_id}[/red]")
        sys.exit(1)
    
    # Verify memory exists
    memory = db.get_memory(uuid)
    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        sys.exit(1)
    
    git_int = GitIntegration(db, config, project_id)
    
    if not git_int.is_available():
        console.print("[red]Git integration not available.[/red]")
        sys.exit(1)
    
    # Get commit info for confirmation
    commit_info = git_int.get_commit_info(commit_sha)
    if not commit_info:
        console.print(f"[red]Commit not found: {commit_sha}[/red]")
        sys.exit(1)
    
    console.print(f"Memory: {memory.content[:60]}...")
    console.print(f"Commit: {commit_info.short_sha} - {commit_info.first_line}")
    
    if not Confirm.ask("Link these?"):
        return
    
    success = git_int.link_memory_to_commit(uuid, commit_sha, LinkType.MENTIONED_IN)
    
    if success:
        console.print(f"[green]✓ Linked memory to commit {commit_info.short_sha}[/green]")
    else:
        console.print("[red]Failed to create link.[/red]")


@git.command("activity")
@click.option("--days", "-d", default=7, help="Number of days to look back")
@click.pass_context
def git_activity(ctx: click.Context, days: int) -> None:
    """Show recent git activity summary."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    from memoryforge.core.git_integration import GitIntegration
    
    git_int = GitIntegration(db, config, project_id)
    
    if not git_int.is_available():
        console.print("[red]Git integration not available.[/red]")
        return
    
    activity = git_int.get_recent_activity(days)
    
    if not activity.get("available"):
        console.print("[red]Could not fetch activity.[/red]")
        return
    
    console.print(Panel.fit(
        f"[bold]Commits:[/bold] {activity['commit_count']}\n"
        f"[bold]Files changed:[/bold] {activity['files_changed']}\n"
        f"[bold]Authors:[/bold] {', '.join(activity['authors'][:5])}",
        title=f"Git Activity (last {days} days)",
        border_style="blue",
    ))


# ============================================================================
# v2 Commands: Memory Consolidation (Phase 3)
# ============================================================================

@main.group()
@click.pass_context
def consolidate(ctx: click.Context) -> None:
    """Memory consolidation commands (v2)."""
    pass


@consolidate.command("suggest")
@click.option("--limit", "-l", default=5, help="Maximum suggestions")
@click.pass_context
def consolidate_suggest(ctx: click.Context, limit: int) -> None:
    """Show consolidation suggestions."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    from memoryforge.core.memory_consolidator import MemoryConsolidator
    from memoryforge.core.embedding_factory import create_embedding_service
    
    embedding_service = create_embedding_service(config)
    consolidator = MemoryConsolidator(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
        threshold=config.consolidation_threshold,
    )
    
    with console.status("[bold green]Finding similar memories..."):
        suggestions = consolidator.suggest_consolidations(max_suggestions=limit)
    
    if not suggestions:
        console.print("[dim]No consolidation suggestions found.[/dim]")
        console.print(f"[dim]Threshold: {config.consolidation_threshold}[/dim]")
        return
    
    console.print(f"\n[bold]Consolidation Suggestions ({len(suggestions)}):[/bold]\n")
    
    for i, suggestion in enumerate(suggestions, 1):
        console.print(Panel(
            f"[bold]Similarity:[/bold] {suggestion.similarity_score:.2%}\n\n"
            f"[bold]Memory 1:[/bold] {suggestion.source_memories[0].content[:100]}...\n"
            f"[dim]ID: {suggestion.source_memories[0].id}[/dim]\n\n"
            f"[bold]Memory 2:[/bold] {suggestion.source_memories[1].content[:100]}...\n"
            f"[dim]ID: {suggestion.source_memories[1].id}[/dim]",
            title=f"Suggestion {i}",
            border_style="yellow",
        ))
    
    console.print("\n[dim]To consolidate, use:[/dim]")
    console.print("[cyan]memoryforge consolidate apply <id1> <id2> --content 'merged content'[/cyan]")


@consolidate.command("apply")
@click.argument("memory_ids", nargs=-1, required=True)
@click.option("--content", "-c", required=True, help="Content for consolidated memory")
@click.option("--type", "-t", "memory_type", help="Memory type (default: use first memory's type)")
@click.pass_context
def consolidate_apply(ctx: click.Context, memory_ids: tuple, content: str, memory_type: str) -> None:
    """Consolidate memories into a new one (archives originals)."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    if len(memory_ids) < 2:
        console.print("[red]Need at least 2 memory IDs to consolidate.[/red]")
        sys.exit(1)
    
    from memoryforge.core.memory_consolidator import MemoryConsolidator
    from memoryforge.core.embedding_factory import create_embedding_service
    
    # Parse memory IDs
    try:
        uuids = [UUID(mid) for mid in memory_ids]
    except ValueError as e:
        console.print(f"[red]Invalid memory ID: {e}[/red]")
        sys.exit(1)
    
    # Parse memory type if provided
    mtype = None
    if memory_type:
        try:
            mtype = MemoryType(memory_type)
        except ValueError:
            console.print(f"[red]Invalid memory type: {memory_type}[/red]")
            sys.exit(1)
    
    embedding_service = create_embedding_service(config)
    consolidator = MemoryConsolidator(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
        threshold=config.consolidation_threshold,
    )
    
    # Show what will be consolidated
    console.print(f"\n[bold]Consolidating {len(uuids)} memories:[/bold]")
    for uid in uuids:
        memory = db.get_memory(uid)
        if memory:
            console.print(f"  • {memory.content[:60]}...")
    
    console.print(f"\n[bold]New content:[/bold] {content[:100]}...")
    
    if not Confirm.ask("Proceed with consolidation?"):
        return
    
    try:
        result = consolidator.consolidate(
            source_ids=uuids,
            merged_content=content,
            memory_type=mtype,
        )
        
        console.print(f"\n[green]✓ Created consolidated memory[/green]")
        console.print(f"  ID: {result.consolidated_memory.id}")
        console.print(f"  Archived: {result.archived_count} memories")
        console.print(f"\n[dim]To undo: memoryforge consolidate rollback {result.consolidated_memory.id}[/dim]")
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@consolidate.command("rollback")
@click.argument("consolidated_id")
@click.pass_context
def consolidate_rollback(ctx: click.Context, consolidated_id: str) -> None:
    """Rollback a consolidation (restore archived memories)."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    from memoryforge.core.memory_consolidator import MemoryConsolidator
    from memoryforge.core.embedding_factory import create_embedding_service
    
    try:
        uid = UUID(consolidated_id)
    except ValueError:
        console.print(f"[red]Invalid memory ID: {consolidated_id}[/red]")
        sys.exit(1)
    
    embedding_service = create_embedding_service(config)
    consolidator = MemoryConsolidator(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
        threshold=config.consolidation_threshold,
    )
    
    if not Confirm.ask("Rollback this consolidation? (will restore archived memories)"):
        return
    
    try:
        restored = consolidator.rollback_consolidation(uid)
        
        console.print(f"\n[green]✓ Rollback complete[/green]")
        console.print(f"  Restored: {len(restored)} memories")
        for mem in restored:
            console.print(f"  • {mem.id}: {mem.content[:50]}...")
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@consolidate.command("stats")
@click.pass_context
def consolidate_stats(ctx: click.Context) -> None:
    """Show consolidation statistics."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    from memoryforge.core.memory_consolidator import MemoryConsolidator
    from memoryforge.core.embedding_factory import create_embedding_service
    
    embedding_service = create_embedding_service(config)
    consolidator = MemoryConsolidator(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
        threshold=config.consolidation_threshold,
    )
    
    with console.status("[bold green]Calculating stats..."):
        stats = consolidator.get_consolidation_stats()
    
    console.print(Panel.fit(
        f"[bold]Active memories:[/bold] {stats['active_memories']}\n"
        f"[bold]Archived memories:[/bold] {stats['archived_memories']}\n"
        f"[bold]Stale memories:[/bold] {stats['stale_memories']}\n"
        f"[bold]Similar pairs:[/bold] {stats['similar_pairs']}\n"
        f"[bold]Threshold:[/bold] {stats['threshold']:.0%}",
        title="Consolidation Stats",
        border_style="blue",
    ))


# ============================================================================
# v2 Commands: Staleness Management (Phase 3)
# ============================================================================

@main.group()
@click.pass_context
def stale(ctx: click.Context) -> None:
    """Staleness management commands (v2)."""
    pass


@stale.command("list")
@click.pass_context
def stale_list(ctx: click.Context) -> None:
    """List stale memories."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    stale_memories = db.get_stale_memories(project_id)
    
    if not stale_memories:
        console.print("[dim]No stale memories.[/dim]")
        return
    
    table = Table(title="Stale Memories")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Content", width=40)
    table.add_column("Reason", style="yellow", width=25)
    table.add_column("Type", width=10)
    
    for memory in stale_memories:
        table.add_row(
            str(memory.id)[:8] + "...",
            memory.content[:38] + "..." if len(memory.content) > 40 else memory.content,
            memory.stale_reason or "Unknown",
            memory.type.value,
        )
    
    console.print(table)
    console.print(f"\n[dim]Clear with: memoryforge stale clear <id>[/dim]")


@stale.command("mark")
@click.argument("memory_id")
@click.option("--reason", "-r", required=True, help="Reason for marking stale")
@click.pass_context
def stale_mark(ctx: click.Context, memory_id: str, reason: str) -> None:
    """Mark a memory as stale."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    try:
        uid = UUID(memory_id)
    except ValueError:
        console.print(f"[red]Invalid memory ID: {memory_id}[/red]")
        sys.exit(1)
    
    memory = db.get_memory(uid)
    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        sys.exit(1)
    
    db.mark_stale(uid, reason)
    console.print(f"[green]✓ Marked memory as stale[/green]")
    console.print(f"  Reason: {reason}")


@stale.command("clear")
@click.argument("memory_id")
@click.pass_context
def stale_clear(ctx: click.Context, memory_id: str) -> None:
    """Clear the stale flag from a memory."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    try:
        uid = UUID(memory_id)
    except ValueError:
        console.print(f"[red]Invalid memory ID: {memory_id}[/red]")
        sys.exit(1)
    
    memory = db.get_memory(uid)
    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        sys.exit(1)
    
    db.clear_stale(uid)
    console.print(f"[green]✓ Cleared stale flag[/green]")


@stale.command("unused")
@click.option("--days", "-d", default=30, help="Days since last access")
@click.pass_context
def stale_unused(ctx: click.Context, days: int) -> None:
    """Find memories not accessed recently."""
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    from memoryforge.core.memory_consolidator import MemoryConsolidator
    from memoryforge.core.embedding_factory import create_embedding_service
    
    embedding_service = create_embedding_service(config)
    consolidator = MemoryConsolidator(
        sqlite_db=db,
        qdrant_store=qdrant,
        embedding_service=embedding_service,
        project_id=project_id,
        threshold=config.consolidation_threshold,
    )
    
    unused = consolidator.find_unused_memories(days_unused=days)
    
    if not unused:
        console.print(f"[dim]No memories unused for {days}+ days.[/dim]")
        return
    
    console.print(f"\n[bold yellow]Unused Memories ({len(unused)}):[/bold yellow]\n")
    
    table = Table()
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Content", width=40)
    table.add_column("Created", width=12)
    table.add_column("Last Access", width=12)
    
    for memory in unused[:20]:
        last_access = memory.last_accessed.strftime("%Y-%m-%d") if memory.last_accessed else "Never"
        table.add_row(
            str(memory.id)[:8] + "...",
            memory.content[:38] + "..." if len(memory.content) > 40 else memory.content,
            memory.created_at.strftime("%Y-%m-%d"),
            last_access,
        )
    
    console.print(table)
    console.print(f"\n[dim]Consider marking as stale: memoryforge stale mark <id> --reason 'not used'[/dim]")


# ============================================================================
# v2 Commands: Migration (Phase 4)
# ============================================================================

@main.command("migrate")
@click.option("--rollback", is_flag=True, help="Rollback to the last backup")
@click.option("--backup-file", help="Specific backup file to restore (for rollback)")
@click.pass_context
def migrate(ctx: click.Context, rollback: bool, backup_file: str) -> None:
    """Migrate database to v2 (or rollback)."""
    config: Config = ctx.obj["config"]
    
    # Don't use ensure_initialized here as it might try to use the DB
    if not config.sqlite_path.exists() and not rollback:
        console.print("[yellow]Database not found. Initializing new v2 database...[/yellow]")
        # Standard init will handle it
        db = SQLiteDatabase(config.sqlite_path)
        console.print("[green]Created new database.[/green]")
        return

    from memoryforge.migrate import Migrator, MigrationError
    
    migrator = Migrator(config)
    
    if rollback:
        if backup_file:
            path = Path(backup_file)
        else:
            # Find latest backup
            backups = sorted(config.sqlite_path.parent.glob("memoryforge_v1_backup_*.sqlite"))
            if not backups:
                console.print("[red]No backups found.[/red]")
                sys.exit(1)
            path = backups[-1]
        
        console.print(f"Restoring from: {path}")
        if Confirm.ask("Are you sure? This will overwrite the current database."):
            try:
                migrator.restore_backup(path)
                console.print("[green]✓ Restore successful[/green]")
            except MigrationError as e:
                console.print(f"[red]Restore failed: {e}[/red]")
                sys.exit(1)
        return
    
    # Run migration
    try:
        if migrator.run_migration():
            console.print("[green]✓ Database is up to date (v2)[/green]")
        else:
            console.print("[red]Migration failed (restored from backup).[/red]")
            sys.exit(1)
    except MigrationError as e:
        console.print(f"[red]Migration error: {e}[/red]")
        sys.exit(1)



# ============================================================================
# v2.1 Commands: Team Sync
# ============================================================================

@main.group()
@click.pass_context
def sync(ctx: click.Context) -> None:
    """Team sync commands (v2.1)."""
    pass


@sync.command("init")
@click.option("--path", "-p", required=True, help="Path for sync storage (e.g. shared drive, git repo)")
@click.option("--key", "-k", help="Existing encryption key (if joining a team)")
@click.pass_context
def sync_init(ctx: click.Context, path: str, key: str) -> None:
    """Initialize team sync."""
    config: Config = ctx.obj["config"]
    
    from memoryforge.sync.encryption import EncryptionLayer
    
    # Verify path
    sync_path = Path(path).resolve()
    if not sync_path.exists():
        if Confirm.ask(f"Create sync directory at {sync_path}?"):
            sync_path.mkdir(parents=True, exist_ok=True)
        else:
            console.print("[red]Aborted.[/red]")
            return
    
    # Handle key
    if key:
        encryption_key = key
    else:
        # Generate new key
        try:
            encryption_key = EncryptionLayer.generate_key()
            console.print("[green]Generated new encryption key.[/green]")
        except ImportError:
            console.print("[red]cryptography package required.[/red]")
            console.print("Run: pip install memoryforge[sync]")
            return
    
    # Save config
    config.sync_path = sync_path
    config.sync_key = encryption_key
    config.sync_backend = "local"
    config.save()
    
    console.print(f"\n[green]✓ Sync initialized[/green]")
    console.print(f"  Path: {sync_path}")
    console.print(f"  Backend: {config.sync_backend}")
    
    if not key:
        console.print(f"\n[bold yellow]IMPORTANT: Save this key safely![/bold yellow]")
        console.print(f"[white on black] {encryption_key} [/white on black]")
        console.print("Share this key with your team members so they can syncing.")


@sync.command("push")
@click.option("--force", is_flag=True, help="Overwrite remote files even if newer")
@click.pass_context
def sync_push(ctx: click.Context, force: bool) -> None:
    """Export memories to sync backend."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    if not config.sync_path or not config.sync_key:
        console.print("[red]Sync not initialized.[/red]")
        console.print("Run: [cyan]memoryforge sync init[/cyan]")
        return
        
    from memoryforge.sync.encryption import EncryptionLayer, EncryptionError
    from memoryforge.sync.local_file_adapter import LocalFileAdapter
    from memoryforge.sync.manager import SyncManager
    
    try:
        encryption = EncryptionLayer(config.sync_key)
        adapter = LocalFileAdapter(config.sync_path)
        manager = SyncManager(db, adapter, encryption, project_id)
        
        with console.status("[bold green]Encrypting and pushing memories..."):
            count = manager.export_memories(force=force)
            
        console.print(f"[green]✓ Pushed {count} memories[/green]")
        
    except (ImportError, EncryptionError) as e:
        console.print(f"[red]Sync failed: {e}[/red]")
        if isinstance(e, ImportError):
             console.print("Try: pip install memoryforge[sync]")


@sync.command("pull")
@click.pass_context
def sync_pull(ctx: click.Context) -> None:
    """Import memories from sync backend."""
    config: Config = ctx.obj["config"]
    db, _, project_id = ensure_initialized(config)
    
    if not config.sync_path or not config.sync_key:
        console.print("[red]Sync not initialized.[/red]")
        console.print("Run: [cyan]memoryforge sync init[/cyan]")
        return
        
    from memoryforge.sync.encryption import EncryptionLayer, EncryptionError
    from memoryforge.sync.local_file_adapter import LocalFileAdapter
    from memoryforge.sync.manager import SyncManager
    
    try:
        encryption = EncryptionLayer(config.sync_key)
        adapter = LocalFileAdapter(config.sync_path)
        manager = SyncManager(db, adapter, encryption, project_id)
        
        with console.status("[bold green]Pulling and decrypting memories..."):
            count = manager.import_memories()
            
        console.print(f"[green]✓ Pulled {count} new/updated memories[/green]")
        
    except (ImportError, EncryptionError) as e:
        console.print(f"[red]Sync failed: {e}[/red]")


@sync.command("status")
@click.pass_context
def sync_status(ctx: click.Context) -> None:
    """Show sync configuration and status."""
    config: Config = ctx.obj["config"]
    
    if not config.sync_path or not config.sync_key:
        console.print("[yellow]Sync not initialized.[/yellow]")
        console.print("Run: [cyan]memoryforge sync init --path <sync-dir>[/cyan]")
        return
    
    console.print(Panel.fit(
        f"[bold]Backend:[/bold] {config.sync_backend or 'local'}\n"
        f"[bold]Path:[/bold] {config.sync_path}\n"
        f"[bold]Key:[/bold] {'*' * 8}...{config.sync_key[-8:] if len(config.sync_key) > 8 else '****'}",
        title="Sync Configuration",
        border_style="blue",
    ))
    
    # Check if sync path exists and has files
    if config.sync_path.exists():
        sync_files = list(config.sync_path.glob("*.json"))
        console.print(f"\n[bold]Remote memories:[/bold] {len(sync_files)} files")
    else:
        console.print("\n[yellow]Sync directory does not exist yet.[/yellow]")


# ============================================================================
# Reindex Command
# ============================================================================

@main.command("reindex")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def reindex(ctx: click.Context, force: bool) -> None:
    """Rebuild vector index from SQLite data.
    
    Use this after changing embedding provider or if Qdrant gets corrupted.
    """
    config: Config = ctx.obj["config"]
    db, qdrant, project_id = ensure_initialized(config)
    
    from memoryforge.core.embedding_factory import create_embedding_service
    
    # Count memories to reindex
    memory_count = db.get_memory_count(project_id, confirmed_only=True)
    
    if memory_count == 0:
        console.print("[dim]No confirmed memories to reindex.[/dim]")
        return
    
    console.print(f"This will re-embed {memory_count} memories.")
    console.print(f"Embedding provider: {config.embedding_provider.value}")
    
    if not force and not Confirm.ask("Proceed with reindex?"):
        return
    
    embedding_service = create_embedding_service(config)
    
    # Get all confirmed memories
    memories = db.list_memories(project_id, confirmed_only=True, limit=10000)
    
    success_count = 0
    error_count = 0
    
    with console.status("[bold green]Reindexing memories...") as status:
        for i, memory in enumerate(memories):
            try:
                # Generate new embedding
                embedding = embedding_service.generate(memory.content)
                
                # Upsert to Qdrant
                vector_id = qdrant.upsert(
                    memory_id=str(memory.id),
                    embedding=embedding,
                    metadata={
                        "type": memory.type.value,
                        "project_id": str(project_id),
                    }
                )
                
                # Update embedding reference
                db.save_embedding_reference(memory.id, vector_id)
                
                success_count += 1
                status.update(f"[bold green]Reindexing memories... {i+1}/{len(memories)}")
                
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to reindex memory {memory.id}: {e}")
    
    console.print(f"\n[green]✓ Reindexed {success_count} memories[/green]")
    if error_count > 0:
        console.print(f"[yellow]⚠ {error_count} memories failed[/yellow]")


if __name__ == "__main__":
    main()
