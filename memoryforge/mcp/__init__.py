"""MCP server for MemoryForge."""

# Avoid circular import by using lazy imports
__all__ = ["create_mcp_server", "run_mcp_server"]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in __all__:
        from memoryforge.mcp.server import create_mcp_server, run_mcp_server
        if name == "create_mcp_server":
            return create_mcp_server
        elif name == "run_mcp_server":
            return run_mcp_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
