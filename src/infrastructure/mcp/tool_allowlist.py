# src/infrastructure/mcp/tool_allowlist.py
"""MCP tool allowlist — restricts tool access to an explicit set."""

from typing import Any

import structlog

from src.domain.ports.mcp_client_port import MCPClientPort, ToolDefinition, ToolResult

logger = structlog.get_logger(__name__)


class ToolAllowlist(MCPClientPort):
    """Wraps an MCPClientPort and filters tools to an allowlist.

    Only tools in the allowlist can be listed or called.
    Calling a non-allowed tool raises PermissionError.
    """

    def __init__(self, client: MCPClientPort, allowed_tools: set[str]) -> None:
        self._client = client
        self._allowed_tools = frozenset(allowed_tools)

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._allowed_tools

    async def list_tools(self) -> list[ToolDefinition]:
        """List only tools that are in the allowlist."""
        all_tools = await self._client.list_tools()
        filtered = [t for t in all_tools if t.name in self._allowed_tools]
        logger.info(
            "mcp_tools_filtered",
            total=len(all_tools),
            allowed=len(filtered),
        )
        return filtered

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool only if it's in the allowlist.

        Raises:
            PermissionError: If the tool is not in the allowlist.
        """
        if name not in self._allowed_tools:
            logger.warning("mcp_tool_denied", tool_name=name)
            raise PermissionError(
                f"Tool '{name}' is not in the allowlist. "
                f"Allowed tools: {sorted(self._allowed_tools)}"
            )
        return await self._client.call_tool(name, args)

    async def close(self) -> None:
        await self._client.close()
