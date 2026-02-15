"""Port for MCP (Model Context Protocol) client interactions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """Descriptor for an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of an MCP tool invocation."""

    content: Any
    is_error: bool = False


class MCPClientPort(ABC):
    """Abstract interface for MCP client interactions.

    Implementations communicate with MCP servers via stdio, HTTP SSE, etc.
    """

    @abstractmethod
    async def list_tools(self) -> list[ToolDefinition]:
        """List available tools from the MCP server."""
        ...

    @abstractmethod
    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Invoke a tool on the MCP server.

        Args:
            name: Tool name.
            args: Tool input arguments.

        Returns:
            ToolResult with the tool's response.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the MCP server."""
        ...
