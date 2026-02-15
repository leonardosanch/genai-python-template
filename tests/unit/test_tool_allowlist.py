# tests/unit/test_tool_allowlist.py
"""Tests for MCP tool allowlist."""

from unittest.mock import AsyncMock

import pytest

from src.domain.ports.mcp_client_port import ToolDefinition, ToolResult
from src.infrastructure.mcp.tool_allowlist import ToolAllowlist


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.list_tools.return_value = [
        ToolDefinition(name="search", description="Search", input_schema={}),
        ToolDefinition(name="delete", description="Delete", input_schema={}),
        ToolDefinition(name="read", description="Read", input_schema={}),
    ]
    client.call_tool.return_value = ToolResult(content="result")
    return client


@pytest.fixture
def allowlist(mock_client: AsyncMock) -> ToolAllowlist:
    return ToolAllowlist(client=mock_client, allowed_tools={"search", "read"})


class TestToolAllowlist:
    async def test_list_tools_filters(self, allowlist: ToolAllowlist) -> None:
        tools = await allowlist.list_tools()
        names = {t.name for t in tools}
        assert names == {"search", "read"}
        assert "delete" not in names

    async def test_call_allowed_tool(
        self, allowlist: ToolAllowlist, mock_client: AsyncMock
    ) -> None:
        result = await allowlist.call_tool("search", {"q": "test"})
        assert result.content == "result"
        mock_client.call_tool.assert_called_once_with("search", {"q": "test"})

    async def test_call_denied_tool_raises(self, allowlist: ToolAllowlist) -> None:
        with pytest.raises(PermissionError, match="not in the allowlist"):
            await allowlist.call_tool("delete", {})

    async def test_allowed_tools_property(self, allowlist: ToolAllowlist) -> None:
        assert allowlist.allowed_tools == frozenset({"search", "read"})

    async def test_close_delegates(self, allowlist: ToolAllowlist, mock_client: AsyncMock) -> None:
        await allowlist.close()
        mock_client.close.assert_called_once()
