"""Tests for stdio MCP client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.ports.mcp_client_port import ToolDefinition, ToolResult
from src.infrastructure.mcp.stdio_mcp_client import StdioMCPClient


@pytest.fixture
def client() -> StdioMCPClient:
    return StdioMCPClient(command="echo", args=["test"])


def _mock_process(response: dict) -> MagicMock:
    """Create a mock subprocess with a pre-loaded response."""
    process = MagicMock()
    process.returncode = None
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()

    response_line = json.dumps(response).encode() + b"\n"
    process.stdout = MagicMock()
    process.stdout.readline = AsyncMock(return_value=response_line)

    process.terminate = MagicMock()
    process.wait = AsyncMock()
    return process


@pytest.mark.asyncio
async def test_list_tools(client: StdioMCPClient) -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": "calculator",
                    "description": "Does math",
                    "inputSchema": {"type": "object"},
                }
            ]
        },
    }
    mock_proc = _mock_process(response)

    with patch.object(client, "_ensure_process", return_value=mock_proc):
        tools = await client.list_tools()

    assert len(tools) == 1
    assert tools[0] == ToolDefinition(
        name="calculator",
        description="Does math",
        input_schema={"type": "object"},
    )


@pytest.mark.asyncio
async def test_call_tool(client: StdioMCPClient) -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "42"}], "isError": False},
    }
    mock_proc = _mock_process(response)

    with patch.object(client, "_ensure_process", return_value=mock_proc):
        result = await client.call_tool("calculator", {"expr": "6*7"})

    assert result == ToolResult(content=[{"type": "text", "text": "42"}], is_error=False)


@pytest.mark.asyncio
async def test_call_tool_error_response(client: StdioMCPClient) -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -1, "message": "not found"},
    }
    mock_proc = _mock_process(response)

    with patch.object(client, "_ensure_process", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="MCP error"):
            await client.call_tool("unknown", {})


@pytest.mark.asyncio
async def test_close(client: StdioMCPClient) -> None:
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock()
    client._process = mock_proc

    await client.close()
    mock_proc.terminate.assert_called_once()
