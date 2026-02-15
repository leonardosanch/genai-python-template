"""MCP client adapter — communicates with MCP servers via stdio JSON-RPC."""

import asyncio
import json
import logging
from typing import Any

from src.domain.ports.mcp_client_port import MCPClientPort, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


class StdioMCPClient(MCPClientPort):
    """MCP client that communicates via subprocess stdio using JSON-RPC.

    Args:
        command: Command to start the MCP server process.
        args: Arguments for the command.
        timeout: Timeout in seconds for tool calls.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._command = command
        self._args = args or []
        self._timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is None or self._process.returncode is not None:
            self._process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        return self._process

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        process = await self._ensure_process()
        assert process.stdin is not None
        assert process.stdout is not None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        payload = json.dumps(request) + "\n"
        logger.debug("MCP request: %s", payload.strip())
        process.stdin.write(payload.encode())
        await process.stdin.drain()

        line = await asyncio.wait_for(process.stdout.readline(), timeout=self._timeout)
        response = json.loads(line.decode())
        logger.debug("MCP response: %s", response)

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result")

    async def list_tools(self) -> list[ToolDefinition]:
        result = await self._send_request("tools/list")
        tools: list[ToolDefinition] = []
        for tool_data in result.get("tools", []):
            tools.append(
                ToolDefinition(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                )
            )
        return tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        logger.info("MCP call_tool: %s args=%s", name, args)
        result = await self._send_request(
            "tools/call",
            params={"name": name, "arguments": args},
        )
        is_error = result.get("isError", False)
        content = result.get("content", [])
        return ToolResult(content=content, is_error=is_error)

    async def close(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
            self._process = None
