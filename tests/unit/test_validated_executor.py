# tests/unit/test_validated_executor.py
"""Tests for MCP validated tool executor."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from src.domain.ports.mcp_client_port import ToolResult
from src.infrastructure.mcp.validated_executor import ValidatedToolExecutor


class SearchInput(BaseModel):
    query: str
    limit: int = 10


class SearchOutput(BaseModel):
    results: list[str]
    total: int


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.call_tool.return_value = ToolResult(content={"results": ["a", "b"], "total": 2})
    return client


@pytest.fixture
def executor(mock_client: AsyncMock) -> ValidatedToolExecutor:
    ex = ValidatedToolExecutor(client=mock_client, default_timeout=5.0)
    ex.register_schema("search", input_schema=SearchInput, output_schema=SearchOutput)
    return ex


class TestValidatedToolExecutor:
    async def test_valid_execution(self, executor: ValidatedToolExecutor) -> None:
        result = await executor.execute("search", {"query": "test", "limit": 5})
        assert result.content == {"results": ["a", "b"], "total": 2}

    async def test_invalid_input_raises(self, executor: ValidatedToolExecutor) -> None:
        with pytest.raises(ValueError, match="Input validation failed"):
            await executor.execute("search", {"wrong_field": "bad"})

    async def test_invalid_output_raises(
        self, executor: ValidatedToolExecutor, mock_client: AsyncMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(content={"bad": "schema"})
        with pytest.raises(ValueError, match="Output validation failed"):
            await executor.execute("search", {"query": "test"})

    async def test_timeout(self, executor: ValidatedToolExecutor, mock_client: AsyncMock) -> None:
        async def slow_call(*args: object, **kwargs: object) -> ToolResult:
            await asyncio.sleep(10)
            return ToolResult(content="late")

        mock_client.call_tool.side_effect = slow_call
        with pytest.raises(TimeoutError):
            await executor.execute("search", {"query": "test"}, timeout=0.05)

    async def test_no_schema_registered(
        self, executor: ValidatedToolExecutor, mock_client: AsyncMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(content="anything")
        result = await executor.execute("unregistered_tool", {"x": 1})
        assert result.content == "anything"

    async def test_error_result_skips_output_validation(
        self, executor: ValidatedToolExecutor, mock_client: AsyncMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(content="error msg", is_error=True)
        # Should not raise even though content doesn't match SearchOutput
        result = await executor.execute("search", {"query": "test"})
        assert result.is_error is True
