# src/infrastructure/mcp/validated_executor.py
"""MCP tool executor with input/output validation and timeouts."""

import asyncio
import time
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

from src.domain.ports.mcp_client_port import MCPClientPort, ToolResult

logger = structlog.get_logger(__name__)


class ValidatedToolExecutor:
    """Wraps an MCPClientPort with schema validation and execution timeouts.

    Register input/output schemas per tool. On execution:
    1. Validate input against registered schema
    2. Execute with asyncio.wait_for timeout
    3. Validate output against registered schema
    4. Log execution details via structlog
    """

    DEFAULT_TIMEOUT: float = 30.0

    def __init__(self, client: MCPClientPort, default_timeout: float | None = None) -> None:
        self._client = client
        self._default_timeout = default_timeout or self.DEFAULT_TIMEOUT
        self._schemas: dict[str, tuple[type[BaseModel] | None, type[BaseModel] | None]] = {}

    def register_schema(
        self,
        tool_name: str,
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        """Register input and/or output validation schemas for a tool."""
        self._schemas[tool_name] = (input_schema, output_schema)

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        timeout: float | None = None,
    ) -> ToolResult:
        """Execute a tool with validation and timeout.

        Raises:
            ValueError: If input or output validation fails.
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        effective_timeout = timeout or self._default_timeout
        input_schema, output_schema = self._schemas.get(name, (None, None))

        # Validate input
        if input_schema is not None:
            try:
                input_schema(**args)
            except ValidationError as e:
                logger.error("mcp_input_validation_failed", tool=name, errors=str(e))
                raise ValueError(f"Input validation failed for tool '{name}': {e}") from e

        # Execute with timeout
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._client.call_tool(name, args),
                timeout=effective_timeout,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("mcp_tool_timeout", tool=name, duration_ms=duration_ms)
            raise

        duration_ms = (time.monotonic() - start) * 1000

        # Validate output
        if output_schema is not None and not result.is_error:
            try:
                if isinstance(result.content, dict):
                    output_schema(**result.content)
                else:
                    logger.warning(
                        "mcp_output_not_dict",
                        tool=name,
                        content_type=type(result.content).__name__,
                    )
            except ValidationError as e:
                logger.error("mcp_output_validation_failed", tool=name, errors=str(e))
                raise ValueError(f"Output validation failed for tool '{name}': {e}") from e

        logger.info(
            "mcp_tool_executed",
            tool=name,
            duration_ms=round(duration_ms, 2),
            is_error=result.is_error,
        )
        return result
