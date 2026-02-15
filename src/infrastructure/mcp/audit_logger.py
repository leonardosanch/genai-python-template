# src/infrastructure/mcp/audit_logger.py
"""Audit logging for MCP tool invocations."""

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ToolCallAuditEntry:
    """Immutable record of an MCP tool invocation."""

    timestamp: datetime
    tool_name: str
    agent_id: str
    duration_ms: float
    is_error: bool
    details: str = ""


class MCPAuditLogger:
    """In-memory audit log for MCP tool calls.

    Stores entries in memory and provides query methods.
    For production, replace with a persistent store.
    """

    def __init__(self) -> None:
        self._entries: list[ToolCallAuditEntry] = []

    def record(
        self,
        tool_name: str,
        agent_id: str,
        duration_ms: float,
        is_error: bool = False,
        details: str = "",
    ) -> ToolCallAuditEntry:
        """Record a tool call audit entry."""
        entry = ToolCallAuditEntry(
            timestamp=datetime.now(UTC),
            tool_name=tool_name,
            agent_id=agent_id,
            duration_ms=duration_ms,
            is_error=is_error,
            details=details,
        )
        self._entries.append(entry)
        return entry

    def get_entries(
        self,
        tool_name: str | None = None,
        agent_id: str | None = None,
        errors_only: bool = False,
        limit: int = 100,
    ) -> list[ToolCallAuditEntry]:
        """Query audit entries with optional filters."""
        results = self._entries
        if tool_name:
            results = [e for e in results if e.tool_name == tool_name]
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if errors_only:
            results = [e for e in results if e.is_error]
        return results[-limit:]

    def get_stats(self) -> dict[str, int]:
        """Get summary statistics."""
        total = len(self._entries)
        errors = sum(1 for e in self._entries if e.is_error)
        unique_tools = len({e.tool_name for e in self._entries})
        return {
            "total_calls": total,
            "error_calls": errors,
            "unique_tools": unique_tools,
        }

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
