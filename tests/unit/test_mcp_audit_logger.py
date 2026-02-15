# tests/unit/test_mcp_audit_logger.py
"""Tests for MCP audit logger."""

from src.infrastructure.mcp.audit_logger import MCPAuditLogger


class TestMCPAuditLogger:
    def test_record_entry(self) -> None:
        logger = MCPAuditLogger()
        entry = logger.record("search", "agent-1", 150.0)
        assert entry.tool_name == "search"
        assert entry.agent_id == "agent-1"
        assert entry.duration_ms == 150.0
        assert entry.is_error is False

    def test_record_error_entry(self) -> None:
        logger = MCPAuditLogger()
        entry = logger.record("delete", "agent-2", 50.0, is_error=True, details="timeout")
        assert entry.is_error is True
        assert entry.details == "timeout"

    def test_get_entries_all(self) -> None:
        logger = MCPAuditLogger()
        logger.record("search", "a1", 100.0)
        logger.record("read", "a2", 200.0)
        entries = logger.get_entries()
        assert len(entries) == 2

    def test_get_entries_by_tool(self) -> None:
        logger = MCPAuditLogger()
        logger.record("search", "a1", 100.0)
        logger.record("read", "a1", 200.0)
        logger.record("search", "a2", 150.0)
        entries = logger.get_entries(tool_name="search")
        assert len(entries) == 2

    def test_get_entries_by_agent(self) -> None:
        logger = MCPAuditLogger()
        logger.record("search", "a1", 100.0)
        logger.record("read", "a2", 200.0)
        entries = logger.get_entries(agent_id="a1")
        assert len(entries) == 1

    def test_get_entries_errors_only(self) -> None:
        logger = MCPAuditLogger()
        logger.record("search", "a1", 100.0)
        logger.record("delete", "a1", 50.0, is_error=True)
        entries = logger.get_entries(errors_only=True)
        assert len(entries) == 1
        assert entries[0].is_error is True

    def test_get_entries_with_limit(self) -> None:
        logger = MCPAuditLogger()
        for i in range(10):
            logger.record(f"tool_{i}", "a1", float(i))
        entries = logger.get_entries(limit=3)
        assert len(entries) == 3

    def test_get_stats(self) -> None:
        logger = MCPAuditLogger()
        logger.record("search", "a1", 100.0)
        logger.record("read", "a1", 200.0)
        logger.record("search", "a2", 150.0, is_error=True)
        stats = logger.get_stats()
        assert stats["total_calls"] == 3
        assert stats["error_calls"] == 1
        assert stats["unique_tools"] == 2

    def test_clear(self) -> None:
        logger = MCPAuditLogger()
        logger.record("search", "a1", 100.0)
        logger.clear()
        assert logger.get_entries() == []
