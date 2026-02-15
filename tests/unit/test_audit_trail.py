# tests/unit/test_audit_trail.py
"""Tests for in-memory audit trail."""

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.entities.audit_record import AuditRecord
from src.infrastructure.governance.in_memory_audit_trail import InMemoryAuditTrail


@pytest.fixture
def audit_trail() -> InMemoryAuditTrail:
    return InMemoryAuditTrail()


def _make_record(**overrides: object) -> AuditRecord:
    defaults = {
        "action": "api_call",
        "actor": "user-1",
        "resource": "/api/v1/chat",
    }
    defaults.update(overrides)
    return AuditRecord(**defaults)  # type: ignore[arg-type]


class TestInMemoryAuditTrail:
    async def test_record_and_query(self, audit_trail: InMemoryAuditTrail) -> None:
        entry = _make_record()
        await audit_trail.record(entry)
        results = await audit_trail.query()
        assert len(results) == 1
        assert results[0].action == "api_call"

    async def test_query_by_action(self, audit_trail: InMemoryAuditTrail) -> None:
        await audit_trail.record(_make_record(action="read"))
        await audit_trail.record(_make_record(action="write"))
        results = await audit_trail.query(action="read")
        assert len(results) == 1
        assert results[0].action == "read"

    async def test_query_by_actor(self, audit_trail: InMemoryAuditTrail) -> None:
        await audit_trail.record(_make_record(actor="alice"))
        await audit_trail.record(_make_record(actor="bob"))
        results = await audit_trail.query(actor="alice")
        assert len(results) == 1

    async def test_query_by_since(self, audit_trail: InMemoryAuditTrail) -> None:
        old = AuditRecord(
            action="old",
            actor="user",
            resource="/",
            timestamp=datetime.now(UTC) - timedelta(hours=2),
        )
        new = _make_record(action="new")
        await audit_trail.record(old)
        await audit_trail.record(new)

        since = datetime.now(UTC) - timedelta(hours=1)
        results = await audit_trail.query(since=since)
        assert len(results) == 1
        assert results[0].action == "new"

    async def test_query_with_limit(self, audit_trail: InMemoryAuditTrail) -> None:
        for i in range(10):
            await audit_trail.record(_make_record(action=f"action_{i}"))
        results = await audit_trail.query(limit=3)
        assert len(results) == 3


class TestAuditRecord:
    def test_frozen(self) -> None:
        record = _make_record()
        with pytest.raises(AttributeError):
            record.action = "changed"  # type: ignore[misc]

    def test_auto_id_and_timestamp(self) -> None:
        record = _make_record()
        assert record.id
        assert record.timestamp
