# src/infrastructure/governance/in_memory_audit_trail.py
"""In-memory audit trail implementation."""

from datetime import datetime

from src.domain.entities.audit_record import AuditRecord
from src.domain.ports.audit_trail_port import AuditTrailPort


class InMemoryAuditTrail(AuditTrailPort):
    """In-memory audit trail for development and testing.

    For production, use a database-backed implementation.
    """

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    async def record(self, entry: AuditRecord) -> None:
        self._records.append(entry)

    async def query(
        self,
        action: str | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        results = self._records
        if action:
            results = [r for r in results if r.action == action]
        if actor:
            results = [r for r in results if r.actor == actor]
        if since:
            results = [r for r in results if r.timestamp >= since]
        return results[-limit:]
