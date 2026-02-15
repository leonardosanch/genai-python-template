# src/domain/ports/audit_trail_port.py
"""Port for audit trail persistence."""

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.audit_record import AuditRecord


class AuditTrailPort(ABC):
    """Abstract interface for audit trail operations.

    Implementations can use databases, files, or external services.
    """

    @abstractmethod
    async def record(self, entry: AuditRecord) -> None:
        """Record an audit entry."""
        ...

    @abstractmethod
    async def query(
        self,
        action: str | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        """Query audit entries with optional filters."""
        ...
