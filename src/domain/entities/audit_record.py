# src/domain/entities/audit_record.py
"""Audit record entity for governance and compliance."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AuditRecord:
    """Immutable audit trail entry.

    Used for tracking actions, compliance (GDPR, EU AI Act), and debugging.
    """

    action: str
    actor: str
    resource: str
    details: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
