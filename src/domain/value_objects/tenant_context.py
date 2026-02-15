# src/domain/value_objects/tenant_context.py
"""Tenant context value object for multi-tenancy support."""

from dataclasses import dataclass
from enum import Enum


class TenantTier(Enum):
    """Subscription tier for tenants."""

    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class TenantContext:
    """Immutable tenant context propagated through the request lifecycle.

    Attributes:
        tenant_id: Unique tenant identifier.
        tenant_name: Human-readable tenant name.
        tier: Subscription tier affecting feature access and limits.
    """

    tenant_id: str
    tenant_name: str = ""
    tier: TenantTier = TenantTier.STANDARD

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
