# src/infrastructure/database/tenant_aware_repository.py
"""Base class for tenant-aware repositories."""

from abc import ABC, abstractmethod
from typing import TypeVar

from src.domain.ports.repository_port import RepositoryPort
from src.interfaces.api.middleware.tenant_middleware import get_current_tenant

T = TypeVar("T")


class TenantAwareRepository(RepositoryPort[T], ABC):
    """Base class for repositories that enforce tenant isolation.

    Subclasses must implement the abstract methods. The tenant_id
    is automatically injected from the current request context.
    """

    def _get_tenant_id(self) -> str | None:
        """Get the current tenant ID from context.

        Returns None if no tenant context is set (e.g., in tests or admin mode).
        """
        ctx = get_current_tenant()
        return ctx.tenant_id if ctx else None

    @abstractmethod
    async def save(self, entity: T) -> str: ...

    @abstractmethod
    async def get_by_id(self, id: str) -> T | None: ...

    @abstractmethod
    async def list(self, offset: int = 0, limit: int = 20) -> list[T]: ...

    @abstractmethod
    async def delete(self, id: str) -> None: ...
