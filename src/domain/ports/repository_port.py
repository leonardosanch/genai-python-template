"""Port for data persistence. Infrastructure repositories implement this."""

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class RepositoryPort[T](ABC):
    """Generic repository port for CRUD operations.

    Each entity should have its own repository port
    that extends this with domain-specific queries.
    """

    @abstractmethod
    async def save(self, entity: T) -> str:
        """Persist an entity. Returns the ID."""
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> T | None:
        """Retrieve an entity by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def list(self, offset: int = 0, limit: int = 20) -> list[T]:
        """List entities with pagination."""
        ...

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete an entity by ID."""
        ...
