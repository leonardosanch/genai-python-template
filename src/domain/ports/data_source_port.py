"""Port for data sources. Infrastructure adapters implement this interface."""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.schema_definition import SchemaDefinition


class DataSourcePort(ABC):
    """Abstract interface for reading data from external sources.

    Implementations handle CSV, JSON, databases, APIs, etc.
    Domain and application layers depend only on this interface.
    """

    @abstractmethod
    async def read_records(self, uri: str, **options: Any) -> list[dict[str, object]]:
        """Read records from a data source.

        Args:
            uri: Source location (file path, URL, connection string).
            **options: Source-specific options.

        Returns:
            List of records as dictionaries.

        Raises:
            DataSourceError: If reading fails.
        """
        ...

    @abstractmethod
    async def read_schema(self, uri: str) -> SchemaDefinition | None:
        """Read or infer schema from a data source.

        Args:
            uri: Source location.

        Returns:
            Inferred schema or None if not available.
        """
        ...
