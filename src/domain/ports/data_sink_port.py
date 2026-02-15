"""Port for data sinks. Infrastructure adapters implement this interface."""

from abc import ABC, abstractmethod
from typing import Any


class DataSinkPort(ABC):
    """Abstract interface for writing data to external destinations.

    Implementations handle CSV, JSON, databases, cloud storage, etc.
    Domain and application layers depend only on this interface.
    """

    @abstractmethod
    async def write_records(
        self, uri: str, records: list[dict[str, object]], **options: Any
    ) -> int:
        """Write records to a data destination.

        Args:
            uri: Destination location (file path, URL, etc.).
            records: List of records to write.
            **options: Sink-specific options.

        Returns:
            Number of records successfully written.

        Raises:
            DataSinkError: If writing fails.
        """
        ...
