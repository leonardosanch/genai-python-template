"""Storage port — abstract interface for file/object storage operations.

This port abstracts storage backends (local filesystem, S3, GCS, Azure Blob, etc.)
to enable testability and flexibility in data pipelines.
"""

from abc import ABC, abstractmethod


class StoragePort(ABC):
    """Abstract interface for storage operations."""

    @abstractmethod
    async def read(self, path: str) -> bytes:
        """Read file content from storage.

        Args:
            path: Path to the file to read

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file does not exist
            PermissionError: If access is denied
        """
        ...

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Write data to storage.

        Args:
            path: Path where to write the file
            data: Content to write

        Raises:
            PermissionError: If write access is denied
            OSError: If write operation fails
        """
        ...

    @abstractmethod
    async def list(self, prefix: str) -> list[str]:
        """List all files matching the given prefix.

        Args:
            prefix: Path prefix to filter files

        Returns:
            List of file paths matching the prefix

        Raises:
            PermissionError: If list access is denied
        """
        ...

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if a file exists in storage.

        Args:
            path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        ...
