"""Local filesystem storage adapter.

Implements StoragePort for local file operations using aiofiles
for async I/O operations.
"""

from pathlib import Path

import aiofiles

from src.domain.ports.storage_port import StoragePort


class LocalStorage(StoragePort):
    """Local filesystem storage implementation.

    All paths are relative to base_path for security and isolation.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize local storage.

        Args:
            base_path: Root directory for all storage operations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path.

        Args:
            path: Relative path

        Returns:
            Absolute resolved path
        """
        resolved = (self.base_path / path).resolve()
        # Security: ensure path is within base_path
        if not str(resolved).startswith(str(self.base_path.resolve())):
            raise ValueError(f"Path {path} is outside base_path")
        return resolved

    async def read(self, path: str) -> bytes:
        """Read file content from local filesystem.

        Args:
            path: Relative path to file

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._resolve_path(path)
        async with aiofiles.open(file_path, mode="rb") as f:
            content: bytes = await f.read()
            return content

    async def write(self, path: str, data: bytes) -> None:
        """Write data to local filesystem.

        Creates parent directories if they don't exist.

        Args:
            path: Relative path where to write
            data: Content to write

        Raises:
            OSError: If write operation fails
        """
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, mode="wb") as f:
            await f.write(data)

    async def list(self, prefix: str) -> list[str]:
        """List all files matching the given prefix.

        Args:
            prefix: Path prefix to filter files (supports glob patterns)

        Returns:
            List of relative file paths matching the prefix
        """
        base = self.base_path.resolve()
        pattern_path = base / prefix

        # Handle glob patterns
        if "*" in prefix or "?" in prefix:
            matches = list(base.glob(prefix))
        else:
            # If no glob, treat as directory prefix
            if pattern_path.is_dir():
                matches = list(pattern_path.rglob("*"))
            else:
                matches = list(base.glob(f"{prefix}*"))

        # Filter only files and return relative paths
        return [str(p.relative_to(base)) for p in matches if p.is_file()]

    async def exists(self, path: str) -> bool:
        """Check if a file exists.

        Args:
            path: Relative path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            file_path = self._resolve_path(path)
            return file_path.exists() and file_path.is_file()
        except ValueError:
            return False
