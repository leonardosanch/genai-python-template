"""Tests for LocalStorage — filesystem storage adapter."""

import pytest

from src.infrastructure.storage.local_storage import LocalStorage


@pytest.fixture()
def storage(tmp_path: object) -> LocalStorage:
    return LocalStorage(base_path=str(tmp_path))


class TestLocalStorage:
    """Tests for the local filesystem storage."""

    @pytest.mark.asyncio()
    async def test_write_and_read(self, storage: LocalStorage) -> None:
        await storage.write("test.txt", b"hello world")
        content = await storage.read("test.txt")
        assert content == b"hello world"

    @pytest.mark.asyncio()
    async def test_read_nonexistent_raises(self, storage: LocalStorage) -> None:
        with pytest.raises(FileNotFoundError):
            await storage.read("nonexistent.txt")

    @pytest.mark.asyncio()
    async def test_exists_true(self, storage: LocalStorage) -> None:
        await storage.write("exists.txt", b"data")
        assert await storage.exists("exists.txt") is True

    @pytest.mark.asyncio()
    async def test_exists_false(self, storage: LocalStorage) -> None:
        assert await storage.exists("nope.txt") is False

    @pytest.mark.asyncio()
    async def test_write_creates_subdirectories(self, storage: LocalStorage) -> None:
        await storage.write("sub/dir/file.txt", b"nested")
        content = await storage.read("sub/dir/file.txt")
        assert content == b"nested"

    @pytest.mark.asyncio()
    async def test_list_files(self, storage: LocalStorage) -> None:
        await storage.write("a.txt", b"1")
        await storage.write("b.txt", b"2")
        files = await storage.list("*")
        assert sorted(files) == ["a.txt", "b.txt"]

    @pytest.mark.asyncio()
    async def test_list_subdirectory(self, storage: LocalStorage) -> None:
        await storage.write("data/f1.csv", b"x")
        await storage.write("data/f2.csv", b"y")
        await storage.write("other.txt", b"z")
        files = await storage.list("data")
        assert len(files) == 2

    @pytest.mark.asyncio()
    async def test_path_traversal_blocked(self, storage: LocalStorage) -> None:
        with pytest.raises(ValueError, match="outside base_path"):
            storage._resolve_path("../../etc/passwd")

    @pytest.mark.asyncio()
    async def test_exists_returns_false_for_traversal(self, storage: LocalStorage) -> None:
        result = await storage.exists("../../etc/passwd")
        assert result is False
