"""Tests for S3 storage adapter."""

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

AIOBOTOCORE_AVAILABLE = importlib.util.find_spec("aiobotocore") is not None


@pytest.mark.skipif(not AIOBOTOCORE_AVAILABLE, reason="aiobotocore not installed")
class TestS3Storage:
    def _make_adapter(self) -> "S3Storage":  # noqa: F821
        from src.infrastructure.cloud.s3_storage import S3Storage

        return S3Storage(bucket="test-bucket", region="us-east-1")

    @pytest.mark.asyncio
    async def test_read(self) -> None:
        adapter = self._make_adapter()

        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=b"hello")
        mock_body.__aenter__ = AsyncMock(return_value=mock_body)
        mock_body.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(return_value={"Body": mock_body})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.create_client = MagicMock(return_value=mock_client)

        with patch("aiobotocore.session.get_session", return_value=mock_session):
            result = await adapter.read("test.txt")

        assert result == b"hello"

    @pytest.mark.asyncio
    async def test_list(self) -> None:
        adapter = self._make_adapter()

        mock_client = AsyncMock()
        mock_client.list_objects_v2 = AsyncMock(
            return_value={"Contents": [{"Key": "a.txt"}, {"Key": "b.txt"}]}
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.create_client = MagicMock(return_value=mock_client)

        with patch("aiobotocore.session.get_session", return_value=mock_session):
            result = await adapter.list("prefix/")

        assert result == ["a.txt", "b.txt"]

    @pytest.mark.asyncio
    async def test_exists_true(self) -> None:
        adapter = self._make_adapter()

        mock_client = AsyncMock()
        mock_client.head_object = AsyncMock(return_value={})
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.create_client = MagicMock(return_value=mock_client)

        with patch("aiobotocore.session.get_session", return_value=mock_session):
            assert await adapter.exists("test.txt") is True

    @pytest.mark.asyncio
    async def test_exists_false(self) -> None:
        adapter = self._make_adapter()

        mock_client = AsyncMock()
        mock_client.head_object = AsyncMock(side_effect=Exception("not found"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.create_client = MagicMock(return_value=mock_client)

        with patch("aiobotocore.session.get_session", return_value=mock_session):
            assert await adapter.exists("missing.txt") is False
