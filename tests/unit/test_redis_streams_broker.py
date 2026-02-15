# tests/unit/test_redis_streams_broker.py
"""Tests for Redis Streams broker."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.infrastructure.events.redis_streams_broker import RedisStreamsBroker


@pytest.fixture
def mock_redis_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def broker(mock_redis_client: AsyncMock) -> RedisStreamsBroker:
    with patch("src.infrastructure.events.redis_streams_broker.redis") as mock_redis:
        mock_redis.from_url.return_value = mock_redis_client
        return RedisStreamsBroker(redis_url="redis://localhost:6379")


class TestRedisStreamsBroker:
    async def test_publish(self, broker: RedisStreamsBroker, mock_redis_client: AsyncMock) -> None:
        await broker.publish("events.topic", {"key": "value"}, key="k1")
        mock_redis_client.xadd.assert_called_once_with(
            "events.topic",
            {"data": json.dumps({"key": "value"}), "key": "k1"},
        )

    async def test_publish_without_key(
        self, broker: RedisStreamsBroker, mock_redis_client: AsyncMock
    ) -> None:
        await broker.publish("topic", {"msg": "hello"})
        call_args = mock_redis_client.xadd.call_args
        payload = call_args[0][1]
        assert "key" not in payload
        assert json.loads(payload["data"]) == {"msg": "hello"}

    async def test_close(self, broker: RedisStreamsBroker, mock_redis_client: AsyncMock) -> None:
        await broker.close()
        mock_redis_client.close.assert_called_once()
