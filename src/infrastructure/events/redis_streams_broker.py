# src/infrastructure/events/redis_streams_broker.py
"""Message broker implementation using Redis Streams."""

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

from src.domain.ports.message_broker_port import MessageBrokerPort

logger = structlog.get_logger(__name__)

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisStreamsBroker(MessageBrokerPort):
    """Message broker using Redis Streams with consumer groups.

    Publish: XADD to stream.
    Subscribe: XREADGROUP with consumer groups for at-least-once delivery.
    """

    def __init__(self, redis_url: str, consumer_name: str = "worker-1") -> None:
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package required. Install with: pip install redis")
        self._client: redis.Redis = redis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
        self._consumer_name = consumer_name

    async def publish(self, topic: str, message: dict[str, Any], key: str | None = None) -> None:
        """Publish a message to a Redis stream (XADD)."""
        payload = {"data": json.dumps(message)}
        if key:
            payload["key"] = key
        await self._client.xadd(topic, payload)  # type: ignore[arg-type]
        logger.debug("redis_stream_published", topic=topic, key=key)

    async def subscribe(self, topic: str, group_id: str) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to a Redis stream using consumer groups (XREADGROUP).

        Creates the consumer group if it doesn't exist.
        """
        # Ensure consumer group exists
        try:
            await self._client.xgroup_create(topic, group_id, id="0", mkstream=True)
        except Exception:
            # Group may already exist
            pass

        while True:
            messages = await self._client.xreadgroup(
                groupname=group_id,
                consumername=self._consumer_name,
                streams={topic: ">"},
                count=10,
                block=1000,
            )
            if not messages:
                continue

            for _stream, entries in messages:
                for msg_id, fields in entries:
                    try:
                        data = json.loads(fields.get("data", "{}"))
                        yield data
                        await self._client.xack(topic, group_id, msg_id)
                    except json.JSONDecodeError:
                        logger.warning("redis_stream_decode_error", msg_id=msg_id)

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.close()
