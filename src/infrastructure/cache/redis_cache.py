"""Redis cache implementation.

Reference implementation for caching LLM responses,
session data, and rate limiting.
"""

import redis.asyncio as redis

from src.infrastructure.config import get_settings


class RedisCache:
    """Async Redis cache client.

    Use cases in GenAI systems:
    - Cache LLM responses by prompt hash
    - Rate limiting counters
    - Session storage
    - Distributed locks
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = redis.from_url(settings.redis.URL, decode_responses=True)  # type: ignore[no-untyped-call]

    async def get(self, key: str) -> str | None:
        """Get value by key. Returns None if not found or expired."""
        return await self._client.get(key)  # type: ignore[no-any-return]

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Set value with TTL in seconds."""
        await self._client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(await self._client.exists(key))

    async def close(self) -> None:
        """Close the connection."""
        await self._client.close()
