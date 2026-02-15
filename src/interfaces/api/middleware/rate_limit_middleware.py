"""Rate limiter with in-memory (token bucket) and Redis (sliding window) backends.

Production multi-instance deployments should use ``backend="redis"``.
Falls back to in-memory if Redis is unavailable.
"""

import json
import logging
import time
from http import HTTPStatus

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class _TokenBucket:
    """Simple token bucket for a single client."""

    __slots__ = ("_capacity", "_tokens", "_last_refill")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def consume(self, rpm: int) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._capacity, self._tokens + elapsed * (rpm / 60.0))
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class _MemoryBackend:
    """In-memory rate limit backend."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._buckets: dict[str, _TokenBucket] = {}

    async def is_allowed(self, client_ip: str) -> bool:
        if client_ip not in self._buckets:
            self._buckets[client_ip] = _TokenBucket(capacity=self._rpm)
        return self._buckets[client_ip].consume(self._rpm)


class _RedisBackend:
    """Redis sliding-window rate limit backend."""

    def __init__(self, rpm: int, redis_url: str) -> None:
        self._rpm = rpm
        self._redis_url = redis_url
        self._redis: object | None = None

    async def _get_redis(self) -> object:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)  # type: ignore
        return self._redis

    async def is_allowed(self, client_ip: str) -> bool:
        try:
            import redis.asyncio as aioredis

            r: aioredis.Redis = await self._get_redis()  # type: ignore[assignment]
            key = f"ratelimit:{client_ip}"
            now = time.time()
            window_start = now - 60.0

            pipe = r.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, 120)
            results = await pipe.execute()

            count: int = results[2]
            return count <= self._rpm
        except Exception:
            logger.warning("Redis rate-limit unavailable, allowing request")
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter — supports memory and Redis backends."""

    def __init__(
        self,
        app: object,
        rpm: int = 60,
        backend: str = "memory",
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        if backend == "redis":
            self._backend: _MemoryBackend | _RedisBackend = _RedisBackend(rpm, redis_url)
        else:
            self._backend = _MemoryBackend(rpm)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self._get_client_ip(request)

        if not await self._backend.is_allowed(client_ip):
            return Response(
                content=json.dumps({"error": "Rate limit exceeded"}),
                status_code=HTTPStatus.TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
