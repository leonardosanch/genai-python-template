# src/infrastructure/cache/semantic_cache.py
"""Semantic cache combining vector similarity with key-value storage."""

from dataclasses import dataclass

import structlog

from src.domain.entities.document import Document
from src.domain.ports.vector_store_port import VectorStorePort
from src.infrastructure.cache.redis_cache import RedisCache

logger = structlog.get_logger(__name__)


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache."""

    similarity_threshold: float = 0.92
    ttl: int = 3600
    cache_prefix: str = "sem_cache:"


class SemanticCache:
    """Cache that uses vector similarity to match semantically equivalent queries.

    Combines a vector store (for similarity search) with Redis (for answer storage).
    On cache hit: finds similar query via vector search → retrieves cached answer.
    On cache set: stores query embedding in vector store + answer in Redis.
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        redis_cache: RedisCache,
        config: SemanticCacheConfig | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._redis = redis_cache
        self._config = config or SemanticCacheConfig()

    async def get(self, query: str) -> str | None:
        """Look up a cached answer for a semantically similar query.

        Returns None on cache miss.
        """
        try:
            results = await self._vector_store.search(query, top_k=1)
            if not results:
                return None

            best = results[0]
            if best.score is not None and best.score >= self._config.similarity_threshold:
                cache_key = f"{self._config.cache_prefix}{best.id}"
                cached = await self._redis.get(cache_key)
                if cached:
                    logger.info("semantic_cache_hit", query=query[:50], score=best.score)
                    return cached
        except Exception:
            logger.warning("semantic_cache_get_error", query=query[:50])
        return None

    async def set(self, query: str, answer: str, doc_id: str | None = None) -> None:
        """Store a query-answer pair in the semantic cache."""
        try:
            cache_id = doc_id or query[:64]
            doc = Document(content=query, id=cache_id, metadata={"type": "cache_query"})
            await self._vector_store.upsert([doc])

            cache_key = f"{self._config.cache_prefix}{cache_id}"
            await self._redis.set(cache_key, answer, ttl=self._config.ttl)
            logger.info("semantic_cache_set", query=query[:50], cache_id=cache_id)
        except Exception:
            logger.warning("semantic_cache_set_error", query=query[:50])

    async def invalidate(self, doc_id: str) -> None:
        """Remove a specific entry from the cache."""
        try:
            cache_key = f"{self._config.cache_prefix}{doc_id}"
            await self._redis.delete(cache_key)
            await self._vector_store.delete([doc_id])
        except Exception:
            logger.warning("semantic_cache_invalidate_error", doc_id=doc_id)
