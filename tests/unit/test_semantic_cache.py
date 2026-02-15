# tests/unit/test_semantic_cache.py
"""Tests for semantic cache."""

from unittest.mock import AsyncMock

import pytest

from src.domain.entities.document import Document
from src.infrastructure.cache.semantic_cache import SemanticCache, SemanticCacheConfig


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_redis() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def cache(mock_vector_store: AsyncMock, mock_redis: AsyncMock) -> SemanticCache:
    config = SemanticCacheConfig(similarity_threshold=0.9, ttl=600)
    return SemanticCache(
        vector_store=mock_vector_store,
        redis_cache=mock_redis,
        config=config,
    )


class TestSemanticCacheGet:
    async def test_cache_hit(
        self, cache: SemanticCache, mock_vector_store: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_vector_store.search.return_value = [
            Document(content="cached query", id="q1", score=0.95, metadata={})
        ]
        mock_redis.get.return_value = "cached answer"

        result = await cache.get("similar query")
        assert result == "cached answer"

    async def test_cache_miss_low_score(
        self, cache: SemanticCache, mock_vector_store: AsyncMock
    ) -> None:
        mock_vector_store.search.return_value = [
            Document(content="different query", id="q1", score=0.5, metadata={})
        ]

        result = await cache.get("very different query")
        assert result is None

    async def test_cache_miss_no_results(
        self, cache: SemanticCache, mock_vector_store: AsyncMock
    ) -> None:
        mock_vector_store.search.return_value = []
        result = await cache.get("new query")
        assert result is None

    async def test_cache_miss_redis_empty(
        self, cache: SemanticCache, mock_vector_store: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_vector_store.search.return_value = [
            Document(content="q", id="q1", score=0.95, metadata={})
        ]
        mock_redis.get.return_value = None

        result = await cache.get("query")
        assert result is None

    async def test_handles_vector_store_error(
        self, cache: SemanticCache, mock_vector_store: AsyncMock
    ) -> None:
        mock_vector_store.search.side_effect = RuntimeError("connection lost")
        result = await cache.get("query")
        assert result is None


class TestSemanticCacheSet:
    async def test_set_stores_in_both(
        self, cache: SemanticCache, mock_vector_store: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        await cache.set("my query", "my answer", doc_id="q1")
        mock_vector_store.upsert.assert_called_once()
        mock_redis.set.assert_called_once()

    async def test_set_with_auto_id(
        self, cache: SemanticCache, mock_vector_store: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        await cache.set("my query", "my answer")
        mock_vector_store.upsert.assert_called_once()
        mock_redis.set.assert_called_once()

    async def test_set_handles_error(
        self, cache: SemanticCache, mock_vector_store: AsyncMock
    ) -> None:
        mock_vector_store.upsert.side_effect = RuntimeError("fail")
        # Should not raise
        await cache.set("query", "answer")


class TestSemanticCacheInvalidate:
    async def test_invalidate(
        self, cache: SemanticCache, mock_vector_store: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        await cache.invalidate("q1")
        mock_redis.delete.assert_called_once()
        mock_vector_store.delete.assert_called_once_with(["q1"])
