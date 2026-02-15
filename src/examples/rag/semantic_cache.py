"""
Semantic Cache Example

Demonstrates:
- Vector-based response caching
- Similarity threshold matching
- Cost reduction through caching
- Cache invalidation strategies

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.rag.semantic_cache
"""

import asyncio
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, cast

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from pydantic import BaseModel


class CachedResponse(BaseModel):
    """A cached LLM response."""

    query: str
    response: str
    created_at: datetime
    hits: int = 0
    cost_saved: float = 0.0


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Uses vector similarity to match queries, not exact string matching.
    This allows cache hits for semantically similar queries.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 24,
    ):
        """
        Initialize semantic cache.

        Args:
            similarity_threshold: Minimum similarity for cache hit (0-1)
            ttl_hours: Time-to-live for cached responses
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"

        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

        # ChromaDB for cache storage
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = self.chroma_client.get_or_create_collection(name="semantic_cache")

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_saved = 0.0

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=[text],
        )
        return response.data[0].embedding

    def _generate_id(self, query: str) -> str:
        """Generate unique ID for query."""
        return hashlib.md5(query.encode()).hexdigest()

    async def get(self, query: str) -> str | None:
        """
        Get cached response for query.

        Args:
            query: User query

        Returns:
            Cached response if found, None otherwise
        """
        # Embed query
        query_embedding = await self.embed_text(query)

        # Search cache
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=1,
        )

        # Check if we have results
        if (
            not results["ids"]
            or not results["ids"][0]
            or not results["metadatas"]
            or not results["metadatas"][0]
            or not results["documents"]
            or not results["documents"][0]
        ):
            self.cache_misses += 1
            return None

        # Check similarity threshold
        distance = results["distances"][0][0] if results["distances"] else 1.0
        similarity = 1.0 - distance

        if similarity < self.similarity_threshold:
            self.cache_misses += 1
            return None

        # Check TTL
        # Ensure metadata is typed correctly for access
        metadata: dict[str, Any] = cast(dict[str, Any], results["metadatas"][0][0])
        created_at_str = metadata.get("created_at")
        if not created_at_str:
            self.cache_misses += 1
            return None
        created_at = datetime.fromisoformat(str(created_at_str))

        if datetime.now() - created_at > self.ttl:
            # Expired, remove from cache
            doc_id = results["ids"][0][0]
            self.collection.delete(ids=[doc_id])
            self.cache_misses += 1
            return None

        # Cache hit!
        self.cache_hits += 1
        cached_response = str(results["documents"][0][0])

        # Update stats
        cost_per_call = 0.0001  # Estimated cost
        self.total_cost_saved += cost_per_call

        # Update hit count in metadata
        doc_id = results["ids"][0][0]
        hits = int(str(metadata.get("hits", 0))) + 1
        metadata["hits"] = hits
        metadata["last_accessed"] = datetime.now().isoformat()

        # Convert all metadata values to string as ChromaDB expects dict[str, str]
        str_metadata = {k: str(v) for k, v in metadata.items()}

        self.collection.update(
            ids=[doc_id],
            metadatas=[str_metadata],
        )

        print(
            f"✅ Cache HIT (similarity: {similarity:.3f}, "
            f"hits: {hits}, saved: ${cost_per_call:.6f})"
        )

        return cached_response

    async def set(self, query: str, response: str) -> None:
        """
        Cache a response.

        Args:
            query: User query
            response: LLM response to cache
        """
        # Generate embedding
        query_embedding = await self.embed_text(query)

        # Generate ID
        doc_id = self._generate_id(query)

        # Metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "hits": 0,
            "query": query[:100],  # Store truncated query for debugging
        }

        # Add to cache
        self.collection.upsert(
            ids=[doc_id],
            documents=[response],
            embeddings=cast(Any, [query_embedding]),
            metadatas=[cast(Any, metadata)],
        )

    async def generate_with_cache(self, query: str) -> str:
        """
        Generate response with semantic caching.

        Args:
            query: User query

        Returns:
            LLM response (cached or fresh)
        """
        # Try cache first
        cached = await self.get(query)
        if cached:
            return cached

        # Cache miss - generate fresh response
        print("❌ Cache MISS - generating fresh response...")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content or ""

        # Cache the response
        await self.set(query, answer)

        return answer

    def get_stats(self) -> dict[str, float]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "total_cost_saved": self.total_cost_saved,
        }

    def clear(self) -> None:
        """Clear all cached responses."""
        self.chroma_client.delete_collection("semantic_cache")
        self.collection = self.chroma_client.get_or_create_collection(name="semantic_cache")
        print("Cache cleared")


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Semantic Cache Example")
    print("=" * 60)

    # Initialize cache
    cache = SemanticCache(
        similarity_threshold=0.92,  # 92% similarity required
        ttl_hours=24,
    )

    # Test queries - some are semantically similar
    queries = [
        # Original query
        "What is Clean Architecture?",
        # Semantically similar (should hit cache)
        "Can you explain Clean Architecture?",
        "Tell me about Clean Architecture",
        # Different query (should miss cache)
        "What is microservices architecture?",
        # Similar to previous (should hit cache)
        "Explain microservices architecture",
        # Back to first topic (should hit cache)
        "What's Clean Architecture about?",
    ]

    print("\nRunning queries with semantic caching...\n")

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 60)

        response = await cache.generate_with_cache(query)
        print(f"Response: {response[:100]}...")

        # Small delay to see the effect
        await asyncio.sleep(0.5)

    # Show statistics
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)

    stats = cache.get_stats()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Misses: {stats['cache_misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Total Cost Saved: ${stats['total_cost_saved']:.6f}")

    # Demonstrate cache clearing
    print("\n" + "=" * 60)
    print("Cache Management")
    print("=" * 60)

    print(f"\nCache size before clear: {cache.collection.count()} entries")
    cache.clear()
    print(f"Cache size after clear: {cache.collection.count()} entries")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Benefits:")
    print("✅ Semantic matching (not exact string matching)")
    print("✅ Reduced API costs")
    print("✅ Faster response times")
    print("✅ Automatic TTL expiration")
    print("✅ Hit statistics tracking")


if __name__ == "__main__":
    asyncio.run(main())
