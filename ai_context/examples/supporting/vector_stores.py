"""
Vector Stores Example

Demonstrates:
- Multiple vector store backends
- Unified interface
- Performance comparison
- Migration patterns

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.supporting.vector_stores
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from pydantic import BaseModel


class Document(BaseModel):
    """Document with embedding."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, str] = {}


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add(self, documents: list[Document]) -> None:
        """Add documents."""
        ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int) -> list[Document]:
        """Search by embedding."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get document count."""
        ...


class ChromaDBStore(VectorStore):
    """ChromaDB vector store."""

    def __init__(self, collection_name: str = "default"):
        """Initialize ChromaDB."""
        self.client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection_name)

    async def add(self, documents: list[Document]) -> None:
        """Add documents."""
        self.collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    async def search(self, query_embedding: list[float], top_k: int) -> list[Document]:
        """Search."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append(
                    Document(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] or {},
                    )
                )

        return docs

    async def count(self) -> int:
        """Count."""
        return self.collection.count()


class InMemoryStore(VectorStore):
    """Simple in-memory vector store."""

    def __init__(self):
        """Initialize in-memory store."""
        self.documents: dict[str, Document] = {}

    async def add(self, documents: list[Document]) -> None:
        """Add documents."""
        for doc in documents:
            self.documents[doc.id] = doc

    async def search(self, query_embedding: list[float], top_k: int) -> list[Document]:
        """Search by cosine similarity."""
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        # Score all documents
        scored = [
            (doc, cosine_similarity(query_embedding, doc.embedding or []))
            for doc in self.documents.values()
            if doc.embedding
        ]

        # Sort and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    async def count(self) -> int:
        """Count."""
        return len(self.documents)


async def benchmark_store(store: VectorStore, name: str, embedder) -> dict[str, float]:
    """Benchmark vector store."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    # Prepare documents
    docs = [
        Document(
            id=f"doc_{i}",
            content=f"Document {i} about vector stores",
            metadata={"index": str(i)},
        )
        for i in range(100)
    ]

    # Generate embeddings
    embeddings = await embedder.embed([doc.content for doc in docs])
    for doc, emb in zip(docs, embeddings):
        doc.embedding = emb

    # Benchmark add
    start = time.time()
    await store.add(docs)
    add_time = time.time() - start

    # Benchmark search
    query_emb = embeddings[0]
    start = time.time()
    results = await store.search(query_emb, top_k=10)
    search_time = time.time() - start

    # Count
    count = await store.count()

    print(f"  Add time:    {add_time * 1000:.2f}ms")
    print(f"  Search time: {search_time * 1000:.2f}ms")
    print(f"  Count:       {count}")
    print(f"  Results:     {len(results)}")

    return {
        "add_time_ms": add_time * 1000,
        "search_time_ms": search_time * 1000,
        "count": count,
    }


class Embedder:
    """Embedding service."""

    def __init__(self):
        """Initialize embedder."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Vector Stores Example")
    print("=" * 60)

    embedder = Embedder()

    # Benchmark ChromaDB
    chroma = ChromaDBStore("benchmark")
    chroma_metrics = await benchmark_store(chroma, "ChromaDB", embedder)

    # Benchmark In-Memory
    memory = InMemoryStore()
    memory_metrics = await benchmark_store(memory, "In-Memory", embedder)

    # Comparison
    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")
    print(f"{'Metric':<20} {'ChromaDB':>15} {'In-Memory':>15}")
    print("-" * 60)
    print(
        f"{'Add Time (ms)':<20} {chroma_metrics['add_time_ms']:>15.2f} "
        f"{memory_metrics['add_time_ms']:>15.2f}"
    )
    print(
        f"{'Search Time (ms)':<20} {chroma_metrics['search_time_ms']:>15.2f} "
        f"{memory_metrics['search_time_ms']:>15.2f}"
    )

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Unified vector store interface")
    print("✅ Multiple backends (ChromaDB, In-Memory)")
    print("✅ Performance benchmarking")
    print("✅ Easy migration between stores")


if __name__ == "__main__":
    asyncio.run(main())
