"""
Modular RAG Example

Demonstrates:
- Composable RAG pipeline
- Pluggable components
- Multiple retrievers
- Custom rerankers
- Pipeline flexibility

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.rag.modular_rag
"""

import asyncio
import os
from typing import Protocol

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from pydantic import BaseModel

# Component Protocols


class Document(BaseModel):
    """A document with metadata."""

    id: str
    content: str
    metadata: dict[str, str] = {}
    score: float = 0.0


class Retriever(Protocol):
    """Retriever interface."""

    async def retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents."""
        ...


class Reranker(Protocol):
    """Reranker interface."""

    async def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Rerank documents."""
        ...


class Generator(Protocol):
    """Generator interface."""

    async def generate(self, query: str, context: list[Document]) -> str:
        """Generate answer."""
        ...


# Retrievers


class VectorRetriever:
    """Vector-based retriever."""

    def __init__(self, collection: chromadb.Collection, embedder):
        """Initialize retriever."""
        self.collection = collection
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve using vector similarity."""
        # Embed query
        embedding = await self.embedder.embed([query])

        # Search
        results = self.collection.query(
            query_embeddings=embedding,  # type: ignore
            n_results=top_k,
        )

        # Convert to documents
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta: dict[str, str] = dict(results["metadatas"][0][i] or {})  # type: ignore
                docs.append(
                    Document(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=meta,
                        score=1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                    )
                )

        return docs


class HybridRetriever:
    """Hybrid retriever combining multiple strategies."""

    def __init__(self, retrievers: list[Retriever]):
        """Initialize with multiple retrievers."""
        self.retrievers = retrievers

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve using all retrievers and merge."""
        all_docs: dict[str, Document] = {}

        # Retrieve from all retrievers
        for retriever in self.retrievers:
            docs = await retriever.retrieve(query, top_k)
            for doc in docs:
                if doc.id not in all_docs:
                    all_docs[doc.id] = doc
                else:
                    # Average scores
                    existing = all_docs[doc.id]
                    existing.score = (existing.score + doc.score) / 2

        # Sort by score
        sorted_docs = sorted(all_docs.values(), key=lambda d: d.score, reverse=True)
        return sorted_docs[:top_k]


# Rerankers


class ScoreReranker:
    """Simple score-based reranker."""

    async def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Rerank by existing scores."""
        return sorted(documents, key=lambda d: d.score, reverse=True)[:top_k]


class LLMReranker:
    """LLM-based reranker."""

    def __init__(self, llm: AsyncOpenAI):
        """Initialize with LLM."""
        self.llm = llm

    async def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Rerank using LLM relevance scoring."""
        # For each document, score relevance
        scored_docs = []

        for doc in documents:
            prompt = f"""Rate relevance (0-10):

Query: {query}
Document: {doc.content[:200]}...

Score (0-10):"""

            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )

            try:
                score = float(response.choices[0].message.content or "5")
                doc.score = score / 10.0  # Normalize to 0-1
            except ValueError:
                doc.score = 0.5

            scored_docs.append(doc)

        # Sort and return top_k
        return sorted(scored_docs, key=lambda d: d.score, reverse=True)[:top_k]


# Generators


class SimpleGenerator:
    """Simple context-based generator."""

    def __init__(self, llm: AsyncOpenAI):
        """Initialize generator."""
        self.llm = llm

    async def generate(self, query: str, context: list[Document]) -> str:
        """Generate answer from context."""
        context_str = "\n\n".join(f"[{i + 1}]: {doc.content}" for i, doc in enumerate(context))

        prompt = f"""Answer based on context.

Context:
{context_str}

Question: {query}

Answer:"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content or ""


# Embedder


class Embedder:
    """Embedding service."""

    def __init__(self, client: AsyncOpenAI):
        """Initialize embedder."""
        self.client = client
        self.model = "text-embedding-3-small"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# Modular RAG Pipeline


class ModularRAG:
    """Composable RAG pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
    ):
        """
        Initialize modular RAG.

        Args:
            retriever: Document retriever
            reranker: Document reranker
            generator: Answer generator
        """
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    async def query(self, question: str, top_k: int = 3) -> str:
        """
        Query pipeline.

        Args:
            question: User question
            top_k: Number of documents to use

        Returns:
            Generated answer
        """
        # 1. Retrieve
        print(f"\n1. Retrieving (top_k={top_k * 2})...")
        docs = await self.retriever.retrieve(question, top_k=top_k * 2)
        print(f"   Retrieved {len(docs)} documents")

        # 2. Rerank
        print(f"\n2. Reranking to top {top_k}...")
        reranked = await self.reranker.rerank(question, docs, top_k=top_k)
        print(f"   Reranked to {len(reranked)} documents")

        # 3. Generate
        print("\n3. Generating answer...")
        answer = await self.generator.generate(question, reranked)

        return answer


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Modular RAG Example")
    print("=" * 60)

    # Initialize components
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    client = AsyncOpenAI(api_key=api_key)
    embedder = Embedder(client)

    # Setup ChromaDB
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
    collection = chroma_client.get_or_create_collection("modular_rag")

    # Add sample documents
    docs = [
        "Modular RAG uses composable components for flexibility.",
        "Retrievers can be swapped without changing the pipeline.",
        "Rerankers improve relevance of retrieved documents.",
        "Generators create answers from retrieved context.",
    ]

    embeddings = await embedder.embed(docs)
    collection.add(
        ids=[f"doc_{i}" for i in range(len(docs))],
        documents=docs,
        embeddings=embeddings,  # type: ignore
    )

    # Example 1: Simple pipeline
    print("\n\nExample 1: Simple Pipeline")
    print("-" * 60)

    retriever = VectorRetriever(collection, embedder)
    reranker = ScoreReranker()
    generator = SimpleGenerator(client)

    rag = ModularRAG(retriever, reranker, generator)

    answer = await rag.query("What is modular RAG?", top_k=2)
    print(f"\nAnswer: {answer}")

    # Example 2: With LLM reranker
    print("\n\nExample 2: With LLM Reranker")
    print("-" * 60)

    llm_reranker = LLMReranker(client)
    rag2 = ModularRAG(retriever, llm_reranker, generator)

    answer2 = await rag2.query("How do rerankers work?", top_k=2)
    print(f"\nAnswer: {answer2}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Composable pipeline with pluggable components")
    print("✅ Multiple retriever strategies")
    print("✅ Custom rerankers (score-based, LLM-based)")
    print("✅ Flexible generator")
    print("✅ Easy to swap components")


if __name__ == "__main__":
    asyncio.run(main())
