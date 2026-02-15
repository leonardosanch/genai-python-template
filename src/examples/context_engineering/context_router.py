"""
Context Router Pattern.

Demonstrates:
- Intent-based context selection
- Multi-source context aggregation
- Relevance scoring
- Context caching

Run: uv run python -m src.examples.context_engineering.context_router
"""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class ContextSource(BaseModel):
    """A source of context information."""

    name: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_type: str  # "database", "vector_store", "api", etc.


class ContextRouter:
    """Route queries to appropriate context sources."""

    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.cache: dict[str, list[ContextSource]] = {}

    async def route_context(
        self,
        query: str,
        available_sources: list[str],
    ) -> list[ContextSource]:
        """
        Route query to relevant context sources.

        Args:
            query: User query
            available_sources: List of available source names

        Returns:
            List of ContextSource objects sorted by relevance
        """
        # Check cache
        cache_key = f"{query}:{','.join(sorted(available_sources))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Determine intent
        intent = await self._classify_intent(query)

        # Select sources based on intent
        sources = []

        if intent == "factual":
            sources.append(
                ContextSource(
                    name="knowledge_base",
                    content="Factual information from knowledge base...",
                    relevance_score=0.9,
                    source_type="database",
                )
            )
        elif intent == "conversational":
            sources.append(
                ContextSource(
                    name="conversation_history",
                    content="Previous conversation context...",
                    relevance_score=0.8,
                    source_type="memory",
                )
            )

        # Score and sort
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        # Cache result
        self.cache[cache_key] = sources

        return sources

    async def _classify_intent(self, query: str) -> str:
        """Classify query intent."""
        prompt = f"""Classify this query intent as one of: factual, conversational, procedural

Query: {query}

Intent:"""

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )

        return (response.choices[0].message.content or "conversational").strip().lower()


async def route_context(query: str) -> list[ContextSource]:
    """Convenience function for context routing."""
    client = AsyncOpenAI()
    router = ContextRouter(client)

    return await router.route_context(
        query,
        available_sources=["knowledge_base", "conversation_history", "web_search"],
    )


async def main() -> None:
    """Example usage of context router."""
    client = AsyncOpenAI()
    router = ContextRouter(client)

    print("🧭 Context Router Example\n")

    queries = [
        "What is the capital of France?",
        "How are you doing today?",
        "How do I install Python?",
    ]

    for query in queries:
        print(f"Query: {query}")
        sources = await router.route_context(query, ["knowledge_base", "conversation_history"])

        print("  Selected sources:")
        for source in sources:
            print(f"    - {source.name} (relevance: {source.relevance_score:.2f})")
        print()


if __name__ == "__main__":
    asyncio.run(main())
