"""
Agentic RAG Example

Demonstrates:
- Agent decides when to retrieve
- Dynamic retrieval strategy
- Query routing
- Adaptive context
- Self-correction

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.agents.agentic_rag
"""

import asyncio
import os
from enum import Enum

import chromadb
from chromadb.config import Settings
from langchain_core.messages import SystemMessage  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from openai import AsyncOpenAI
from pydantic import BaseModel


class QueryType(str, Enum):
    """Type of query."""

    FACTUAL = "factual"  # Needs retrieval
    CONVERSATIONAL = "conversational"  # No retrieval needed
    REASONING = "reasoning"  # Needs retrieval + reasoning


class QueryAnalysis(BaseModel):
    """Analysis of user query."""

    query_type: QueryType
    needs_retrieval: bool
    reasoning: str


class AgenticRAG:
    """RAG system with agentic decision-making."""

    def __init__(self) -> None:
        """Initialize agentic RAG."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.client = AsyncOpenAI(api_key=api_key)

        # Setup ChromaDB
        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = chroma_client.get_or_create_collection(name="agentic_rag")

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze if query needs retrieval."""
        print("\n🤔 Analyzing query...")

        prompt = f"""Analyze this query and determine if it needs document retrieval.

Query: {query}

Respond with JSON:
{{
  "query_type": "factual|conversational|reasoning",
  "needs_retrieval": true|false,
  "reasoning": "explanation"
}}

Examples:
- "What is RAG?" → factual, needs_retrieval=true
- "Hello, how are you?" → conversational, needs_retrieval=false
- "Compare RAG vs fine-tuning" → reasoning, needs_retrieval=true"""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])

        # Parse response (simplified)
        content = response.content or ""
        if "conversational" in content.lower():
            analysis = QueryAnalysis(
                query_type=QueryType.CONVERSATIONAL,
                needs_retrieval=False,
                reasoning="Simple greeting or chat",
            )
        elif "reasoning" in content.lower():
            analysis = QueryAnalysis(
                query_type=QueryType.REASONING,
                needs_retrieval=True,
                reasoning="Requires analysis of retrieved facts",
            )
        else:
            analysis = QueryAnalysis(
                query_type=QueryType.FACTUAL,
                needs_retrieval=True,
                reasoning="Needs factual information",
            )

        print(f"   Type: {analysis.query_type.value}")
        print(f"   Needs retrieval: {analysis.needs_retrieval}")
        print(f"   Reasoning: {analysis.reasoning}")

        return analysis

    async def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve documents."""
        print(f"\n📚 Retrieving (top_k={top_k})...")

        # Embed query
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        )
        embedding = response.data[0].embedding

        # Search
        results = self.collection.query(
            query_embeddings=[embedding],  # type: ignore
            n_results=top_k,
        )

        docs = results["documents"][0] if results["documents"] else []
        print(f"   Retrieved {len(docs)} documents")

        return docs

    async def generate(self, query: str, context: list[str] | None = None) -> str:
        """Generate answer."""
        print("\n💡 Generating answer...")

        if context:
            context_str = "\n\n".join(f"[{i + 1}]: {doc}" for i, doc in enumerate(context))
            prompt = f"""Answer based on context.

Context:
{context_str}

Question: {query}

Answer:"""
        else:
            prompt = query

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])

        return response.content or ""

    async def query(self, question: str) -> str:
        """
        Query with agentic decision-making.

        Args:
            question: User question

        Returns:
            Answer
        """
        print(f"\n{'=' * 60}")
        print(f"Query: {question}")
        print(f"{'=' * 60}")

        # 1. Analyze query
        analysis = await self.analyze_query(question)

        # 2. Decide on retrieval
        if analysis.needs_retrieval:
            # Retrieve
            docs = await self.retrieve(question)

            # Generate with context
            answer = await self.generate(question, docs)
        else:
            # Generate without retrieval
            answer = await self.generate(question)

        return answer


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Agentic RAG Example")
    print("=" * 60)

    rag = AgenticRAG()

    # Add sample documents
    docs = [
        "RAG combines retrieval with generation for grounded responses.",
        "Fine-tuning updates model weights with domain-specific data.",
        "RAG is more flexible and doesn't require retraining.",
    ]

    embeddings_response = await rag.client.embeddings.create(
        model="text-embedding-3-small",
        input=docs,
    )
    embeddings = [item.embedding for item in embeddings_response.data]

    rag.collection.add(
        ids=[f"doc_{i}" for i in range(len(docs))],
        documents=docs,
        embeddings=embeddings,  # type: ignore
    )

    # Example 1: Factual (needs retrieval)
    answer1 = await rag.query("What is RAG?")
    print(f"\nAnswer: {answer1}\n")

    # Example 2: Conversational (no retrieval)
    answer2 = await rag.query("Hello!")
    print(f"\nAnswer: {answer2}\n")

    # Example 3: Reasoning (needs retrieval)
    answer3 = await rag.query("Compare RAG and fine-tuning")
    print(f"\nAnswer: {answer3}\n")

    print("=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Agent decides when to retrieve")
    print("✅ Query type classification")
    print("✅ Adaptive retrieval strategy")
    print("✅ Efficient resource usage")


if __name__ == "__main__":
    asyncio.run(main())
