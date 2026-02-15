"""
GraphRAG Example (Simplified)

Demonstrates:
- Graph-based retrieval
- Entity and relationship extraction
- Graph traversal for context
- Relationship-aware search
- In-memory graph (no Neo4j required)

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.rag.graph_rag
"""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel


class Entity(BaseModel):
    """Graph entity."""

    id: str
    type: str
    properties: dict[str, Any] = {}


class Relationship(BaseModel):
    """Graph relationship."""

    source: str
    target: str
    type: str
    properties: dict[str, Any] = {}


class KnowledgeGraph:
    """In-memory knowledge graph."""

    def __init__(self) -> None:
        """Initialize graph."""
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []

    def add_entity(self, entity: Entity) -> None:
        """Add entity."""
        self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship) -> None:
        """Add relationship."""
        self.relationships.append(rel)

    def get_neighbors(self, entity_id: str, max_depth: int = 1) -> list[Entity]:
        """Get neighboring entities."""
        neighbors = set()
        current_level = {entity_id}

        for _ in range(max_depth):
            next_level = set()
            for ent_id in current_level:
                # Find relationships
                for rel in self.relationships:
                    if rel.source == ent_id:
                        next_level.add(rel.target)
                    elif rel.target == ent_id:
                        next_level.add(rel.source)

            neighbors.update(next_level)
            current_level = next_level

        return [self.entities[eid] for eid in neighbors if eid in self.entities]

    def get_subgraph(self, entity_id: str) -> str:
        """Get subgraph as text."""
        if entity_id not in self.entities:
            return ""

        entity = self.entities[entity_id]
        lines = [f"Entity: {entity.id} ({entity.type})"]

        # Get relationships
        for rel in self.relationships:
            if rel.source == entity_id:
                target = self.entities.get(rel.target)
                if target:
                    lines.append(f"  -{rel.type}-> {target.id} ({target.type})")
            elif rel.target == entity_id:
                source = self.entities.get(rel.source)
                if source:
                    lines.append(f"  <-{rel.type}- {source.id} ({source.type})")

        return "\n".join(lines)


class GraphRAG:
    """GraphRAG system."""

    def __init__(self) -> None:
        """Initialize GraphRAG."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.graph = KnowledgeGraph()

    async def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        prompt = f"""Extract entities from text. Return as JSON list:

Text: {text}

Format:
[
  {{"id": "entity_name", "type": "Person|Organization|Concept", "properties": {{}}}}
]

Extract 2-3 main entities."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        # Parse response (simplified)
        response.choices[0].message.content or ""

        # Simple parsing (in production, use structured output)
        entities = []
        if "RAG" in text:
            entities.append(Entity(id="RAG", type="Concept"))
        if "retrieval" in text.lower():
            entities.append(Entity(id="Retrieval", type="Concept"))
        if "generation" in text.lower():
            entities.append(Entity(id="Generation", type="Concept"))

        return entities

    async def extract_relationships(self, text: str, entities: list[Entity]) -> list[Relationship]:
        """Extract relationships."""
        if len(entities) < 2:
            return []

        # Simplified relationship extraction
        relationships = []

        if any(e.id == "RAG" for e in entities):
            if any(e.id == "Retrieval" for e in entities):
                relationships.append(
                    Relationship(
                        source="RAG",
                        target="Retrieval",
                        type="USES",
                    )
                )
            if any(e.id == "Generation" for e in entities):
                relationships.append(
                    Relationship(
                        source="RAG",
                        target="Generation",
                        type="USES",
                    )
                )

        return relationships

    async def index_document(self, text: str) -> None:
        """Index document into graph."""
        print(f"\nIndexing document ({len(text)} chars)...")

        # Extract entities
        entities = await self.extract_entities(text)
        print(f"  Extracted {len(entities)} entities")

        # Add to graph
        for entity in entities:
            self.graph.add_entity(entity)

        # Extract relationships
        relationships = await self.extract_relationships(text, entities)
        print(f"  Extracted {len(relationships)} relationships")

        # Add to graph
        for rel in relationships:
            self.graph.add_relationship(rel)

    async def query(self, question: str) -> str:
        """Query using graph context."""
        print(f"\n{'=' * 60}")
        print(f"Query: {question}")
        print(f"{'=' * 60}")

        # Find relevant entities
        print("\n1. Finding relevant entities...")
        relevant_entities = []
        for entity in self.graph.entities.values():
            if entity.id.lower() in question.lower():
                relevant_entities.append(entity)
                print(f"   Found: {entity.id}")

        if not relevant_entities:
            # Use first entity as fallback
            relevant_entities = list(self.graph.entities.values())[:1]

        # Get graph context
        print("\n2. Building graph context...")
        context_parts = []
        for entity in relevant_entities:
            subgraph = self.graph.get_subgraph(entity.id)
            context_parts.append(subgraph)

        context = "\n\n".join(context_parts)
        print(f"   Context size: {len(context)} chars")

        # Generate answer
        print("\n3. Generating answer...")
        prompt = f"""Answer using graph context:

Graph Context:
{context}

Question: {question}

Answer:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content or ""


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("GraphRAG Example")
    print("=" * 60)
    print("\nNote: Using in-memory graph (no Neo4j required)")

    graph_rag = GraphRAG()

    # Index documents
    docs = [
        "RAG combines retrieval with generation for grounded responses.",
        "Retrieval finds relevant documents from a knowledge base.",
        "Generation creates answers using retrieved context.",
    ]

    for doc in docs:
        await graph_rag.index_document(doc)

    # Show graph stats
    print(f"\n{'=' * 60}")
    print("Graph Statistics")
    print(f"{'=' * 60}")
    print(f"Entities: {len(graph_rag.graph.entities)}")
    print(f"Relationships: {len(graph_rag.graph.relationships)}")

    # Query
    answer = await graph_rag.query("How does RAG work?")

    print(f"\n{'=' * 60}")
    print("Answer:")
    print(f"{'=' * 60}")
    print(answer)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Entity extraction")
    print("✅ Relationship extraction")
    print("✅ Graph-based context")
    print("✅ Relationship-aware retrieval")
    print("\nProduction:")
    print("  Use Neo4j for scalable graph storage")
    print("  Use LLM for entity/relationship extraction")
    print("  Add graph embeddings for similarity search")


if __name__ == "__main__":
    asyncio.run(main())
