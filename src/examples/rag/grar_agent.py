"""
GRaR (Graph-Retrieval-Augmented Reasoning) Example

Demonstrates:
- Multi-hop reasoning over graphs
- Agent-driven graph exploration
- Causal reasoning
- Impact analysis
- Dynamic query planning

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.rag.grar_agent
"""

import asyncio
import os

from langchain_core.messages import SystemMessage  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from pydantic import BaseModel


class GraphNode(BaseModel):
    """Graph node."""

    id: str
    type: str
    properties: dict[str, str] = {}


class GraphEdge(BaseModel):
    """Graph edge."""

    source: str
    target: str
    relation: str


class SimpleGraph:
    """Simple in-memory graph."""

    def __init__(self) -> None:
        """Initialize graph."""
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        """Add node."""
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """Add edge."""
        self.edges.append(edge)

    def get_neighbors(self, node_id: str) -> list[tuple[str, str]]:
        """Get neighbors with relations."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                neighbors.append((edge.target, edge.relation))
            elif edge.target == node_id:
                neighbors.append((edge.source, f"inverse_{edge.relation}"))
        return neighbors

    def get_path(self, start: str, end: str, max_hops: int = 3) -> list[str]:
        """Find path between nodes."""
        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)

            if current == end:
                return path

            if len(path) > max_hops:
                continue

            if current in visited:
                continue

            visited.add(current)

            for neighbor, _ in self.get_neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return []


class GRaRAgent:
    """GRaR agent for graph reasoning."""

    def __init__(self, graph: SimpleGraph):
        """Initialize agent."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.graph = graph

    async def plan_exploration(self, question: str) -> list[str]:
        """Plan graph exploration."""
        print("\n🤔 Planning exploration...")

        prompt = f"""Given this question, identify key entities to explore:

Question: {question}

Available entities: {", ".join(self.graph.nodes.keys())}

List 2-3 entities to start exploration (comma-separated):"""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])

        content = response.content or ""
        entities = [e.strip() for e in content.split(",") if e.strip() in self.graph.nodes]

        if not entities:
            entities = list(self.graph.nodes.keys())[:2]

        print(f"   Starting entities: {entities}")
        return entities

    async def explore_node(self, node_id: str) -> str:
        """Explore a node and its neighborhood."""
        print(f"\n🔍 Exploring: {node_id}")

        if node_id not in self.graph.nodes:
            return f"Node {node_id} not found"

        node = self.graph.nodes[node_id]
        neighbors = self.graph.get_neighbors(node_id)

        info = [f"Node: {node.id} (type: {node.type})"]

        if neighbors:
            info.append("Connections:")
            for neighbor, relation in neighbors[:5]:  # Limit to 5
                neighbor_node = self.graph.nodes.get(neighbor)
                if neighbor_node:
                    info.append(f"  -{relation}-> {neighbor} ({neighbor_node.type})")

        result = "\n".join(info)
        print(f"   Found {len(neighbors)} connections")

        return result

    async def reason(self, question: str, exploration_results: list[str]) -> str:
        """Reason over exploration results."""
        print("\n💡 Reasoning over graph...")

        context = "\n\n".join(exploration_results)

        prompt = f"""Answer using graph exploration results:

Graph Exploration:
{context}

Question: {question}

Provide reasoning and answer:"""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])

        return response.content or ""

    async def query(self, question: str) -> str:
        """
        Query with graph reasoning.

        Args:
            question: User question

        Returns:
            Answer with reasoning
        """
        print(f"\n{'=' * 60}")
        print(f"GRaR Query: {question}")
        print(f"{'=' * 60}")

        # 1. Plan exploration
        start_entities = await self.plan_exploration(question)

        # 2. Explore graph
        exploration_results = []
        for entity in start_entities:
            result = await self.explore_node(entity)
            exploration_results.append(result)

            # Explore neighbors
            neighbors = self.graph.get_neighbors(entity)
            for neighbor, _ in neighbors[:2]:  # Explore 2 neighbors
                neighbor_result = await self.explore_node(neighbor)
                exploration_results.append(neighbor_result)

        # 3. Reason over results
        answer = await self.reason(question, exploration_results)

        return answer


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("GRaR (Graph-Retrieval-Augmented Reasoning) Example")
    print("=" * 60)

    # Build sample graph
    graph = SimpleGraph()

    # Add nodes
    nodes = [
        GraphNode(id="RAG", type="Concept"),
        GraphNode(id="Retrieval", type="Concept"),
        GraphNode(id="Generation", type="Concept"),
        GraphNode(id="VectorDB", type="Technology"),
        GraphNode(id="LLM", type="Technology"),
        GraphNode(id="Embeddings", type="Concept"),
    ]

    for node in nodes:
        graph.add_node(node)

    # Add edges
    edges = [
        GraphEdge(source="RAG", target="Retrieval", relation="USES"),
        GraphEdge(source="RAG", target="Generation", relation="USES"),
        GraphEdge(source="Retrieval", target="VectorDB", relation="REQUIRES"),
        GraphEdge(source="Retrieval", target="Embeddings", relation="USES"),
        GraphEdge(source="Generation", target="LLM", relation="REQUIRES"),
        GraphEdge(source="VectorDB", target="Embeddings", relation="STORES"),
    ]

    for edge in edges:
        graph.add_edge(edge)

    # Create agent
    agent = GRaRAgent(graph)

    # Example 1: Simple query
    answer1 = await agent.query("How does RAG work?")
    print(f"\n{'=' * 60}")
    print("Answer:")
    print(f"{'=' * 60}")
    print(answer1)

    # Example 2: Multi-hop query
    print("\n\n")
    answer2 = await agent.query("What technologies does RAG depend on?")
    print(f"\n{'=' * 60}")
    print("Answer:")
    print(f"{'=' * 60}")
    print(answer2)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Agent-driven graph exploration")
    print("✅ Multi-hop reasoning")
    print("✅ Dynamic query planning")
    print("✅ Relationship-aware answers")
    print("\nAdvanced Features:")
    print("  - Causal reasoning")
    print("  - Impact analysis")
    print("  - Path finding")
    print("  - Subgraph extraction")


if __name__ == "__main__":
    asyncio.run(main())
