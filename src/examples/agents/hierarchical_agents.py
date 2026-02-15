"""
Hierarchical Agents Example

Demonstrates:
- Multi-level agent hierarchy
- Manager-worker delegation
- Specialized sub-teams
- Task decomposition
- Result aggregation

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.agents.hierarchical_agents
"""

import asyncio
import os
from typing import TypedDict

from langchain_core.messages import SystemMessage  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore


class TaskState(TypedDict):
    """State for hierarchical tasks."""

    task: str
    subtasks: list[str]
    results: dict[str, str]
    final_result: str


class Worker:
    """Base worker agent."""

    def __init__(self, name: str, specialty: str, llm: ChatOpenAI):
        """Initialize worker."""
        self.name = name
        self.specialty = specialty
        self.llm = llm

    async def execute(self, task: str) -> str:
        """Execute task."""
        print(f"  👷 {self.name} ({self.specialty}): working...")

        prompt = f"""You are {self.name}, specialized in {self.specialty}.

Task: {task}

Provide a focused response based on your specialty."""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content or ""


class Manager:
    """Manager coordinating workers."""

    def __init__(self, name: str, workers: list[Worker], llm: ChatOpenAI):
        """Initialize manager."""
        self.name = name
        self.workers = workers
        self.llm = llm

    async def delegate(self, task: str) -> dict[str, str]:
        """Delegate task to workers."""
        print(f"\n👔 {self.name}: delegating task...")

        results = {}
        for worker in self.workers:
            result = await worker.execute(task)
            results[worker.name] = result

        return results

    async def synthesize(self, results: dict[str, str]) -> str:
        """Synthesize worker results."""
        print(f"\n👔 {self.name}: synthesizing results...")

        results_str = "\n\n".join(f"{name}: {result}" for name, result in results.items())

        prompt = f"""Synthesize these worker results into a cohesive answer:

{results_str}

Provide a comprehensive, well-structured response."""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content or ""


class Director:
    """Top-level director managing managers."""

    def __init__(self, managers: list[Manager], llm: ChatOpenAI):
        """Initialize director."""
        self.managers = managers
        self.llm = llm

    async def decompose(self, task: str) -> list[str]:
        """Decompose task into subtasks."""
        print("\n🎯 Director: decomposing task...")

        prompt = f"""Break down this task into {len(self.managers)} subtasks:

Task: {task}

Provide {len(self.managers)} specific subtasks, one per line."""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])

        subtasks = [
            line.strip()
            for line in (response.content or "").split("\n")
            if line.strip() and not line.strip().startswith("#")
        ][: len(self.managers)]

        for i, subtask in enumerate(subtasks, 1):
            print(f"  {i}. {subtask}")

        return subtasks

    async def coordinate(self, task: str) -> str:
        """Coordinate managers on task."""
        # Decompose
        subtasks = await self.decompose(task)

        # Assign to managers
        all_results = {}
        for manager, subtask in zip(self.managers, subtasks):
            print(f"\n📋 Assigning to {manager.name}: {subtask[:50]}...")
            results = await manager.delegate(subtask)
            synthesized = await manager.synthesize(results)
            all_results[manager.name] = synthesized

        # Final synthesis
        print("\n🎯 Director: final synthesis...")
        final_str = "\n\n".join(f"{name}:\n{result}" for name, result in all_results.items())

        prompt = f"""Create final comprehensive answer from manager reports:

{final_str}

Provide a complete, well-organized response."""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content or ""


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Hierarchical Agents Example")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Build hierarchy
    # Team 1: Technical
    tech_workers = [
        Worker("Backend Dev", "backend systems", llm),
        Worker("Frontend Dev", "user interfaces", llm),
    ]
    tech_manager = Manager("Tech Lead", tech_workers, llm)

    # Team 2: Business
    biz_workers = [
        Worker("Product Manager", "product strategy", llm),
        Worker("Marketing", "market analysis", llm),
    ]
    biz_manager = Manager("Business Lead", biz_workers, llm)

    # Director
    director = Director([tech_manager, biz_manager], llm)

    # Execute
    task = "Plan a new AI-powered code review tool"

    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print(f"{'=' * 60}")

    result = await director.coordinate(task)

    print(f"\n{'=' * 60}")
    print("Final Result:")
    print(f"{'=' * 60}")
    print(result)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nHierarchy:")
    print("  Director")
    print("  ├── Tech Lead")
    print("  │   ├── Backend Dev")
    print("  │   └── Frontend Dev")
    print("  └── Business Lead")
    print("      ├── Product Manager")
    print("      └── Marketing")


if __name__ == "__main__":
    asyncio.run(main())
