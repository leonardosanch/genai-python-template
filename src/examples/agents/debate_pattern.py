"""
Debate Pattern Example

Demonstrates:
- Multi-agent debate/discussion
- Consensus building
- Iterative refinement
- Perspective diversity
- Moderator coordination

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.agents.debate_pattern
"""

import asyncio
import os

from langchain_core.messages import SystemMessage  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore


class DebateAgent:
    """Agent participating in debate."""

    def __init__(self, name: str, perspective: str, llm: ChatOpenAI):
        """Initialize debate agent."""
        self.name = name
        self.perspective = perspective
        self.llm = llm

    async def argue(self, topic: str, previous_args: list[str]) -> str:
        """Make argument."""
        context = "\n\n".join(previous_args) if previous_args else "None yet"

        prompt = f"""You are {self.name} with perspective: {self.perspective}

Topic: {topic}

Previous arguments:
{context}

Provide your argument (2-3 sentences). Be constructive."""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content or ""


class Moderator:
    """Moderator facilitating debate."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize moderator."""
        self.llm = llm

    async def synthesize(self, topic: str, arguments: list[tuple[str, str]]) -> str:
        """Synthesize debate into consensus."""
        args_str = "\n\n".join(f"{name}: {arg}" for name, arg in arguments)

        prompt = f"""Synthesize this debate into a balanced conclusion:

Topic: {topic}

Arguments:
{args_str}

Provide a comprehensive, balanced conclusion."""

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content or ""


class DebateSystem:
    """Multi-agent debate system."""

    def __init__(self, agents: list[DebateAgent], moderator: Moderator):
        """Initialize debate system."""
        self.agents = agents
        self.moderator = moderator

    async def run(self, topic: str, rounds: int = 2) -> str:
        """
        Run debate.

        Args:
            topic: Debate topic
            rounds: Number of debate rounds

        Returns:
            Final consensus
        """
        print(f"\n{'=' * 60}")
        print(f"Debate: {topic}")
        print(f"{'=' * 60}")

        all_arguments: list[tuple[str, str]] = []
        previous_args: list[str] = []

        for round_num in range(1, rounds + 1):
            print(f"\n🔄 Round {round_num}/{rounds}")
            print("-" * 60)

            for agent in self.agents:
                print(f"\n💬 {agent.name}:")
                arg = await agent.argue(topic, previous_args)
                print(f"   {arg}")

                all_arguments.append((agent.name, arg))
                previous_args.append(f"{agent.name}: {arg}")

        # Synthesize
        print(f"\n{'=' * 60}")
        print("Moderator Synthesis")
        print(f"{'=' * 60}")

        consensus = await self.moderator.synthesize(topic, all_arguments)
        return consensus


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Debate Pattern Example")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create agents with different perspectives
    agents = [
        DebateAgent(
            "Pragmatist",
            "Focus on practical implementation and real-world constraints",
            llm,
        ),
        DebateAgent(
            "Idealist",
            "Focus on best practices and long-term quality",
            llm,
        ),
        DebateAgent(
            "Skeptic",
            "Question assumptions and identify risks",
            llm,
        ),
    ]

    moderator = Moderator(llm)
    debate = DebateSystem(agents, moderator)

    # Run debate
    topic = "Should we use microservices or monolith for a new project?"
    consensus = await debate.run(topic, rounds=2)

    print(f"\n{consensus}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Multi-agent debate")
    print("✅ Diverse perspectives")
    print("✅ Iterative refinement")
    print("✅ Consensus building")


if __name__ == "__main__":
    asyncio.run(main())
