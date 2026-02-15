"""
Semantic Blueprint Implementation.

Demonstrates:
- Context structure definition
- Priority-based context selection
- Token budget management
- Dynamic context assembly

Run: uv run python -m src.examples.context_engineering.semantic_blueprint
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field


class ContextSlot(BaseModel):
    """A single context slot with priority and budget."""

    name: str = Field(description="Slot identifier")
    content: str = Field(description="Actual content")
    priority: int = Field(ge=1, le=5, description="Priority (5=highest)")
    max_tokens: int = Field(gt=0, description="Maximum tokens for this slot")
    category: Literal["system", "user", "retrieved", "memory"] = Field(
        description="Context category"
    )

    def estimate_tokens(self) -> int:
        """Estimate token count (simplified: ~4 chars per token)."""
        return len(self.content) // 4


class SemanticBlueprint(BaseModel):
    """Blueprint for assembling context with budget constraints."""

    slots: list[ContextSlot] = Field(description="All context slots")
    total_budget: int = Field(default=8000, description="Total token budget")

    def assemble(self) -> str:
        """
        Assemble context respecting budget and priorities.

        Returns:
            Assembled context string
        """
        # Sort by priority (descending)
        sorted_slots = sorted(self.slots, key=lambda s: s.priority, reverse=True)

        assembled = []
        used_tokens = 0

        for slot in sorted_slots:
            slot_tokens = min(slot.estimate_tokens(), slot.max_tokens)

            if used_tokens + slot_tokens <= self.total_budget:
                # Truncate content if needed
                if slot.estimate_tokens() > slot.max_tokens:
                    char_limit = slot.max_tokens * 4
                    content = slot.content[:char_limit] + "..."
                else:
                    content = slot.content

                assembled.append(f"[{slot.category.upper()}: {slot.name}]\n{content}\n")
                used_tokens += slot_tokens
            else:
                # Budget exceeded, skip lower priority slots
                break

        return "\n".join(assembled)

    def get_usage_stats(self) -> dict[str, int]:
        """Get token usage statistics."""
        assembled = self.assemble()
        used = len(assembled) // 4

        return {
            "total_budget": self.total_budget,
            "used_tokens": used,
            "remaining_tokens": self.total_budget - used,
            "utilization_pct": int(used / self.total_budget * 100),
        }


async def main() -> None:
    """Example usage of semantic blueprint."""
    print("🎯 Semantic Blueprint Example\n")

    # Define context slots
    slots = [
        ContextSlot(
            name="system_instructions",
            content="You are a helpful AI assistant. Be concise and accurate.",
            priority=5,  # Highest priority
            max_tokens=100,
            category="system",
        ),
        ContextSlot(
            name="user_query",
            content="Explain quantum computing in simple terms.",
            priority=5,
            max_tokens=200,
            category="user",
        ),
        ContextSlot(
            name="retrieved_doc_1",
            content=(
                "Quantum computing uses quantum bits (qubits) that can exist in superposition..."
            ),
            priority=4,
            max_tokens=500,
            category="retrieved",
        ),
        ContextSlot(
            name="retrieved_doc_2",
            content="Classical computers use bits (0 or 1), while quantum computers use qubits...",
            priority=3,
            max_tokens=500,
            category="retrieved",
        ),
        ContextSlot(
            name="conversation_history",
            content="Previous conversation about computer science topics...",
            priority=2,
            max_tokens=300,
            category="memory",
        ),
    ]

    # Create blueprint
    blueprint = SemanticBlueprint(slots=slots, total_budget=2000)

    # Assemble context
    context = blueprint.assemble()

    print("Assembled Context:")
    print("=" * 60)
    print(context)
    print("=" * 60)

    # Show usage stats
    stats = blueprint.get_usage_stats()
    print("\n📊 Token Usage:")
    print(f"  Budget: {stats['total_budget']}")
    print(f"  Used: {stats['used_tokens']}")
    print(f"  Remaining: {stats['remaining_tokens']}")
    print(f"  Utilization: {stats['utilization_pct']}%")


if __name__ == "__main__":
    asyncio.run(main())
