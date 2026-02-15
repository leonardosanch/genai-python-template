"""
Conversation Memory Management.

Demonstrates:
- Sliding window memory
- Summary memory
- Entity memory
- Hybrid memory strategies

Run: uv run python -m src.examples.context_engineering.memory_management
"""

import asyncio
from collections import deque
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single conversation message."""

    role: str
    content: str
    timestamp: float = Field(default_factory=lambda: asyncio.get_event_loop().time())


class SlidingWindowMemory:
    """Keep only the last N messages."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: deque[Message] = deque(maxlen=window_size)

    def add(self, message: Message) -> None:
        """Add message to memory."""
        self.messages.append(message)

    def get_context(self) -> list[dict[str, str]]:
        """Get messages for LLM context."""
        return [{"role": m.role, "content": m.content} for m in self.messages]


class SummaryMemory:
    """Summarize old messages to save tokens."""

    def __init__(self, client: AsyncOpenAI, summary_threshold: int = 10):
        self.client = client
        self.summary_threshold = summary_threshold
        self.summary: str = ""
        self.recent_messages: list[Message] = []

    async def add(self, message: Message) -> None:
        """Add message and summarize if needed."""
        self.recent_messages.append(message)

        if len(self.recent_messages) >= self.summary_threshold:
            await self._summarize()

    async def _summarize(self) -> None:
        """Summarize old messages."""
        messages_text = "\n".join(f"{m.role}: {m.content}" for m in self.recent_messages[:-3])

        prompt = f"Summarize this conversation:\n{messages_text}"

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        new_summary = response.choices[0].message.content or ""
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.recent_messages = self.recent_messages[-3:]  # Keep last 3

    def get_context(self) -> list[dict[str, str]]:
        """Get context with summary + recent messages."""
        context = []

        if self.summary:
            context.append(
                {"role": "system", "content": f"Previous conversation summary: {self.summary}"}
            )

        context.extend({"role": m.role, "content": m.content} for m in self.recent_messages)

        return context


class EntityMemory:
    """Track entities mentioned in conversation."""

    def __init__(self) -> None:
        self.entities: dict[str, Any] = {}

    def extract_entities(self, text: str) -> None:
        """Extract entities (simplified)."""
        # In production, use NER model
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 3:
                self.entities[word] = self.entities.get(word, 0) + 1

    def get_context(self) -> str:
        """Get entity context."""
        if not self.entities:
            return ""

        top_entities = sorted(self.entities.items(), key=lambda x: x[1], reverse=True)[:5]
        return "Key entities: " + ", ".join(e[0] for e in top_entities)


async def main() -> None:
    """Example usage of memory management."""
    client = AsyncOpenAI()

    print("🧠 Memory Management Example\n")

    # Sliding window
    print("1. Sliding Window Memory")
    sliding = SlidingWindowMemory(window_size=3)

    for i in range(5):
        sliding.add(Message(role="user", content=f"Message {i}"))

    print(f"  Stored messages: {len(sliding.messages)}/3")
    print(f"  Messages: {[m.content for m in sliding.messages]}\n")

    # Summary memory
    print("2. Summary Memory")
    summary_mem = SummaryMemory(client, summary_threshold=3)

    await summary_mem.add(Message(role="user", content="Tell me about Python"))
    await summary_mem.add(Message(role="assistant", content="Python is a programming language"))
    await summary_mem.add(Message(role="user", content="What about its history?"))
    await summary_mem.add(Message(role="assistant", content="Created by Guido van Rossum in 1991"))

    print(f"  Summary: {summary_mem.summary[:100] if summary_mem.summary else 'None yet'}")
    print(f"  Recent messages: {len(summary_mem.recent_messages)}\n")

    # Entity memory
    print("3. Entity Memory")
    entity_mem = EntityMemory()
    entity_mem.extract_entities("Python was created by Guido van Rossum")
    entity_mem.extract_entities("Guido van Rossum also created Python Enhancement Proposals")

    print(f"  {entity_mem.get_context()}")


if __name__ == "__main__":
    asyncio.run(main())
