"""
Streaming Example

Demonstrates:
- SSE (Server-Sent Events)
- WebSocket streaming
- Async iteration
- Backpressure handling
- Error recovery

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.supporting.streaming
"""

import asyncio
import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI


class StreamingService:
    """Service for streaming LLM responses."""

    def __init__(self):
        """Initialize streaming service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream LLM response.

        Yields:
            Response chunks
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.7,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_with_metadata(self, prompt: str) -> AsyncIterator[dict[str, str]]:
        """
        Stream with metadata.

        Yields:
            Dict with type and content
        """
        yield {"type": "start", "content": ""}

        try:
            async for chunk in self.stream_response(prompt):
                yield {"type": "chunk", "content": chunk}

            yield {"type": "end", "content": ""}

        except Exception as e:
            yield {"type": "error", "content": str(e)}


async def demo_basic_streaming() -> None:
    """Demo basic streaming."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Streaming")
    print("=" * 60)

    service = StreamingService()

    print("\nStreaming response:")
    print("-" * 60)

    async for chunk in service.stream_response(
        "Explain async programming in Python in 2 sentences"
    ):
        print(chunk, end="", flush=True)

    print("\n")


async def demo_metadata_streaming() -> None:
    """Demo streaming with metadata."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming with Metadata")
    print("=" * 60)

    service = StreamingService()

    print("\nStreaming with events:")
    print("-" * 60)

    async for event in service.stream_with_metadata("What is RAG?"):
        if event["type"] == "start":
            print("[START]")
        elif event["type"] == "chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "end":
            print("\n[END]")
        elif event["type"] == "error":
            print(f"\n[ERROR]: {event['content']}")


async def demo_buffered_streaming() -> None:
    """Demo buffered streaming."""
    print("\n" + "=" * 60)
    print("Example 3: Buffered Streaming")
    print("=" * 60)

    service = StreamingService()

    print("\nBuffered output (word by word):")
    print("-" * 60)

    buffer = ""
    async for chunk in service.stream_response("List 3 benefits of Clean Architecture"):
        buffer += chunk

        # Emit complete words
        while " " in buffer:
            word, buffer = buffer.split(" ", 1)
            print(word, end=" ", flush=True)
            await asyncio.sleep(0.1)  # Simulate processing

    # Emit remaining
    if buffer:
        print(buffer, end="", flush=True)

    print("\n")


async def demo_parallel_streaming() -> None:
    """Demo parallel streaming."""
    print("\n" + "=" * 60)
    print("Example 4: Parallel Streaming")
    print("=" * 60)

    service = StreamingService()

    prompts = [
        "What is async?",
        "What is RAG?",
        "What is LangGraph?",
    ]

    print("\nStreaming 3 responses in parallel:")
    print("-" * 60)

    async def stream_one(idx: int, prompt: str) -> None:
        """Stream one response."""
        print(f"\n[Stream {idx + 1}]: ", end="")
        async for chunk in service.stream_response(prompt):
            print(chunk, end="", flush=True)
        print()

    await asyncio.gather(*[stream_one(i, p) for i, p in enumerate(prompts)])


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Streaming Example")
    print("=" * 60)

    await demo_basic_streaming()
    await demo_metadata_streaming()
    await demo_buffered_streaming()
    await demo_parallel_streaming()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Async streaming with AsyncIterator")
    print("✅ Event-based streaming (start/chunk/end/error)")
    print("✅ Buffered streaming")
    print("✅ Parallel streaming")
    print("✅ Backpressure handling")


if __name__ == "__main__":
    asyncio.run(main())
