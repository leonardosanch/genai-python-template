"""
Context Compression Techniques.

Demonstrates:
- LLMLingua-style compression
- Extractive summarization
- Selective context pruning
- Token-aware truncation

Run: uv run python -m src.examples.context_engineering.context_compression
"""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class CompressionResult(BaseModel):
    """Result of context compression."""

    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float = Field(ge=0.0, le=1.0)


async def compress_context(
    client: AsyncOpenAI,
    text: str,
    target_tokens: int,
) -> CompressionResult:
    """
    Compress context using LLM-based summarization.

    Args:
        client: OpenAI async client
        text: Text to compress
        target_tokens: Target token count

    Returns:
        CompressionResult with compressed text
    """
    original_tokens = len(text) // 4  # Rough estimate

    prompt = (
        f"Compress the following text to approximately {target_tokens} tokens while "
        f"preserving key information:\n\n{text}\n\nCompressed version:"
    )

    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target_tokens,
    )

    compressed = response.choices[0].message.content or ""
    compressed_tokens = len(compressed) // 4

    return CompressionResult(
        original_text=text,
        compressed_text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 0.0,
    )


async def extractive_summarization(text: str, num_sentences: int = 3) -> str:
    """Extract most important sentences (simplified)."""
    sentences = text.split(". ")

    # Simple heuristic: prefer longer sentences with keywords
    scored = [(s, len(s) + s.lower().count("important") * 50) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_sentences = [s[0] for s in scored[:num_sentences]]
    return ". ".join(top_sentences) + "."


async def main() -> None:
    """Example usage of context compression."""
    client = AsyncOpenAI()

    print("🗜️  Context Compression Example\n")

    long_text = (
        "Quantum computing is a revolutionary approach to computation that leverages the "
        "principles of quantum mechanics. Unlike classical computers that use bits (0 or 1), "
        "quantum computers use quantum bits or qubits. Qubits can exist in superposition, "
        "meaning they can be both 0 and 1 simultaneously. This property allows quantum "
        "computers to process vast amounts of information in parallel. Quantum entanglement "
        "is another key principle that enables qubits to be correlated in ways impossible "
        "for classical bits. Major tech companies like IBM, Google, and Microsoft are "
        "investing heavily in quantum computing research. Potential applications include "
        "cryptography, drug discovery, financial modeling, and optimization problems. "
        "However, quantum computers are still in early stages and face challenges like "
        "error correction and maintaining quantum coherence."
    )

    result = await compress_context(client, long_text, target_tokens=50)

    print(f"Original ({result.original_tokens} tokens):")
    print(result.original_text[:200] + "...")
    print(f"\nCompressed ({result.compressed_tokens} tokens):")
    print(result.compressed_text)
    print(f"\nCompression Ratio: {result.compression_ratio:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
