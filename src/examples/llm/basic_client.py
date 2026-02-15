"""
Basic LLM Client Example

Demonstrates:
- Multi-provider LLM usage (OpenAI, Anthropic)
- Async patterns
- Error handling and retries
- Cost tracking
- Response streaming

Usage:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."

    python -m src.examples.llm.basic_client
"""

import asyncio
import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Structured LLM response."""

    content: str
    model: str
    tokens_used: int
    cost_usd: float


class BasicLLMClient:
    """
    Basic LLM client supporting multiple providers.

    Demonstrates best practices:
    - Provider abstraction
    - Error handling
    - Token/cost tracking
    - Async operations
    """

    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider ("openai" or "anthropic")
        """
        self.provider = provider

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
            # Pricing per 1M tokens (as of 2024)
            self.input_cost_per_1m = 0.15
            self.output_cost_per_1m = 0.60

        elif provider == "anthropic":
            # Note: Would use anthropic library here
            raise NotImplementedError("Anthropic example coming soon")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate completion from prompt.

        Args:
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content and metadata

        Raises:
            OpenAIError: On API errors
        """
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract response
            content = response.choices[0].message.content or ""
            usage = response.usage

            if not usage:
                raise ValueError("No usage data in response")

            # Calculate cost
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            cost_usd = (
                input_tokens * self.input_cost_per_1m / 1_000_000
                + output_tokens * self.output_cost_per_1m / 1_000_000
            )

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=total_tokens,
                cost_usd=cost_usd,
            )

        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            raise

    async def stream_generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens.

        Args:
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature

        Yields:
            Content chunks as they arrive
        """
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            print(f"OpenAI streaming error: {e}")
            raise


async def main() -> None:
    """Run example demonstrations."""
    print("=" * 60)
    print("Basic LLM Client Example")
    print("=" * 60)

    # Initialize client
    client = BasicLLMClient(provider="openai")

    # Example 1: Simple generation
    print("\n1. Simple Generation")
    print("-" * 60)

    response = await client.generate(
        prompt="Explain Clean Architecture in 2 sentences.",
        system="You are a senior software architect.",
        temperature=0.3,
    )

    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Cost: ${response.cost_usd:.6f}")

    # Example 2: Streaming
    print("\n2. Streaming Generation")
    print("-" * 60)
    print("Response: ", end="", flush=True)

    async for chunk in client.stream_generate(
        prompt="List 3 benefits of async Python in one sentence each.",
        temperature=0.5,
    ):
        print(chunk, end="", flush=True)

    print("\n")

    # Example 3: Multiple calls with cost tracking
    print("\n3. Batch Processing with Cost Tracking")
    print("-" * 60)

    prompts = [
        "What is RAG?",
        "What is a vector database?",
        "What is semantic search?",
    ]

    total_cost = 0.0
    total_tokens = 0

    for i, prompt in enumerate(prompts, 1):
        response = await client.generate(
            prompt=prompt,
            system="Answer in one concise sentence.",
            temperature=0.2,
            max_tokens=100,
        )
        total_cost += response.cost_usd
        total_tokens += response.tokens_used

        print(f"\nQ{i}: {prompt}")
        print(f"A{i}: {response.content}")

    print(f"\n{'─' * 60}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
