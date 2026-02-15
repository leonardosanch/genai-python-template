"""
Observability Example

Demonstrates:
- Structured logging
- Metrics tracking
- Tracing (simplified)
- Performance monitoring
- Cost tracking

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.supporting.observability
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel


class LogEvent(BaseModel):
    """Structured log event."""

    timestamp: datetime
    level: str
    component: str
    event: str
    metadata: dict[str, Any] = {}


class Metrics(BaseModel):
    """System metrics."""

    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0


class ObservableRAG:
    """RAG system with observability."""

    def __init__(self) -> None:
        """Initialize observable RAG."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # Metrics
        self.metrics = Metrics()
        self.latencies: list[float] = []

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger."""
        logger = logging.getLogger("ObservableRAG")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log_event(self, level: str, component: str, event: str, **metadata: Any) -> None:
        """Log structured event."""
        log_event = LogEvent(
            timestamp=datetime.now(),
            level=level,
            component=component,
            event=event,
            metadata=metadata,
        )

        # Log as JSON
        self.logger.info(json.dumps(log_event.model_dump(), default=str))

    async def query(self, question: str) -> str:
        """Query with full observability."""
        request_id = f"req_{int(time.time() * 1000)}"

        # Log request start
        self.log_event(
            "INFO",
            "RAG",
            "query_start",
            request_id=request_id,
            question=question[:50],
        )

        start_time = time.time()

        try:
            # Generate
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
            )

            answer = response.choices[0].message.content or ""
            usage = response.usage

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens = usage.total_tokens if usage else 0
            cost = (
                (
                    usage.prompt_tokens * 0.15 / 1_000_000
                    + usage.completion_tokens * 0.60 / 1_000_000
                )
                if usage
                else 0.0
            )

            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_tokens += tokens
            self.metrics.total_cost_usd += cost
            self.latencies.append(latency_ms)
            self.metrics.avg_latency_ms = sum(self.latencies) / len(self.latencies)

            # Log success
            self.log_event(
                "INFO",
                "RAG",
                "query_success",
                request_id=request_id,
                latency_ms=latency_ms,
                tokens=tokens,
                cost_usd=cost,
            )

            return answer

        except Exception as e:
            # Log error
            self.metrics.error_count += 1
            self.log_event(
                "ERROR",
                "RAG",
                "query_error",
                request_id=request_id,
                error=str(e),
            )
            raise

    def get_metrics(self) -> Metrics:
        """Get current metrics."""
        return self.metrics

    def print_metrics(self) -> None:
        """Print metrics summary."""
        print(f"\n{'=' * 60}")
        print("Metrics Summary")
        print(f"{'=' * 60}")
        print(f"Total Requests:    {self.metrics.total_requests}")
        print(f"Total Tokens:      {self.metrics.total_tokens:,}")
        print(f"Total Cost:        ${self.metrics.total_cost_usd:.6f}")
        print(f"Avg Latency:       {self.metrics.avg_latency_ms:.2f}ms")
        print(f"Error Count:       {self.metrics.error_count}")


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Observability Example")
    print("=" * 60)

    rag = ObservableRAG()

    # Run queries
    questions = [
        "What is observability?",
        "Why is logging important?",
        "What are metrics?",
    ]

    print("\nRunning queries with full observability...\n")

    for question in questions:
        answer = await rag.query(question)
        print(f"\nQ: {question}")
        print(f"A: {answer[:100]}...")

    # Show metrics
    rag.print_metrics()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Structured logging (JSON)")
    print("✅ Metrics tracking (requests, tokens, cost)")
    print("✅ Latency monitoring")
    print("✅ Error tracking")
    print("✅ Request tracing")


if __name__ == "__main__":
    asyncio.run(main())
