"""
Celery Tasks Example

Demonstrates:
- Background task processing
- Async task queues
- Task scheduling
- Result tracking
- Error handling

Usage:
    export OPENAI_API_KEY="sk-..."

    # Start Redis (required for Celery)
    # docker run -d -p 6379:6379 redis

    # Start worker
    # celery -A src.examples.integration.celery_tasks worker --loglevel=info

    # Run example
    python -m src.examples.integration.celery_tasks
"""

import os
import time
from collections.abc import Callable
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

# Note: Celery setup (simplified for demonstration)
# In production, use proper Celery configuration


class TaskResult(BaseModel):
    """Task result."""

    task_id: str
    status: str
    result: Any | None = None
    error: str | None = None


class MockCelery:
    """Mock Celery for demonstration (use real Celery in production)."""

    def __init__(self) -> None:
        """Initialize mock."""
        self.tasks: dict[str, TaskResult] = {}
        self.task_counter = 0

    def task(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Task decorator."""

        def wrapper(*args: Any, **kwargs: Any) -> TaskResult:
            # Generate task ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"

            # Store pending task
            self.tasks[task_id] = TaskResult(
                task_id=task_id,
                status="PENDING",
            )

            # Execute (in real Celery, this would be async)
            try:
                result = func(*args, **kwargs)
                self.tasks[task_id] = TaskResult(
                    task_id=task_id,
                    status="SUCCESS",
                    result=result,
                )
            except Exception as e:
                self.tasks[task_id] = TaskResult(
                    task_id=task_id,
                    status="FAILURE",
                    error=str(e),
                )

            return self.tasks[task_id]

        return wrapper

    def get_result(self, task_id: str) -> TaskResult:
        """Get task result."""
        return self.tasks.get(
            task_id,
            TaskResult(task_id=task_id, status="NOT_FOUND"),
        )


# Initialize mock Celery
celery_app = MockCelery()


@celery_app.task
def process_document(content: str) -> dict[str, Any]:
    """
    Process document in background.

    Args:
        content: Document content

    Returns:
        Processing result
    """
    print(f"Processing document ({len(content)} chars)...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    client = OpenAI(api_key=api_key)

    # Simulate processing
    time.sleep(1)

    # Generate summary
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Summarize in 1 sentence:\n\n{content}",
            }
        ],
        temperature=0.3,
    )

    summary = response.choices[0].message.content or ""

    return {
        "summary": summary,
        "word_count": len(content.split()),
        "processed_at": time.time(),
    }


@celery_app.task
def batch_process(documents: list[str]) -> list[dict[str, Any]]:
    """
    Process multiple documents.

    Args:
        documents: List of documents

    Returns:
        List of results
    """
    print(f"Batch processing {len(documents)} documents...")

    results = []
    for i, doc in enumerate(documents):
        print(f"  Processing {i + 1}/{len(documents)}...")
        result = process_document(doc)
        results.append(result.result if hasattr(result, "result") else result)

    return results


def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Celery Tasks Example")
    print("=" * 60)
    print("\nNote: Using mock Celery for demonstration")
    print("In production, use real Celery with Redis/RabbitMQ")

    # Example 1: Single task
    print("\n\nExample 1: Single Document Processing")
    print("-" * 60)

    doc = "RAG combines retrieval with generation for grounded AI responses."

    task = process_document(doc)
    print(f"Task ID: {task.task_id}")
    print(f"Status: {task.status}")
    if task.result:
        print(f"Summary: {task.result['summary']}")
        print(f"Word count: {task.result['word_count']}")

    # Example 2: Batch processing
    print("\n\nExample 2: Batch Processing")
    print("-" * 60)

    docs = [
        "Clean Architecture separates concerns into layers.",
        "Async programming enables concurrent execution.",
        "Vector databases store and search embeddings.",
    ]

    batch_task = batch_process(docs)
    print(f"Task ID: {batch_task.task_id}")
    print(f"Status: {batch_task.status}")
    if batch_task.result:
        print(f"Processed {len(batch_task.result)} documents")
        for i, result in enumerate(batch_task.result, 1):
            print(f"\n  Document {i}:")
            print(f"    Summary: {result['summary']}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Background task processing")
    print("✅ Task queues")
    print("✅ Batch processing")
    print("✅ Result tracking")
    print("✅ Error handling")
    print("\nProduction Setup:")
    print("  1. Install: pip install celery redis")
    print("  2. Start Redis: docker run -d -p 6379:6379 redis")
    print("  3. Start worker: celery -A app worker")
    print("  4. Submit tasks via API")


if __name__ == "__main__":
    main()
