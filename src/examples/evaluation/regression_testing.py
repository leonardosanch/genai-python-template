"""
LLM Output Regression Testing.

Demonstrates:
- Golden dataset management
- Semantic similarity comparison
- Drift detection
- CI/CD integration

Run: uv run python -m src.examples.evaluation.regression_testing
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class GoldenExample(BaseModel):
    """Golden dataset example for regression testing."""

    id: str = Field(description="Unique identifier")
    input: dict[str, Any] = Field(description="Input data")
    expected_output: str = Field(description="Expected LLM output")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class RegressionResult(BaseModel):
    """Result of regression test."""

    example_id: str
    input: dict[str, Any]
    expected_output: str
    actual_output: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    drift_detected: bool = False


async def run_regression_tests(
    client: AsyncOpenAI,
    golden_dataset: list[GoldenExample],
    similarity_threshold: float = 0.85,
) -> list[RegressionResult]:
    """
    Run regression tests against golden dataset.

    Args:
        client: OpenAI async client
        golden_dataset: List of golden examples
        similarity_threshold: Minimum similarity to pass

    Returns:
        List of RegressionResult objects
    """
    results = []

    for example in golden_dataset:
        # Generate output (simplified - use actual prompt in production)
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": str(example.input)}],
        )

        actual_output = response.choices[0].message.content or ""

        # Calculate semantic similarity
        similarity = await calculate_semantic_similarity(
            client,
            example.expected_output,
            actual_output,
        )

        passed = similarity >= similarity_threshold
        drift_detected = similarity < 0.7  # Significant drift threshold

        results.append(
            RegressionResult(
                example_id=example.id,
                input=example.input,
                expected_output=example.expected_output,
                actual_output=actual_output,
                similarity_score=similarity,
                passed=passed,
                drift_detected=drift_detected,
            )
        )

    return results


async def calculate_semantic_similarity(
    client: AsyncOpenAI,
    text1: str,
    text2: str,
) -> float:
    """Calculate semantic similarity using embeddings."""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[text1, text2],
    )

    emb1 = np.array(response.data[0].embedding)
    emb2 = np.array(response.data[1].embedding)

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return float(similarity)


async def detect_output_drift(
    client: AsyncOpenAI,
    golden_dataset: list[GoldenExample],
    drift_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Detect drift in LLM outputs.

    Args:
        client: OpenAI async client
        golden_dataset: Golden examples
        drift_threshold: Threshold for drift detection

    Returns:
        Drift analysis report
    """
    results = await run_regression_tests(client, golden_dataset, drift_threshold)

    drifted = [r for r in results if r.drift_detected]
    avg_similarity = sum(r.similarity_score for r in results) / len(results)

    return {
        "total_examples": len(results),
        "drifted_count": len(drifted),
        "drift_percentage": len(drifted) / len(results) * 100,
        "average_similarity": avg_similarity,
        "drifted_examples": [
            {
                "id": r.example_id,
                "similarity": r.similarity_score,
                "expected": r.expected_output[:100],
                "actual": r.actual_output[:100],
            }
            for r in drifted
        ],
    }


def save_golden_dataset(dataset: list[GoldenExample], path: Path) -> None:
    """Save golden dataset to file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [ex.model_dump(mode="json") for ex in dataset]
    path.write_text(json.dumps(data, indent=2, default=str))


def load_golden_dataset(path: Path) -> list[GoldenExample]:
    """Load golden dataset from file."""
    data = json.loads(path.read_text())
    return [GoldenExample(**ex) for ex in data]


async def main() -> None:
    """Example usage of regression testing."""
    client = AsyncOpenAI()

    print("🧪 LLM Regression Testing Example\n")

    # Create golden dataset
    golden_dataset = [
        GoldenExample(
            id="test_001",
            input={"query": "What is Python?"},
            expected_output="Python is a high-level, interpreted programming language "
            "known for its simplicity and readability.",
        ),
        GoldenExample(
            id="test_002",
            input={"query": "What is machine learning?"},
            expected_output="Machine learning is a subset of artificial intelligence "
            "that enables systems to learn and improve from experience without being "
            "explicitly programmed.",
        ),
    ]

    # Save golden dataset
    dataset_path = Path("tests/data/golden_dataset.json")
    save_golden_dataset(golden_dataset, dataset_path)
    print(f"💾 Golden dataset saved to: {dataset_path}\n")

    # Run regression tests
    print("Running regression tests...")
    results = await run_regression_tests(client, golden_dataset)

    passed = sum(1 for r in results if r.passed)
    print("\n📊 Results:")
    print(f"  Total: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {len(results) - passed}")
    print(f"  Pass Rate: {passed / len(results) * 100:.1f}%")

    # Detect drift
    print("\n🔍 Drift Detection:")
    drift_report = await detect_output_drift(client, golden_dataset)
    print(f"  Drifted Examples: {drift_report['drifted_count']}/{drift_report['total_examples']}")
    print(f"  Drift Percentage: {drift_report['drift_percentage']:.1f}%")
    print(f"  Average Similarity: {drift_report['average_similarity']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
