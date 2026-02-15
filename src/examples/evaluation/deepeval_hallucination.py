"""
DeepEval Hallucination Detection Example.

Demonstrates:
- Faithfulness metric
- Hallucination metric
- Answer relevancy
- Contextual precision/recall

Run: uv run python -m src.examples.evaluation.deepeval_hallucination
"""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

# Note: DeepEval is an optional dependency
# Install with: uv pip install deepeval
try:
    from deepeval import evaluate  # type: ignore
    from deepeval.metrics import (  # type: ignore
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase  # type: ignore

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("⚠️  DeepEval not installed. Install with: uv pip install deepeval")


class EvaluationResult(BaseModel):
    """Result of hallucination evaluation."""

    test_case_id: str = Field(description="Unique identifier for test case")
    query: str = Field(description="User query")
    answer: str = Field(description="LLM generated answer")
    context: list[str] = Field(description="Retrieved context documents")

    faithfulness_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Faithfulness to context"
    )
    hallucination_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Hallucination detection score"
    )
    answer_relevancy_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Relevancy to query"
    )
    contextual_precision: float | None = Field(
        None, ge=0.0, le=1.0, description="Precision of retrieved context"
    )
    contextual_recall: float | None = Field(
        None, ge=0.0, le=1.0, description="Recall of retrieved context"
    )

    passed: bool = Field(description="Whether all thresholds were met")
    issues: list[str] = Field(default_factory=list, description="Detected issues")


async def evaluate_faithfulness(
    query: str,
    answer: str,
    context: list[str],
    threshold: float = 0.7,
) -> EvaluationResult:
    """
    Evaluate faithfulness of an answer to its context.

    Args:
        query: User query
        answer: LLM generated answer
        context: Retrieved context documents
        threshold: Minimum acceptable faithfulness score

    Returns:
        EvaluationResult with metrics

    Example:
        >>> result = await evaluate_faithfulness(
        ...     query="What is the capital of France?",
        ...     answer="The capital of France is Paris.",
        ...     context=["Paris is the capital and largest city of France."],
        ... )
        >>> assert result.passed
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval is required. Install with: uv pip install deepeval")

    # Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context,
    )

    # Define metrics
    faithfulness_metric = FaithfulnessMetric(threshold=threshold)
    hallucination_metric = HallucinationMetric(threshold=0.5)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

    # Evaluate
    metrics = [faithfulness_metric, hallucination_metric, relevancy_metric]

    # Run evaluation (DeepEval uses sync API)
    evaluation_results = evaluate(
        test_cases=[test_case],
        metrics=metrics,
        run_async=False,  # DeepEval handles async internally
    )

    # Extract scores
    test_result = evaluation_results.test_results[0]

    issues = []
    if faithfulness_metric.score < threshold:
        issues.append(f"Low faithfulness: {faithfulness_metric.score:.2f} < {threshold}")

    if hallucination_metric.score > 0.5:
        issues.append(f"Hallucination detected: {hallucination_metric.score:.2f}")

    if relevancy_metric.score < 0.7:
        issues.append(f"Low relevancy: {relevancy_metric.score:.2f}")

    return EvaluationResult(
        test_case_id=f"test_{hash(query) % 10000}",
        query=query,
        answer=answer,
        context=context,
        faithfulness_score=faithfulness_metric.score,
        hallucination_score=hallucination_metric.score,
        answer_relevancy_score=relevancy_metric.score,
        contextual_precision=None,  # Requires expected output
        contextual_recall=None,  # Requires expected output
        passed=test_result.success,
        issues=issues,
    )


async def evaluate_hallucination_batch(
    test_cases: list[dict[str, Any]],
    faithfulness_threshold: float = 0.7,
) -> list[EvaluationResult]:
    """
    Evaluate multiple test cases for hallucinations.

    Args:
        test_cases: List of dicts with 'query', 'answer', 'context' keys
        faithfulness_threshold: Minimum acceptable faithfulness

    Returns:
        List of EvaluationResult objects

    Example:
        >>> test_cases = [
        ...     {
        ...         "query": "What is Python?",
        ...         "answer": "Python is a programming language.",
        ...         "context": ["Python is a high-level programming language."],
        ...     },
        ... ]
        >>> results = await evaluate_hallucination_batch(test_cases)
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval is required. Install with: uv pip install deepeval")

    results = []

    for tc in test_cases:
        result = await evaluate_faithfulness(
            query=tc["query"],
            answer=tc["answer"],
            context=tc["context"],
            threshold=faithfulness_threshold,
        )
        results.append(result)

    return results


async def evaluate_with_contextual_metrics(
    query: str,
    answer: str,
    context: list[str],
    expected_output: str,
) -> EvaluationResult:
    """
    Evaluate with contextual precision and recall metrics.

    Requires expected output for comparison.

    Args:
        query: User query
        answer: LLM generated answer
        context: Retrieved context documents
        expected_output: Ground truth answer

    Returns:
        EvaluationResult with all metrics including contextual
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval is required. Install with: uv pip install deepeval")

    # Create test case with expected output
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=context,
    )

    # Define all metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        HallucinationMetric(threshold=0.5),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
    ]

    # Evaluate
    evaluation_results = evaluate(
        test_cases=[test_case],
        metrics=metrics,
        run_async=False,
    )

    test_result = evaluation_results.test_results[0]

    # Extract all scores
    scores = {metric.__class__.__name__: metric.score for metric in metrics}

    return EvaluationResult(
        test_case_id=f"test_{hash(query) % 10000}",
        query=query,
        answer=answer,
        context=context,
        faithfulness_score=scores.get("FaithfulnessMetric"),
        hallucination_score=scores.get("HallucinationMetric"),
        answer_relevancy_score=scores.get("AnswerRelevancyMetric"),
        contextual_precision=scores.get("ContextualPrecisionMetric"),
        contextual_recall=scores.get("ContextualRecallMetric"),
        passed=test_result.success,
        issues=[],
    )


# Example usage and pytest integration
async def main() -> None:
    """Example usage of DeepEval hallucination detection."""
    if not DEEPEVAL_AVAILABLE:
        print("❌ DeepEval not installed. Install with: uv pip install deepeval")
        return

    print("🔍 DeepEval Hallucination Detection Example\n")

    # Example 1: Faithful answer
    print("Example 1: Faithful Answer")
    result1 = await evaluate_faithfulness(
        query="What is the capital of France?",
        answer="The capital of France is Paris, which is also its largest city.",
        context=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
        ],
    )

    print(f"  Faithfulness: {result1.faithfulness_score:.2f}")
    print(f"  Hallucination: {result1.hallucination_score:.2f}")
    print(f"  Relevancy: {result1.answer_relevancy_score:.2f}")
    print(f"  Passed: {'✅' if result1.passed else '❌'}")

    if result1.issues:
        print(f"  Issues: {', '.join(result1.issues)}")

    print()

    # Example 2: Hallucinated answer
    print("Example 2: Hallucinated Answer")
    result2 = await evaluate_faithfulness(
        query="What is the population of Paris?",
        answer="Paris has a population of 50 million people and is the largest city in Europe.",
        context=[
            "Paris has a population of approximately 2.2 million people within city limits.",
            "The Paris metropolitan area has about 12 million inhabitants.",
        ],
    )

    print(f"  Faithfulness: {result2.faithfulness_score:.2f}")
    print(f"  Hallucination: {result2.hallucination_score:.2f}")
    print(f"  Relevancy: {result2.answer_relevancy_score:.2f}")
    print(f"  Passed: {'✅' if result2.passed else '❌'}")

    if result2.issues:
        print("  Issues:")
        for issue in result2.issues:
            print(f"    - {issue}")

    print()

    # Example 3: Batch evaluation
    print("Example 3: Batch Evaluation")
    test_cases = [
        {
            "query": "What is Python?",
            "answer": "Python is a high-level programming language known for its simplicity.",
            "context": ["Python is a high-level, interpreted programming language."],
        },
        {
            "query": "Who created Python?",
            "answer": "Python was created by Guido van Rossum in 1991.",
            "context": ["Guido van Rossum created Python, first released in 1991."],
        },
    ]

    batch_results = await evaluate_hallucination_batch(test_cases)

    passed = sum(1 for r in batch_results if r.passed)
    print(f"  Passed: {passed}/{len(batch_results)}")
    avg_faith = sum(r.faithfulness_score or 0 for r in batch_results) / len(batch_results)
    print(f"  Average Faithfulness: {avg_faith:.2f}")


# Pytest integration example
def test_faithfulness_high_quality_answer() -> None:
    """Test that high-quality answers pass faithfulness check."""
    if not DEEPEVAL_AVAILABLE:
        return  # Skip if DeepEval not installed

    result = asyncio.run(
        evaluate_faithfulness(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI that enables systems to learn from data.",
            context=[
                "Machine learning is a branch of artificial intelligence focused "
                "on learning from data."
            ],
            threshold=0.7,
        )
    )

    assert result.passed, f"Expected to pass but got issues: {result.issues}"
    assert result.faithfulness_score is not None and result.faithfulness_score >= 0.7


def test_hallucination_detection() -> None:
    """Test that hallucinated answers are detected."""
    if not DEEPEVAL_AVAILABLE:
        return  # Skip if DeepEval not installed

    result = asyncio.run(
        evaluate_faithfulness(
            query="What is the speed of light?",
            answer="The speed of light is 500,000 km/s and it can be exceeded by neutrinos.",
            context=["The speed of light in vacuum is approximately 299,792 km/s."],
            threshold=0.7,
        )
    )

    assert not result.passed, "Expected hallucination to be detected"
    assert len(result.issues) > 0


if __name__ == "__main__":
    asyncio.run(main())
