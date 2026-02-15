"""
RAGAS RAG Evaluation Example.

Demonstrates:
- Context precision
- Context recall
- Faithfulness
- Answer relevancy
- End-to-end RAG evaluation

Run: uv run python -m src.examples.evaluation.ragas_evaluation
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

# Note: RAGAS is an optional dependency
# Install with: uv pip install ragas
try:
    from datasets import Dataset  # type: ignore
    from ragas import evaluate as ragas_evaluate  # type: ignore
    from ragas.metrics import (  # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS not installed. Install with: uv pip install ragas")


class RAGEvaluationResult(BaseModel):
    """Result of RAG pipeline evaluation."""

    dataset_name: str = Field(description="Name of evaluation dataset")
    num_examples: int = Field(description="Number of examples evaluated")

    context_precision: float | None = Field(
        None, ge=0.0, le=1.0, description="Precision of retrieved context"
    )
    context_recall: float | None = Field(
        None, ge=0.0, le=1.0, description="Recall of retrieved context"
    )
    faithfulness: float | None = Field(None, ge=0.0, le=1.0, description="Faithfulness to context")
    answer_relevancy: float | None = Field(None, ge=0.0, le=1.0, description="Relevancy to query")

    overall_score: float | None = Field(None, ge=0.0, le=1.0, description="Average of all metrics")


async def evaluate_rag_pipeline(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> RAGEvaluationResult:
    """
    Evaluate RAG pipeline using RAGAS metrics.

    Args:
        questions: List of user queries
        answers: List of generated answers
        contexts: List of retrieved context lists (one list per query)
        ground_truths: Optional list of ground truth answers

    Returns:
        RAGEvaluationResult with all metrics

    Example:
        >>> result = await evaluate_rag_pipeline(
        ...     questions=["What is Python?"],
        ...     answers=["Python is a programming language."],
        ...     contexts=[["Python is a high-level language."]],
        ...     ground_truths=["Python is a programming language."],
        ... )
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS is required. Install with: uv pip install ragas")

    # Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    # Define metrics
    metrics = [
        context_precision,
        faithfulness,
        answer_relevancy,
    ]

    if ground_truths:
        metrics.append(context_recall)

    # Evaluate
    result = ragas_evaluate(
        dataset,
        metrics=metrics,
    )

    # Extract scores
    scores = result.to_pandas()

    avg_scores = {
        "context_precision": (
            scores["context_precision"].mean() if "context_precision" in scores else None
        ),
        "context_recall": (scores["context_recall"].mean() if "context_recall" in scores else None),
        "faithfulness": scores["faithfulness"].mean() if "faithfulness" in scores else None,
        "answer_relevancy": (
            scores["answer_relevancy"].mean() if "answer_relevancy" in scores else None
        ),
    }

    # Calculate overall score
    valid_scores = [v for v in avg_scores.values() if v is not None]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return RAGEvaluationResult(
        dataset_name="custom",
        num_examples=len(questions),
        context_precision=avg_scores["context_precision"],
        context_recall=avg_scores["context_recall"],
        faithfulness=avg_scores["faithfulness"],
        answer_relevancy=avg_scores["answer_relevancy"],
        overall_score=overall,
    )


async def compare_chunking_strategies(
    questions: list[str],
    documents: list[str],
    ground_truths: list[str],
    chunk_sizes: list[int] = [256, 512, 1024],
) -> dict[int, RAGEvaluationResult]:
    """
    Compare different chunking strategies for RAG.

    Args:
        questions: List of user queries
        documents: List of source documents
        ground_truths: List of ground truth answers
        chunk_sizes: List of chunk sizes to test

    Returns:
        Dict mapping chunk size to evaluation results

    Example:
        >>> results = await compare_chunking_strategies(
        ...     questions=["What is Python?"],
        ...     documents=["Python is a high-level programming language..."],
        ...     ground_truths=["Python is a programming language."],
        ... )
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS is required. Install with: uv pip install ragas")

    results = {}

    for chunk_size in chunk_sizes:
        # Simulate chunking (in production, use actual chunking logic)
        contexts = []
        for doc in documents:
            # Simple chunking by character count
            chunks = [doc[i : i + chunk_size] for i in range(0, len(doc), chunk_size)]
            contexts.append(chunks[:3])  # Top 3 chunks

        # Simulate answer generation (in production, use actual RAG pipeline)
        answers = [f"Answer based on {chunk_size} char chunks" for _ in questions]

        # Evaluate
        result = await evaluate_rag_pipeline(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )

        results[chunk_size] = result

    return results


async def export_results(
    result: RAGEvaluationResult,
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Export evaluation results to file.

    Args:
        result: Evaluation result to export
        output_path: Path to output file
        format: Export format ('json' or 'csv')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        output_path.write_text(result.model_dump_json(indent=2))
    elif format == "csv":
        import pandas as pd

        df = pd.DataFrame([result.model_dump()])
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


async def main() -> None:
    """Example usage of RAGAS evaluation."""
    if not RAGAS_AVAILABLE:
        print("❌ RAGAS not installed. Install with: uv pip install ragas")
        return

    print("📊 RAGAS RAG Evaluation Example\n")

    # Example dataset
    questions = [
        "What is the capital of France?",
        "What is machine learning?",
        "Who invented Python?",
    ]

    answers = [
        "The capital of France is Paris.",
        "Machine learning is a subset of AI that learns from data.",
        "Python was created by Guido van Rossum.",
    ]

    contexts = [
        ["Paris is the capital and largest city of France."],
        ["Machine learning is a branch of AI focused on learning from data."],
        ["Guido van Rossum created Python in 1991."],
    ]

    ground_truths = [
        "Paris",
        "Machine learning is a type of artificial intelligence.",
        "Guido van Rossum",
    ]

    # Evaluate
    print("Evaluating RAG pipeline...")
    result = await evaluate_rag_pipeline(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )

    print("\n📈 Results:")
    print(f"  Context Precision: {result.context_precision:.3f}")
    print(f"  Context Recall: {result.context_recall:.3f}")
    print(f"  Faithfulness: {result.faithfulness:.3f}")
    print(f"  Answer Relevancy: {result.answer_relevancy:.3f}")
    print(f"  Overall Score: {result.overall_score:.3f}")

    # Export results
    output_path = Path("evaluations/ragas_results.json")
    await export_results(result, output_path, format="json")
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
