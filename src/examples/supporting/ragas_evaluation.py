"""
RAGAS Evaluation Example

Demonstrates:
- RAG evaluation metrics
- Faithfulness scoring
- Answer relevancy
- Context precision
- Automated testing

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.supporting.ragas_evaluation
"""

import asyncio
import os

from openai import AsyncOpenAI
from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    """RAG evaluation metrics."""

    faithfulness: float  # Answer grounded in context
    answer_relevancy: float  # Answer relevant to question
    context_precision: float  # Retrieved context is relevant
    overall_score: float


class RAGEvaluator:
    """Evaluator for RAG systems."""

    def __init__(self) -> None:
        """Initialize evaluator."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    async def evaluate_faithfulness(self, answer: str, context: list[str]) -> float:
        """Evaluate if answer is grounded in context."""
        context_str = "\n".join(context)

        prompt = f"""Rate faithfulness (0-1): Is the answer grounded in context?

Context:
{context_str}

Answer:
{answer}

Score (0.0-1.0):"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )

        try:
            score = float(response.choices[0].message.content or "0.5")
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    async def evaluate_relevancy(self, question: str, answer: str) -> float:
        """Evaluate answer relevancy to question."""
        prompt = f"""Rate relevancy (0-1): Does answer address the question?

Question: {question}
Answer: {answer}

Score (0.0-1.0):"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )

        try:
            score = float(response.choices[0].message.content or "0.5")
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    async def evaluate_context_precision(self, question: str, context: list[str]) -> float:
        """Evaluate if retrieved context is relevant."""
        context_str = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(context))

        prompt = f"""Rate context precision (0-1): Is context relevant to question?

Question: {question}

Context:
{context_str}

Score (0.0-1.0):"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )

        try:
            score = float(response.choices[0].message.content or "0.5")
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    async def evaluate(self, question: str, answer: str, context: list[str]) -> EvaluationMetrics:
        """
        Evaluate RAG response.

        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context

        Returns:
            Evaluation metrics
        """
        print(f"\n{'=' * 60}")
        print("Evaluating RAG Response")
        print(f"{'=' * 60}")

        # Run evaluations in parallel
        faithfulness, relevancy, precision = await asyncio.gather(
            self.evaluate_faithfulness(answer, context),
            self.evaluate_relevancy(question, answer),
            self.evaluate_context_precision(question, context),
        )

        overall = (faithfulness + relevancy + precision) / 3

        metrics = EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            overall_score=overall,
        )

        print("\nMetrics:")
        print(f"  Faithfulness:      {metrics.faithfulness:.2f}")
        print(f"  Answer Relevancy:  {metrics.answer_relevancy:.2f}")
        print(f"  Context Precision: {metrics.context_precision:.2f}")
        print(f"  Overall Score:     {metrics.overall_score:.2f}")

        return metrics


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("RAGAS Evaluation Example")
    print("=" * 60)

    evaluator = RAGEvaluator()

    # Example 1: Good response
    print("\nExample 1: High-Quality Response")
    print("-" * 60)

    await evaluator.evaluate(
        question="What is RAG?",
        answer="RAG combines retrieval with generation for grounded AI responses.",
        context=[
            "RAG (Retrieval-Augmented Generation) combines retrieval with generation.",
            "RAG provides grounded, factual responses.",
        ],
    )

    # Example 2: Poor response
    print("\n\nExample 2: Low-Quality Response")
    print("-" * 60)

    await evaluator.evaluate(
        question="What is RAG?",
        answer="Python is a programming language.",
        context=[
            "RAG combines retrieval with generation.",
            "RAG provides grounded responses.",
        ],
    )

    # Example 3: Irrelevant context
    print("\n\nExample 3: Irrelevant Context")
    print("-" * 60)

    await evaluator.evaluate(
        question="What is RAG?",
        answer="RAG combines retrieval with generation.",
        context=[
            "Python is a programming language.",
            "FastAPI is a web framework.",
        ],
    )

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Metrics:")
    print("✅ Faithfulness - Answer grounded in context")
    print("✅ Answer Relevancy - Answer addresses question")
    print("✅ Context Precision - Retrieved context is relevant")
    print("✅ Overall Score - Combined metric")


if __name__ == "__main__":
    asyncio.run(main())
