"""
LLM-as-Judge Evaluation Pattern.

Demonstrates:
- Custom evaluation prompts
- Pairwise comparison
- Rubric-based scoring
- Multi-aspect evaluation

Run: uv run python -m src.examples.evaluation.llm_as_judge
"""

import asyncio
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class PairwiseResult(BaseModel):
    """Result of pairwise comparison."""

    winner: Literal["A", "B", "tie"] = Field(description="Which response won")
    reasoning: str = Field(description="Judge's reasoning")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level")


class RubricScore(BaseModel):
    """Score for a single rubric criterion."""

    criterion: str = Field(description="Criterion name")
    score: int = Field(ge=1, le=5, description="Score 1-5")
    justification: str = Field(description="Why this score was given")


class MultiAspectResult(BaseModel):
    """Result of multi-aspect evaluation."""

    scores: list[RubricScore] = Field(description="Scores for each criterion")
    overall_score: float = Field(ge=1.0, le=5.0, description="Average score")
    strengths: list[str] = Field(description="Identified strengths")
    weaknesses: list[str] = Field(description="Identified weaknesses")


async def pairwise_comparison(
    client: AsyncOpenAI,
    query: str,
    response_a: str,
    response_b: str,
    criteria: str = "accuracy, helpfulness, and clarity",
) -> PairwiseResult:
    """
    Compare two responses using LLM as judge.

    Args:
        client: OpenAI async client
        query: Original user query
        response_a: First response
        response_b: Second response
        criteria: Evaluation criteria

    Returns:
        PairwiseResult with winner and reasoning
    """
    prompt = f"""You are an expert evaluator. Compare these two responses to the user's query.

Query: {query}

Response A:
{response_a}

Response B:
{response_b}

Evaluate based on: {criteria}

Provide your evaluation in JSON format:
{{
    "winner": "A" | "B" | "tie",
    "reasoning": "detailed explanation",
    "confidence": "high" | "medium" | "low"
}}
"""

    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    import json

    result_data = json.loads(response.choices[0].message.content or "{}")

    return PairwiseResult(**result_data)


async def rubric_based_evaluation(
    client: AsyncOpenAI,
    query: str,
    response: str,
    rubric: dict[str, str],
) -> MultiAspectResult:
    """
    Evaluate response using a rubric.

    Args:
        client: OpenAI async client
        query: Original query
        response: Response to evaluate
        rubric: Dict mapping criterion name to description

    Returns:
        MultiAspectResult with scores for each criterion
    """
    rubric_text = "\n".join(f"- {name}: {description}" for name, description in rubric.items())

    prompt = f"""You are an expert evaluator. Score this response on a 1-5 scale for each criterion.

Query: {query}

Response:
{response}

Rubric (1=Poor, 5=Excellent):
{rubric_text}

Provide scores in JSON format:
{{
    "scores": [
        {{"criterion": "name", "score": 1-5, "justification": "why"}},
        ...
    ],
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"]
}}
"""

    completion = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    import json

    result_data = json.loads(completion.choices[0].message.content or "{}")

    scores = [RubricScore(**s) for s in result_data.get("scores", [])]
    avg_score = sum(s.score for s in scores) / len(scores) if scores else 0.0

    return MultiAspectResult(
        scores=scores,
        overall_score=avg_score,
        strengths=result_data.get("strengths", []),
        weaknesses=result_data.get("weaknesses", []),
    )


async def main() -> None:
    """Example usage of LLM-as-Judge evaluation."""
    client = AsyncOpenAI()

    print("⚖️  LLM-as-Judge Evaluation Example\n")

    # Example 1: Pairwise comparison
    print("Example 1: Pairwise Comparison")
    query = "Explain quantum computing in simple terms."
    response_a = "Quantum computing uses quantum bits that can be 0 and 1 at the same time."
    response_b = (
        "Quantum computers leverage quantum mechanics principles like superposition and "
        "entanglement to perform calculations exponentially faster than classical "
        "computers for certain problems."
    )

    result = await pairwise_comparison(client, query, response_a, response_b)
    print(f"  Winner: {result.winner}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Reasoning: {result.reasoning}\n")

    # Example 2: Rubric-based evaluation
    print("Example 2: Rubric-Based Evaluation")
    rubric = {
        "Accuracy": "Factual correctness of information",
        "Clarity": "Ease of understanding for target audience",
        "Completeness": "Coverage of important aspects",
        "Conciseness": "Brevity without losing essential information",
    }

    eval_result = await rubric_based_evaluation(
        client,
        query="What is machine learning?",
        response="Machine learning is a subset of AI where systems learn from data "
        "without explicit programming.",
        rubric=rubric,
    )

    print(f"  Overall Score: {eval_result.overall_score:.2f}/5.0")
    print("  Scores by Criterion:")
    for score in eval_result.scores:
        print(f"    - {score.criterion}: {score.score}/5 - {score.justification}")

    print(f"\n  Strengths: {', '.join(eval_result.strengths)}")
    print(f"  Weaknesses: {', '.join(eval_result.weaknesses)}")


if __name__ == "__main__":
    asyncio.run(main())
