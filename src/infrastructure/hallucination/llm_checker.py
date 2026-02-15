"""LLM-based hallucination checker — uses any LLMPort to verify faithfulness."""

import structlog
from pydantic import BaseModel, Field

from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.ports.llm_port import LLMPort
from src.domain.value_objects.verification_result import VerificationResult

logger = structlog.get_logger()

FAITHFULNESS_JUDGE_PROMPT_V1 = """You are a faithfulness judge. Your task is to evaluate whether
an AI-generated answer is fully supported by the provided context.

## Context
{context}

## Question
{query}

## Answer to evaluate
{answer}

## Instructions
1. Identify each claim in the answer.
2. For each claim, check if it is directly supported by the context.
3. List any claims NOT supported by the context.
4. List context passages that support the answer (citations).
5. Assign a faithfulness score from 0.0 (completely hallucinated) to 1.0 (fully grounded).

Be strict: if a claim cannot be verified from the context, it is unsupported.
"""


class FaithfulnessJudgment(BaseModel):
    """Structured output schema for the faithfulness judge."""

    faithfulness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score from 0.0 (hallucinated) to 1.0 (grounded)",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by the context",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Context passages that support the answer",
    )


class LLMHallucinationChecker(HallucinationCheckerPort):
    """Verifies answer faithfulness using an LLM as judge.

    Depends on LLMPort — works with any provider (OpenAI, Anthropic, etc.).
    Uses structured output for deterministic parsing.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def verify(self, answer: str, context: str, query: str) -> VerificationResult:
        prompt = FAITHFULNESS_JUDGE_PROMPT_V1.format(
            context=context,
            query=query,
            answer=answer,
        )

        judgment = await self._llm.generate_structured(
            prompt,
            schema=FaithfulnessJudgment,
            temperature=0.0,
        )

        is_grounded = len(judgment.unsupported_claims) == 0

        logger.info(
            "hallucination_check_completed",
            faithfulness_score=judgment.faithfulness_score,
            is_grounded=is_grounded,
            unsupported_claims_count=len(judgment.unsupported_claims),
        )

        return VerificationResult(
            is_grounded=is_grounded,
            faithfulness_score=judgment.faithfulness_score,
            unsupported_claims=judgment.unsupported_claims,
            citations=judgment.citations,
        )
