"""Evaluation tests for verified RAG pipeline — CI/CD quality gates.

All tests use mocked LLM/retriever (no real API calls).
Tests verify the verification pipeline behavior, not actual LLM quality.
"""

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.verified_rag import VerifiedRAGUseCase
from src.domain.exceptions import HallucinationError
from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.value_objects.verification_result import VerificationResult


class TestVerifiedRAGThresholds:
    """Verify that the pipeline correctly enforces faithfulness thresholds."""

    @pytest.mark.asyncio
    async def test_faithful_answer_passes(
        self,
        mock_llm_faithful: AsyncMock,
        mock_retriever: AsyncMock,
        mock_checker_grounded: AsyncMock,
    ) -> None:
        use_case = VerifiedRAGUseCase(
            llm=mock_llm_faithful,
            retriever=mock_retriever,
            hallucination_checker=mock_checker_grounded,
            faithfulness_threshold=0.7,
        )

        answer, verification = await use_case.execute("What is Python?")

        assert verification.faithfulness_score >= 0.7
        assert verification.is_grounded is True
        assert answer.answer != ""

    @pytest.mark.asyncio
    async def test_hallucinated_answer_raises(
        self,
        mock_llm_hallucinated: AsyncMock,
        mock_retriever: AsyncMock,
        mock_checker_hallucinated: AsyncMock,
    ) -> None:
        use_case = VerifiedRAGUseCase(
            llm=mock_llm_hallucinated,
            retriever=mock_retriever,
            hallucination_checker=mock_checker_hallucinated,
            faithfulness_threshold=0.7,
            max_retries=0,
        )

        with pytest.raises(HallucinationError, match="failed faithfulness"):
            await use_case.execute("When was Python created?")


class TestVerifiedRAGRetry:
    """Verify retry behavior when initial answer fails verification."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(
        self,
        mock_llm_faithful: AsyncMock,
        mock_retriever: AsyncMock,
    ) -> None:
        checker = AsyncMock(spec=HallucinationCheckerPort)
        checker.verify.side_effect = [
            VerificationResult(
                is_grounded=False,
                faithfulness_score=0.4,
                unsupported_claims=["bad claim"],
            ),
            VerificationResult(
                is_grounded=True,
                faithfulness_score=0.9,
                unsupported_claims=[],
                citations=["good source"],
            ),
        ]

        use_case = VerifiedRAGUseCase(
            llm=mock_llm_faithful,
            retriever=mock_retriever,
            hallucination_checker=checker,
            faithfulness_threshold=0.7,
            max_retries=1,
        )

        answer, verification = await use_case.execute("query")

        assert verification.faithfulness_score == 0.9
        assert checker.verify.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(
        self,
        mock_llm_hallucinated: AsyncMock,
        mock_retriever: AsyncMock,
        mock_checker_hallucinated: AsyncMock,
    ) -> None:
        use_case = VerifiedRAGUseCase(
            llm=mock_llm_hallucinated,
            retriever=mock_retriever,
            hallucination_checker=mock_checker_hallucinated,
            faithfulness_threshold=0.7,
            max_retries=2,
        )

        with pytest.raises(HallucinationError):
            await use_case.execute("query")

        # 1 initial + 2 retries = 3 total calls
        assert mock_checker_hallucinated.verify.call_count == 3


class TestFaithfulnessThresholds:
    """Boundary tests for faithfulness threshold enforcement."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("score", "threshold", "should_pass"),
        [
            (0.71, 0.7, True),
            (0.70, 0.7, True),
            (0.69, 0.7, False),
            (1.0, 0.7, True),
            (0.0, 0.7, False),
            (0.5, 0.5, True),
            (0.49, 0.5, False),
        ],
    )
    async def test_threshold_boundaries(
        self,
        mock_llm_faithful: AsyncMock,
        mock_retriever: AsyncMock,
        score: float,
        threshold: float,
        should_pass: bool,
    ) -> None:
        checker = AsyncMock(spec=HallucinationCheckerPort)
        checker.verify.return_value = VerificationResult(
            is_grounded=score >= threshold,
            faithfulness_score=score,
            unsupported_claims=[] if score >= threshold else ["claim"],
        )

        use_case = VerifiedRAGUseCase(
            llm=mock_llm_faithful,
            retriever=mock_retriever,
            hallucination_checker=checker,
            faithfulness_threshold=threshold,
            max_retries=0,
        )

        if should_pass:
            answer, verification = await use_case.execute("q")
            assert verification.faithfulness_score == score
        else:
            with pytest.raises(HallucinationError):
                await use_case.execute("q")
