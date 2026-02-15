"""Unit tests for LLMHallucinationChecker adapter."""

from unittest.mock import AsyncMock

import pytest

from src.domain.ports.llm_port import LLMPort
from src.domain.value_objects.verification_result import VerificationResult
from src.infrastructure.hallucination.llm_checker import (
    FaithfulnessJudgment,
    LLMHallucinationChecker,
)


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock(spec=LLMPort)


class TestLLMHallucinationChecker:
    @pytest.mark.asyncio
    async def test_grounded_answer(self, mock_llm: AsyncMock) -> None:
        mock_llm.generate_structured.return_value = FaithfulnessJudgment(
            faithfulness_score=0.95,
            unsupported_claims=[],
            citations=["Python is a programming language"],
        )

        checker = LLMHallucinationChecker(llm=mock_llm)
        result = await checker.verify(
            answer="Python is a programming language.",
            context="Python is a programming language created by Guido.",
            query="What is Python?",
        )

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is True
        assert result.faithfulness_score == 0.95
        assert result.unsupported_claims == []
        assert len(result.citations) == 1

    @pytest.mark.asyncio
    async def test_hallucinated_answer(self, mock_llm: AsyncMock) -> None:
        mock_llm.generate_structured.return_value = FaithfulnessJudgment(
            faithfulness_score=0.3,
            unsupported_claims=["Python was created in 2020"],
            citations=[],
        )

        checker = LLMHallucinationChecker(llm=mock_llm)
        result = await checker.verify(
            answer="Python was created in 2020.",
            context="Python was created by Guido van Rossum in 1991.",
            query="When was Python created?",
        )

        assert result.is_grounded is False
        assert result.faithfulness_score == 0.3
        assert "Python was created in 2020" in result.unsupported_claims

    @pytest.mark.asyncio
    async def test_uses_temperature_zero(self, mock_llm: AsyncMock) -> None:
        mock_llm.generate_structured.return_value = FaithfulnessJudgment(
            faithfulness_score=0.9,
            unsupported_claims=[],
            citations=[],
        )

        checker = LLMHallucinationChecker(llm=mock_llm)
        await checker.verify(answer="answer", context="context", query="query")

        call_kwargs = mock_llm.generate_structured.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_uses_structured_output(self, mock_llm: AsyncMock) -> None:
        mock_llm.generate_structured.return_value = FaithfulnessJudgment(
            faithfulness_score=0.8,
            unsupported_claims=[],
            citations=[],
        )

        checker = LLMHallucinationChecker(llm=mock_llm)
        await checker.verify(answer="a", context="c", query="q")

        call_kwargs = mock_llm.generate_structured.call_args
        assert call_kwargs.kwargs["schema"] is FaithfulnessJudgment

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self, mock_llm: AsyncMock) -> None:
        mock_llm.generate_structured.return_value = FaithfulnessJudgment(
            faithfulness_score=0.9,
            unsupported_claims=[],
            citations=[],
        )

        checker = LLMHallucinationChecker(llm=mock_llm)
        await checker.verify(
            answer="test answer",
            context="test context",
            query="test query",
        )

        prompt_arg = mock_llm.generate_structured.call_args.args[0]
        assert "test answer" in prompt_arg
        assert "test context" in prompt_arg
        assert "test query" in prompt_arg
