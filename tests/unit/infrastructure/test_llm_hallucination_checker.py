"""Tests for LLMHallucinationChecker — LLM-as-judge faithfulness verification."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.domain.ports.llm_port import LLMPort
from src.infrastructure.hallucination.llm_checker import (
    FaithfulnessJudgment,
    LLMHallucinationChecker,
)


@pytest.fixture()
def mock_llm() -> LLMPort:
    return create_autospec(LLMPort, instance=True)


@pytest.fixture()
def checker(mock_llm: LLMPort) -> LLMHallucinationChecker:
    return LLMHallucinationChecker(llm=mock_llm)


class TestLLMHallucinationChecker:
    """Tests for the LLM-based hallucination checker."""

    @pytest.mark.asyncio()
    async def test_grounded_answer(
        self,
        checker: LLMHallucinationChecker,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=FaithfulnessJudgment(
                faithfulness_score=0.95,
                unsupported_claims=[],
                citations=["The sky is blue."],
            ),
        )

        result = await checker.verify(
            answer="The sky is blue.",
            context="The sky is blue during the day.",
            query="What color is the sky?",
        )

        assert result.is_grounded is True
        assert result.faithfulness_score == 0.95
        assert result.unsupported_claims == []
        assert len(result.citations) == 1

    @pytest.mark.asyncio()
    async def test_hallucinated_answer(
        self,
        checker: LLMHallucinationChecker,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=FaithfulnessJudgment(
                faithfulness_score=0.2,
                unsupported_claims=["The sky is green", "Stars are square"],
                citations=[],
            ),
        )

        result = await checker.verify(
            answer="The sky is green and stars are square.",
            context="The sky is blue. Stars are spherical.",
            query="Describe the sky.",
        )

        assert result.is_grounded is False
        assert result.faithfulness_score == 0.2
        assert len(result.unsupported_claims) == 2

    @pytest.mark.asyncio()
    async def test_passes_correct_prompt(
        self,
        checker: LLMHallucinationChecker,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=FaithfulnessJudgment(faithfulness_score=0.8),
        )

        await checker.verify(answer="A", context="C", query="Q")

        call_args = mock_llm.generate_structured.call_args
        prompt = call_args[0][0]
        assert "C" in prompt  # context present
        assert "Q" in prompt  # query present
        assert "A" in prompt  # answer present
        assert call_args[1]["schema"] == FaithfulnessJudgment
        assert call_args[1]["temperature"] == 0.0
