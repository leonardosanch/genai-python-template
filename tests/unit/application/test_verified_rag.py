"""Tests for VerifiedRAGUseCase — RAG with hallucination checking."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.application.use_cases.query_rag import RAGAnswer
from src.application.use_cases.verified_rag import VerifiedRAGUseCase
from src.domain.entities.document import Document
from src.domain.exceptions import HallucinationError
from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.value_objects.verification_result import VerificationResult


@pytest.fixture()
def mock_llm() -> LLMPort:
    return create_autospec(LLMPort, instance=True)


@pytest.fixture()
def mock_retriever() -> RetrieverPort:
    return create_autospec(RetrieverPort, instance=True)


@pytest.fixture()
def mock_checker() -> HallucinationCheckerPort:
    return create_autospec(HallucinationCheckerPort, instance=True)


def _make_doc(content: str = "Test content") -> Document:
    return Document(content=content, metadata={"source": "test"})


class TestVerifiedRAGUseCase:
    """Tests for the verified RAG pipeline."""

    @pytest.mark.asyncio()
    async def test_passes_when_grounded(
        self,
        mock_llm: LLMPort,
        mock_retriever: RetrieverPort,
        mock_checker: HallucinationCheckerPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=RAGAnswer(answer="Grounded answer", model="gpt-4"),
        )
        mock_retriever.retrieve = AsyncMock(return_value=[_make_doc()])
        mock_checker.verify = AsyncMock(
            return_value=VerificationResult(
                is_grounded=True,
                faithfulness_score=0.9,
            ),
        )

        uc = VerifiedRAGUseCase(
            llm=mock_llm,
            retriever=mock_retriever,
            hallucination_checker=mock_checker,
        )
        answer, verification = await uc.execute("test query")

        assert answer.answer == "Grounded answer"
        assert verification.faithfulness_score == 0.9

    @pytest.mark.asyncio()
    async def test_raises_hallucination_error_after_retries(
        self,
        mock_llm: LLMPort,
        mock_retriever: RetrieverPort,
        mock_checker: HallucinationCheckerPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=RAGAnswer(answer="Bad answer", model="gpt-4"),
        )
        mock_retriever.retrieve = AsyncMock(return_value=[_make_doc()])
        mock_checker.verify = AsyncMock(
            return_value=VerificationResult(
                is_grounded=False,
                faithfulness_score=0.3,
                unsupported_claims=["fabricated claim"],
            ),
        )

        uc = VerifiedRAGUseCase(
            llm=mock_llm,
            retriever=mock_retriever,
            hallucination_checker=mock_checker,
            faithfulness_threshold=0.7,
            max_retries=1,
        )

        with pytest.raises(HallucinationError, match="failed faithfulness verification"):
            await uc.execute("test query")

        # 1 initial + 1 retry = 2 calls
        assert mock_checker.verify.call_count == 2

    @pytest.mark.asyncio()
    async def test_retries_then_succeeds(
        self,
        mock_llm: LLMPort,
        mock_retriever: RetrieverPort,
        mock_checker: HallucinationCheckerPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=RAGAnswer(answer="Eventually good", model="gpt-4"),
        )
        mock_retriever.retrieve = AsyncMock(return_value=[_make_doc()])

        # First attempt fails, second passes
        mock_checker.verify = AsyncMock(
            side_effect=[
                VerificationResult(is_grounded=False, faithfulness_score=0.4),
                VerificationResult(is_grounded=True, faithfulness_score=0.85),
            ],
        )

        uc = VerifiedRAGUseCase(
            llm=mock_llm,
            retriever=mock_retriever,
            hallucination_checker=mock_checker,
            max_retries=1,
        )
        answer, verification = await uc.execute("query")
        assert verification.faithfulness_score == 0.85

    @pytest.mark.asyncio()
    async def test_custom_threshold(
        self,
        mock_llm: LLMPort,
        mock_retriever: RetrieverPort,
        mock_checker: HallucinationCheckerPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=RAGAnswer(answer="answer", model="m"),
        )
        mock_retriever.retrieve = AsyncMock(return_value=[_make_doc()])
        mock_checker.verify = AsyncMock(
            return_value=VerificationResult(is_grounded=True, faithfulness_score=0.5),
        )

        # Threshold=0.5 should pass with score=0.5
        uc = VerifiedRAGUseCase(
            llm=mock_llm,
            retriever=mock_retriever,
            hallucination_checker=mock_checker,
            faithfulness_threshold=0.5,
        )
        answer, verification = await uc.execute("q")
        assert verification.faithfulness_score == 0.5
