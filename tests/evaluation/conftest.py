"""Fixtures for evaluation tests — no real LLM calls."""

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.query_rag import RAGAnswer
from src.domain.entities.document import Document
from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.value_objects.verification_result import VerificationResult
from src.infrastructure.hallucination.llm_checker import FaithfulnessJudgment


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            content="Python was created by Guido van Rossum in 1991.",
            metadata={"source": "wiki"},
        ),
        Document(
            content="FastAPI is a modern web framework for Python.",
            metadata={"source": "docs"},
        ),
    ]


@pytest.fixture
def faithful_answer() -> RAGAnswer:
    return RAGAnswer(
        answer="Python was created by Guido van Rossum in 1991.",
        model="test-model",
    )


@pytest.fixture
def hallucinated_answer() -> RAGAnswer:
    return RAGAnswer(
        answer="Python was created by Linus Torvalds in 2005.",
        model="test-model",
    )


@pytest.fixture
def faithful_judgment() -> FaithfulnessJudgment:
    return FaithfulnessJudgment(
        faithfulness_score=0.95,
        unsupported_claims=[],
        citations=["Python was created by Guido van Rossum in 1991."],
    )


@pytest.fixture
def hallucinated_judgment() -> FaithfulnessJudgment:
    return FaithfulnessJudgment(
        faithfulness_score=0.2,
        unsupported_claims=["Python was created by Linus Torvalds in 2005"],
        citations=[],
    )


@pytest.fixture
def mock_retriever(sample_docs: list[Document]) -> AsyncMock:
    retriever = AsyncMock(spec=RetrieverPort)
    retriever.retrieve.return_value = sample_docs
    return retriever


@pytest.fixture
def mock_llm_faithful(faithful_answer: RAGAnswer) -> AsyncMock:
    llm = AsyncMock(spec=LLMPort)
    llm.generate_structured.return_value = faithful_answer
    return llm


@pytest.fixture
def mock_llm_hallucinated(hallucinated_answer: RAGAnswer) -> AsyncMock:
    llm = AsyncMock(spec=LLMPort)
    llm.generate_structured.return_value = hallucinated_answer
    return llm


@pytest.fixture
def mock_checker_grounded() -> AsyncMock:
    checker = AsyncMock(spec=HallucinationCheckerPort)
    checker.verify.return_value = VerificationResult(
        is_grounded=True,
        faithfulness_score=0.95,
        unsupported_claims=[],
        citations=["Python was created by Guido van Rossum in 1991."],
    )
    return checker


@pytest.fixture
def mock_checker_hallucinated() -> AsyncMock:
    checker = AsyncMock(spec=HallucinationCheckerPort)
    checker.verify.return_value = VerificationResult(
        is_grounded=False,
        faithfulness_score=0.2,
        unsupported_claims=["Python was created by Linus Torvalds in 2005"],
        citations=[],
    )
    return checker
