"""Tests for SummarizeDocumentUseCase."""

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.summarize_document import SummarizeDocumentUseCase, Summary
from src.domain.entities.document import Document
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock(spec=RetrieverPort)
    retriever.retrieve.return_value = [
        Document(content="Python is versatile", metadata={"source": "wiki"}),
        Document(content="FastAPI is async", metadata={"source": "docs"}),
    ]
    return retriever


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock(spec=LLMPort)
    llm.generate_structured.return_value = Summary(
        title="Python Overview",
        key_points=["Python is versatile", "FastAPI is async"],
        confidence=0.9,
        sources=["wiki", "docs"],
    )
    return llm


@pytest.mark.asyncio
async def test_execute_retrieves_documents(mock_llm: AsyncMock, mock_retriever: AsyncMock) -> None:
    uc = SummarizeDocumentUseCase(llm=mock_llm, retriever=mock_retriever)
    await uc.execute("summarize python")
    mock_retriever.retrieve.assert_awaited_once_with("summarize python", top_k=5)


@pytest.mark.asyncio
async def test_execute_calls_llm_with_context(
    mock_llm: AsyncMock, mock_retriever: AsyncMock
) -> None:
    uc = SummarizeDocumentUseCase(llm=mock_llm, retriever=mock_retriever)
    await uc.execute("summarize python")
    call_args = mock_llm.generate_structured.call_args
    prompt = call_args[0][0]
    assert "Python is versatile" in prompt
    assert "FastAPI is async" in prompt
    assert call_args[1]["schema"] is Summary


@pytest.mark.asyncio
async def test_execute_returns_summary(mock_llm: AsyncMock, mock_retriever: AsyncMock) -> None:
    uc = SummarizeDocumentUseCase(llm=mock_llm, retriever=mock_retriever)
    result = await uc.execute("summarize python")
    assert isinstance(result, Summary)
    assert result.title == "Python Overview"
    assert len(result.key_points) == 2
    assert result.confidence == 0.9
