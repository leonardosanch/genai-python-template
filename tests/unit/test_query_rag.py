"""Unit tests for QueryRAGUseCase.

Reference test showing:
- Mocking ports with AsyncMock
- Testing use case orchestration logic
- Edge cases (empty retrieval, LLM errors)
"""

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.query_rag import QueryRAGUseCase, RAGAnswer
from src.domain.entities.document import Document
from src.domain.exceptions import LLMError, RetrievalError
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock(spec=RetrieverPort)
    retriever.retrieve.return_value = [
        Document(content="Python is versatile", metadata={"source": "wiki"}),
        Document(content="FastAPI uses async", metadata={"source": "docs"}),
    ]
    return retriever


@pytest.fixture
def mock_llm_port() -> AsyncMock:
    llm = AsyncMock(spec=LLMPort)
    llm.generate_structured.return_value = RAGAnswer(
        answer="Python is a versatile language used with FastAPI.",
        model="gpt-4o",
    )
    return llm


class TestQueryRAGUseCase:
    @pytest.mark.asyncio
    async def test_execute_returns_structured_answer(
        self, mock_llm_port: AsyncMock, mock_retriever: AsyncMock
    ) -> None:
        use_case = QueryRAGUseCase(llm=mock_llm_port, retriever=mock_retriever)
        result = await use_case.execute("What is Python?")

        assert isinstance(result, RAGAnswer)
        assert "Python" in result.answer
        mock_retriever.retrieve.assert_called_once_with("What is Python?", top_k=5)
        mock_llm_port.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_passes_top_k(
        self, mock_llm_port: AsyncMock, mock_retriever: AsyncMock
    ) -> None:
        use_case = QueryRAGUseCase(llm=mock_llm_port, retriever=mock_retriever)
        await use_case.execute("query", top_k=3)

        mock_retriever.retrieve.assert_called_once_with("query", top_k=3)

    @pytest.mark.asyncio
    async def test_execute_with_no_documents(self, mock_llm_port: AsyncMock) -> None:
        retriever = AsyncMock(spec=RetrieverPort)
        retriever.retrieve.return_value = []

        use_case = QueryRAGUseCase(llm=mock_llm_port, retriever=retriever)
        result = await use_case.execute("unknown topic")

        assert isinstance(result, RAGAnswer)
        mock_llm_port.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_propagates_llm_error(self, mock_retriever: AsyncMock) -> None:
        llm = AsyncMock(spec=LLMPort)
        llm.generate_structured.side_effect = LLMError("timeout")

        use_case = QueryRAGUseCase(llm=llm, retriever=mock_retriever)
        with pytest.raises(LLMError):
            await use_case.execute("query")

    @pytest.mark.asyncio
    async def test_execute_propagates_retrieval_error(self, mock_llm_port: AsyncMock) -> None:
        retriever = AsyncMock(spec=RetrieverPort)
        retriever.retrieve.side_effect = RetrievalError("db down")

        use_case = QueryRAGUseCase(llm=mock_llm_port, retriever=retriever)
        with pytest.raises(RetrievalError):
            await use_case.execute("query")
