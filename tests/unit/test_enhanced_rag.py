# tests/unit/test_enhanced_rag.py
"""Tests for enhanced RAG use case."""

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.enhanced_rag import EnhancedRAGUseCase
from src.domain.entities.document import Document


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.generate.return_value = "The answer is 42."
    return llm


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve.return_value = [
        Document(content="Doc 1", id="d1", metadata={}),
        Document(content="Doc 2", id="d2", metadata={}),
    ]
    return retriever


@pytest.fixture
def mock_reranker() -> AsyncMock:
    reranker = AsyncMock()
    reranker.rerank.return_value = [
        Document(content="Doc 1", id="d1", score=0.9, metadata={}),
    ]
    return reranker


@pytest.fixture
def mock_cache() -> AsyncMock:
    return AsyncMock()


class TestEnhancedRAGUseCase:
    async def test_basic_flow(self, mock_llm: AsyncMock, mock_retriever: AsyncMock) -> None:
        use_case = EnhancedRAGUseCase(llm=mock_llm, retriever=mock_retriever)
        result = await use_case.execute("What is life?")
        assert result.answer == "The answer is 42."
        assert len(result.sources) == 2
        assert result.cache_hit is False
        mock_retriever.retrieve.assert_called_once()

    async def test_with_reranker(
        self, mock_llm: AsyncMock, mock_retriever: AsyncMock, mock_reranker: AsyncMock
    ) -> None:
        use_case = EnhancedRAGUseCase(
            llm=mock_llm, retriever=mock_retriever, reranker=mock_reranker, top_k=3
        )
        result = await use_case.execute("query")
        # With reranker, retrieves top_k * 2 = 6
        mock_retriever.retrieve.assert_called_once_with("query", top_k=6)
        mock_reranker.rerank.assert_called_once()
        assert len(result.sources) == 1  # reranker returned 1

    async def test_cache_hit(
        self, mock_llm: AsyncMock, mock_retriever: AsyncMock, mock_cache: AsyncMock
    ) -> None:
        mock_cache.get.return_value = "cached answer"
        use_case = EnhancedRAGUseCase(llm=mock_llm, retriever=mock_retriever, cache=mock_cache)
        result = await use_case.execute("cached query")
        assert result.answer == "cached answer"
        assert result.cache_hit is True
        mock_retriever.retrieve.assert_not_called()
        mock_llm.generate.assert_not_called()

    async def test_cache_miss_then_store(
        self, mock_llm: AsyncMock, mock_retriever: AsyncMock, mock_cache: AsyncMock
    ) -> None:
        mock_cache.get.return_value = None
        use_case = EnhancedRAGUseCase(llm=mock_llm, retriever=mock_retriever, cache=mock_cache)
        result = await use_case.execute("new query")
        assert result.cache_hit is False
        mock_cache.set.assert_called_once_with("new query", "The answer is 42.")

    async def test_no_documents_still_generates(self, mock_llm: AsyncMock) -> None:
        retriever = AsyncMock()
        retriever.retrieve.return_value = []
        use_case = EnhancedRAGUseCase(llm=mock_llm, retriever=retriever)
        result = await use_case.execute("obscure query")
        assert result.answer == "The answer is 42."
        assert result.sources == []
