# tests/unit/test_reranker.py
"""Tests for LLM-based reranker."""

from unittest.mock import AsyncMock

import pytest

from src.domain.entities.document import Document
from src.infrastructure.rag.llm_reranker import LLMReranker, RelevanceScore


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def reranker(mock_llm: AsyncMock) -> LLMReranker:
    return LLMReranker(llm=mock_llm)


def _make_doc(content: str, doc_id: str) -> Document:
    return Document(content=content, id=doc_id, metadata={})


class TestLLMReranker:
    async def test_rerank_sorts_by_score(self, reranker: LLMReranker, mock_llm: AsyncMock) -> None:
        docs = [_make_doc("Low relevance", "d1"), _make_doc("High relevance", "d2")]
        mock_llm.generate_structured.side_effect = [
            RelevanceScore(score=0.3, reasoning="Low"),
            RelevanceScore(score=0.9, reasoning="High"),
        ]

        result = await reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].id == "d2"
        assert result[0].score == 0.9
        assert result[1].id == "d1"

    async def test_rerank_respects_top_k(self, reranker: LLMReranker, mock_llm: AsyncMock) -> None:
        docs = [_make_doc(f"Doc {i}", f"d{i}") for i in range(5)]
        mock_llm.generate_structured.side_effect = [
            RelevanceScore(score=i * 0.2, reasoning=f"Score {i}") for i in range(5)
        ]

        result = await reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2

    async def test_rerank_empty_documents(self, reranker: LLMReranker) -> None:
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    async def test_rerank_handles_llm_failure(
        self, reranker: LLMReranker, mock_llm: AsyncMock
    ) -> None:
        docs = [_make_doc("Good doc", "d1"), _make_doc("Failed doc", "d2")]
        mock_llm.generate_structured.side_effect = [
            RelevanceScore(score=0.8, reasoning="Good"),
            RuntimeError("LLM failed"),
        ]

        result = await reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        # Failed doc gets score 0.0, so it should be last
        assert result[0].id == "d1"
        assert result[0].score == 0.8
        assert result[1].score == 0.0
