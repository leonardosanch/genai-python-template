"""Tests for LLMReranker — LLM-based document reranking."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.domain.entities.document import Document
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.rag.llm_reranker import LLMReranker, RelevanceScore


@pytest.fixture()
def mock_llm() -> LLMPort:
    return create_autospec(LLMPort, instance=True)


@pytest.fixture()
def reranker(mock_llm: LLMPort) -> LLMReranker:
    return LLMReranker(llm=mock_llm)


def _doc(content: str, doc_id: str = "d1") -> Document:
    return Document(content=content, metadata={}, id=doc_id)


class TestLLMReranker:
    """Tests for the LLM-based reranker."""

    @pytest.mark.asyncio()
    async def test_empty_documents_returns_empty(self, reranker: LLMReranker) -> None:
        result = await reranker.rerank(query="test", documents=[], top_k=5)
        assert result == []

    @pytest.mark.asyncio()
    async def test_reranks_by_score(self, reranker: LLMReranker, mock_llm: LLMPort) -> None:
        docs = [_doc("low relevance", "d1"), _doc("high relevance", "d2"), _doc("mid", "d3")]
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                RelevanceScore(score=0.2, reasoning="low"),
                RelevanceScore(score=0.9, reasoning="high"),
                RelevanceScore(score=0.5, reasoning="mid"),
            ],
        )

        result = await reranker.rerank(query="important topic", documents=docs, top_k=3)

        assert len(result) == 3
        assert result[0].id == "d2"  # highest score first
        assert result[0].score == 0.9
        assert result[1].id == "d3"
        assert result[2].id == "d1"

    @pytest.mark.asyncio()
    async def test_respects_top_k(self, reranker: LLMReranker, mock_llm: LLMPort) -> None:
        docs = [_doc(f"doc {i}", f"d{i}") for i in range(5)]
        mock_llm.generate_structured = AsyncMock(
            side_effect=[RelevanceScore(score=0.1 * (i + 1), reasoning="r") for i in range(5)],
        )

        result = await reranker.rerank(query="q", documents=docs, top_k=2)
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_handles_scoring_failure_gracefully(
        self,
        reranker: LLMReranker,
        mock_llm: LLMPort,
    ) -> None:
        docs = [_doc("good", "d1"), _doc("error", "d2")]
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                RelevanceScore(score=0.8, reasoning="ok"),
                RuntimeError("LLM unavailable"),
            ],
        )

        result = await reranker.rerank(query="q", documents=docs, top_k=2)

        assert len(result) == 2
        assert result[0].id == "d1"  # scored 0.8
        assert result[0].score == 0.8
        assert result[1].score == 0.0  # failed, defaults to 0.0
