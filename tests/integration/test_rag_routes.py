"""Integration tests for RAG routes.

Reference test showing:
- FastAPI TestClient with httpx
- Dependency override for mocking ports
- Full request/response cycle validation
- Error handling verification
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.application.use_cases.query_rag import RAGAnswer
from src.domain.exceptions import LLMError, LLMRateLimitError, RetrievalError
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort
from src.interfaces.api.main import app
from src.interfaces.api.routes.rag_routes import get_llm, get_retriever


@pytest.fixture
def mock_retriever() -> AsyncMock:
    return AsyncMock(spec=RetrieverPort)


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock(spec=LLMPort)
    llm.generate_structured.return_value = RAGAnswer(
        answer="Test answer",
        model="gpt-4o-test",
    )
    return llm


@pytest.fixture
async def client(
    mock_llm: AsyncMock, mock_retriever: AsyncMock
) -> AsyncGenerator[AsyncClient, None]:
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


class TestRAGQueryEndpoint:
    @pytest.mark.asyncio
    async def test_successful_query(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "What is Python?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["model"] == "gpt-4o-test"

    @pytest.mark.asyncio
    async def test_query_with_custom_top_k(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test", "top_k": 3},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_top_k_out_of_range(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_llm_rate_limit_returns_429(
        self, client: AsyncClient, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate_structured.side_effect = LLMRateLimitError("rate limited")
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test"},
        )
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_llm_error_returns_502(self, client: AsyncClient, mock_llm: AsyncMock) -> None:
        mock_llm.generate_structured.side_effect = LLMError("failed")
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test"},
        )
        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_retrieval_error_returns_502(
        self, client: AsyncClient, mock_retriever: AsyncMock
    ) -> None:
        mock_retriever.retrieve.side_effect = RetrievalError("db down")
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test"},
        )
        assert response.status_code == 502
