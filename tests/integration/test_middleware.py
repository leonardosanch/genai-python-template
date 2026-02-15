# tests/integration/test_middleware.py
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from src.application.use_cases.query_rag import QueryRAGUseCase
from src.domain.exceptions import DomainError, LLMError
from src.interfaces.api.main import app
from src.interfaces.api.routes.rag_routes import get_use_case


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def mock_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture to mock structlog's logger and capture logs."""
    mock_log_processor = MagicMock()
    mock_logger_factory = MagicMock(return_value=mock_log_processor)

    monkeypatch.setattr(structlog, "PrintLoggerFactory", mock_logger_factory)
    monkeypatch.setattr(structlog, "get_logger", lambda: mock_log_processor)

    # Reconfigure structlog with the mock factory
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=mock_logger_factory,
    )
    return mock_log_processor


class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_correlation_id_in_response_header_and_logs(
        self, client: AsyncClient, mock_logger: MagicMock
    ) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        correlation_id = response.headers["X-Correlation-ID"]

        # Check logs for correlation_id
        mock_logger.info.assert_called_once()
        log_entry = json.loads(
            mock_logger.info.call_args[0][0]
        )  # Get the JSON string from the first arg
        assert log_entry["event"] == "api_request"
        assert log_entry["correlation_id"] == correlation_id
        assert log_entry["method"] == "GET"
        assert log_entry["path"] == "/health"
        assert log_entry["status_code"] == 200


class TestErrorHandler:
    @pytest.mark.asyncio
    async def test_generic_domain_error_returns_500(
        self, client: AsyncClient, mock_logger: MagicMock
    ) -> None:
        def mock_use_case_error() -> AsyncMock:
            mock = AsyncMock(spec=QueryRAGUseCase)
            mock.execute.side_effect = DomainError("Something bad happened")
            return mock

        app.dependency_overrides[get_use_case] = mock_use_case_error

        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test query"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Something bad happened"
        assert "correlation_id" in data

        # Check that error was logged as critical
        mock_logger.error.assert_called_once()
        log_entry = json.loads(mock_logger.error.call_args[0][0])
        assert log_entry["event"] == "domain_exception_caught"
        assert log_entry["status_code"] == 500

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_llm_error_returns_502(self, client: AsyncClient, mock_logger: MagicMock) -> None:
        def mock_use_case_llm_error() -> AsyncMock:
            mock = AsyncMock(spec=QueryRAGUseCase)
            mock.execute.side_effect = LLMError("LLM failed")
            return mock

        app.dependency_overrides[get_use_case] = mock_use_case_llm_error

        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "test query"},
        )
        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "LLM failed"
        assert "correlation_id" in data

        # Check that error was logged as error
        mock_logger.error.assert_called_once()
        log_entry = json.loads(mock_logger.error.call_args[0][0])
        assert log_entry["event"] == "domain_exception_caught"
        assert log_entry["status_code"] == 502

        app.dependency_overrides.clear()
