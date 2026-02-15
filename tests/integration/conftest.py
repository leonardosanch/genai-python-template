"""Integration test fixtures.

Provides test clients and container mocks for API integration testing.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.infrastructure.container import Container
from src.interfaces.api.main import app


@pytest.fixture
def mock_container() -> Container:
    container = AsyncMock(spec=Container)
    container.llm_adapter = AsyncMock()
    container.vector_store = AsyncMock()
    return container  # type: ignore[return-value]


@pytest.fixture
async def async_client() -> AsyncClient:  # type: ignore[misc]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client  # type: ignore[misc]
