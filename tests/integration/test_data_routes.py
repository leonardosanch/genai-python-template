"""Integration tests for data engineering routes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.value_objects.data_quality_result import DataQualityResult
from src.interfaces.api.main import app


@pytest.fixture
def mock_container(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the container on app state."""
    container = MagicMock()

    # Mock data source
    container.data_source.read_records = AsyncMock(return_value=[{"id": 1, "name": "test"}])
    container.data_source.read_schema = AsyncMock(return_value=None)

    # Mock data sink
    container.data_sink.write_records = AsyncMock(return_value=1)

    # Mock data validator
    container.data_validator.validate = AsyncMock(
        return_value=DataQualityResult(
            is_valid=True,
            total_records=1,
            valid_records=1,
            invalid_records=0,
            errors=(),
        )
    )

    # Mock event bus
    container.event_bus.publish = AsyncMock()

    # Mock close
    container.close = AsyncMock()

    return container


@pytest.fixture
async def client(mock_container: MagicMock) -> AsyncClient:
    app.state.container = mock_container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_etl_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/data/etl",
        json={"source_uri": "test.csv", "sink_uri": "output.csv"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["records_extracted"] == 1


@pytest.mark.asyncio
async def test_validate_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/data/validate",
        json={"source_uri": "test.csv"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
