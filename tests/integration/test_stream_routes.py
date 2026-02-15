# tests/integration/test_stream_routes.py
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.ports.llm_port import LLMPort
from src.infrastructure.container import Container
from src.interfaces.api.main import app


@pytest.fixture
def mock_llm():
    """Mock LLM Port that streams tokens."""
    llm = MagicMock(spec=LLMPort)

    async def token_generator(prompt: str):
        tokens = ["Hello", " ", "World", "!"]
        for token in tokens:
            yield token

    llm.stream = token_generator
    return llm


@pytest.fixture
def override_container(mock_llm):
    """Override the container to use the mock LLM."""
    container = MagicMock(spec=Container)
    container.llm_adapter = mock_llm

    # We need to override the dependency in FastAPI.
    # Since the route uses Depends(lambda: Container()),
    # we might need to override app.state.container.
    # OR override the dependency if we used a proper
    # dependency provider function.
    #
    # In main.py:
    # app.state.container = Container(...)
    #
    # In routes:
    # container = Depends(lambda: Container())
    # -> This creates a NEW container every time!
    #
    # Valid point:
    # The implementation in main.py sets app.state.container,
    # but routes use Depends(lambda: Container()).
    # The current implementation in routes creates a new
    # Container(), which creates new adapters.
    #
    # This makes testing hard without mocking the Container
    # class or the dependency.

    # Let's try to mock the Container class init or use
    # app.dependency_overrides if we refactor.
    #
    # But given the current code:
    # container = Depends(lambda: Container())
    #
    # We should probably refactor routes to use
    # `request.app.state.container` or a singleton provider.

    # FOR NOW:
    # We will monkeypatch Container in the route module
    # scope or ensure we can inject.
    pass
    return container


# Implementation Note:
# The current route `stream_chat` does:
# container: Container = Depends(lambda: Container())
#
# This instantiates a new Container().
#
# To test this with a mock, we should ideally change the
# route to use a dependency that we can override.
#
# OR we can mock
# `src.interfaces.api.routes.stream_routes.Container`.


@pytest.mark.asyncio
async def test_stream_chat_success(mock_llm, monkeypatch):
    """Test successful streaming response."""

    # Mock the Container class to return a mock instance
    # with our llm_adapter
    mock_container_instance = MagicMock()
    mock_container_instance.llm_adapter = mock_llm

    def mock_container_factory(*args, **kwargs):
        return mock_container_instance

    # We need to mock the Container class in the route's
    # namespace
    monkeypatch.setattr(
        "src.interfaces.api.routes.stream_routes.Container",
        mock_container_factory,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/stream/chat",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Read the stream.
        # httpx StreamingResponse needs explicit handling,
        # or we can use .aread() if not consuming line by line.
        # For SSE verification we want the raw body.
        content = await response.aread()
        decoded = content.decode("utf-8")

        # Check for SSE format
        assert 'data: {"token": "Hello"}\n\n' in decoded
        assert 'data: {"token": "World"}\n\n' in decoded
        assert "data: [DONE]\n\n" in decoded
