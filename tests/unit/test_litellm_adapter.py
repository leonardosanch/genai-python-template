"""Tests for LiteLLM adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from src.domain.exceptions import LLMRateLimitError, LLMTimeoutError


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not installed")
class TestLiteLLMAdapter:
    """Tests for the LiteLLM adapter."""

    def _make_adapter(self) -> "LiteLLMAdapter":  # noqa: F821
        from src.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        return LiteLLMAdapter(model="gpt-4o", api_key="test-key")

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        adapter = self._make_adapter()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.generate("test prompt")

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_generate_timeout(self) -> None:
        adapter = self._make_adapter()

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.Timeout(
                message="request timed out", model="gpt-4o", llm_provider="openai"
            ),
        ):
            with pytest.raises(LLMTimeoutError):
                await adapter.generate("test prompt")

    @pytest.mark.asyncio
    async def test_generate_rate_limit(self) -> None:
        adapter = self._make_adapter()

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.RateLimitError(
                message="rate limited",
                model="gpt-4o",
                llm_provider="openai",
            ),
        ):
            with pytest.raises(LLMRateLimitError):
                await adapter.generate("test prompt")

    @pytest.mark.asyncio
    async def test_stream(self) -> None:
        adapter = self._make_adapter()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        async def mock_stream(*args: object, **kwargs: object) -> AsyncMock:
            mock = AsyncMock()
            mock.__aiter__ = MagicMock(return_value=iter([chunk1, chunk2]))
            return mock

        # Use a simpler approach: mock acompletion to return async iterable
        async def fake_acompletion(*args: object, **kwargs: object) -> MagicMock:
            class FakeStream:
                def __init__(self) -> None:
                    self._chunks = [chunk1, chunk2]
                    self._index = 0

                def __aiter__(self) -> "FakeStream":
                    return self

                async def __anext__(self) -> MagicMock:
                    if self._index >= len(self._chunks):
                        raise StopAsyncIteration
                    chunk = self._chunks[self._index]
                    self._index += 1
                    return chunk

            return FakeStream()  # type: ignore[return-value]

        with patch("litellm.acompletion", side_effect=fake_acompletion):
            tokens = []
            async for token in adapter.stream("test"):
                tokens.append(token)

        assert tokens == ["Hello", " world"]
