"""Tests for ResilientLLMPort — circuit breaker decorator for LLM calls."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.domain.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)
from src.infrastructure.resilience.resilient_llm import ResilientLLMPort


def _make_cb_config(
    threshold: int = 2,
    timeout: float = 0.1,
) -> CircuitBreakerConfig:
    """Small threshold + short timeout for fast tests."""
    return CircuitBreakerConfig(
        failure_threshold=threshold,
        recovery_timeout=timeout,
        expected_exceptions=(LLMError,),
    )


@pytest.fixture()
def mock_llm() -> LLMPort:
    return create_autospec(LLMPort, instance=True)


@pytest.fixture()
def resilient(mock_llm: LLMPort) -> ResilientLLMPort:
    return ResilientLLMPort(mock_llm, config=_make_cb_config())


class TestGenerate:
    """Circuit breaker around generate()."""

    @pytest.mark.asyncio()
    async def test_success_passes_through(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate = AsyncMock(return_value="hello")
        result = await resilient.generate("prompt")
        assert result == "hello"
        mock_llm.generate.assert_awaited_once_with("prompt")

    @pytest.mark.asyncio()
    async def test_failure_counts_toward_threshold(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate = AsyncMock(side_effect=LLMTimeoutError("timeout"))

        with pytest.raises(LLMTimeoutError):
            await resilient.generate("p")

        assert resilient.circuit_breaker.failure_count == 1
        assert resilient.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio()
    async def test_circuit_opens_after_threshold(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate = AsyncMock(side_effect=LLMError("fail"))

        for _ in range(2):
            with pytest.raises(LLMError):
                await resilient.generate("p")

        assert resilient.circuit_breaker.state == CircuitState.OPEN

        # Next call should fail fast
        with pytest.raises(CircuitBreakerError, match="Circuit is OPEN"):
            await resilient.generate("p")

    @pytest.mark.asyncio()
    async def test_non_llm_error_does_not_trip_breaker(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate = AsyncMock(side_effect=ValueError("bad"))

        with pytest.raises(ValueError, match="bad"):
            await resilient.generate("p")

        assert resilient.circuit_breaker.failure_count == 0


class TestGenerateStructured:
    """Circuit breaker around generate_structured()."""

    @pytest.mark.asyncio()
    async def test_success(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        expected = Answer(text="ok")
        mock_llm.generate_structured = AsyncMock(return_value=expected)

        result = await resilient.generate_structured("p", schema=Answer)
        assert result == expected

    @pytest.mark.asyncio()
    async def test_rate_limit_trips_breaker(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            side_effect=LLMRateLimitError("429"),
        )

        for _ in range(2):
            with pytest.raises(LLMRateLimitError):
                await resilient.generate_structured("p", schema=object)

        assert resilient.circuit_breaker.state == CircuitState.OPEN


class TestStream:
    """Circuit breaker around stream()."""

    @pytest.mark.asyncio()
    async def test_success_streams_tokens(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        async def fake_stream(prompt: str, **kwargs):  # noqa: ANN003, ARG001, ANN001
            for t in ["a", "b", "c"]:
                yield t

        mock_llm.stream = fake_stream

        tokens = [t async for t in resilient.stream("p")]
        assert tokens == ["a", "b", "c"]
        assert resilient.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio()
    async def test_failure_during_stream_counts(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        async def failing_stream(prompt: str, **kwargs):  # noqa: ANN003, ARG001, ANN001
            yield "partial"
            raise LLMError("mid-stream failure")

        mock_llm.stream = failing_stream

        with pytest.raises(LLMError, match="mid-stream"):
            async for _ in resilient.stream("p"):
                pass

        assert resilient.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio()
    async def test_stream_rejected_when_open(
        self,
        resilient: ResilientLLMPort,
        mock_llm: LLMPort,
    ) -> None:
        # Force circuit open
        mock_llm.generate = AsyncMock(side_effect=LLMError("fail"))
        for _ in range(2):
            with pytest.raises(LLMError):
                await resilient.generate("p")

        assert resilient.circuit_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            async for _ in resilient.stream("p"):
                pass


class TestRecovery:
    """Half-open recovery after timeout."""

    @pytest.mark.asyncio()
    async def test_recovers_after_timeout(
        self,
        mock_llm: LLMPort,
    ) -> None:
        import asyncio

        resilient = ResilientLLMPort(
            mock_llm,
            config=_make_cb_config(threshold=1, timeout=0.05),
        )

        # Trip the breaker
        mock_llm.generate = AsyncMock(side_effect=LLMError("fail"))
        with pytest.raises(LLMError):
            await resilient.generate("p")

        assert resilient.circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Next successful call should close the circuit
        mock_llm.generate = AsyncMock(return_value="recovered")
        result = await resilient.generate("p")
        assert result == "recovered"
        assert resilient.circuit_breaker.state == CircuitState.CLOSED
