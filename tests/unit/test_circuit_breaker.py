# tests/unit/test_circuit_breaker.py
"""Tests for the circuit breaker resilience pattern."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


@pytest.fixture
def config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.1)


@pytest.fixture
def cb(config: CircuitBreakerConfig) -> CircuitBreaker:
    return CircuitBreaker(config=config)


class TestCircuitBreakerStates:
    async def test_initial_state_closed(self, cb: CircuitBreaker) -> None:
        assert cb.state == CircuitState.CLOSED

    async def test_success_stays_closed(self, cb: CircuitBreaker) -> None:
        func = AsyncMock(return_value="ok")
        result = await cb.call(func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    async def test_opens_after_threshold(self, cb: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(func)
        assert cb.state == CircuitState.OPEN

    async def test_open_rejects_calls(self, cb: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(func)

        with pytest.raises(CircuitBreakerError, match="OPEN"):
            await cb.call(AsyncMock(return_value="ok"))

    async def test_half_open_after_recovery_timeout(self, cb: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(func)
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.15)  # Wait for recovery timeout

        success_func = AsyncMock(return_value="recovered")
        result = await cb.call(success_func)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self, cb: CircuitBreaker) -> None:
        fail_func = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fail_func)

        await asyncio.sleep(0.15)

        # Next call will transition to HALF_OPEN, then fail → back to OPEN
        with pytest.raises(RuntimeError):
            await cb.call(fail_func)
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerDecorator:
    async def test_decorator_success(self, cb: CircuitBreaker) -> None:
        @cb
        async def my_func(x: int) -> int:
            return x * 2

        result = await my_func(5)
        assert result == 10

    async def test_decorator_failure_tracking(self, cb: CircuitBreaker) -> None:
        call_count = 0

        @cb
        async def my_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                await my_func()

        assert cb.state == CircuitState.OPEN
        assert call_count == 3


class TestCircuitBreakerReset:
    async def test_manual_reset(self, cb: CircuitBreaker) -> None:
        func = AsyncMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(func)
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerConfig:
    async def test_custom_expected_exceptions(self) -> None:
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exceptions=(ValueError,),
        )
        cb = CircuitBreaker(config=config)

        # ValueError counts as failure
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(AsyncMock(side_effect=ValueError("bad")))
        assert cb.state == CircuitState.OPEN

    async def test_unexpected_exception_not_counted(self) -> None:
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exceptions=(ValueError,),
        )
        cb = CircuitBreaker(config=config)

        # TypeError is not in expected_exceptions — still raised but state doesn't change
        with pytest.raises(TypeError):
            await cb.call(AsyncMock(side_effect=TypeError("nope")))
        assert cb.state == CircuitState.CLOSED

    async def test_default_config(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
