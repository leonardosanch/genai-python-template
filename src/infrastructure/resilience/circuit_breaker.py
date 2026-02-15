# src/infrastructure/resilience/circuit_breaker.py
"""Circuit breaker pattern for protecting calls to external services."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class CircuitState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation — requests pass through
    OPEN = "open"  # Failure threshold reached — requests rejected
    HALF_OPEN = "half_open"  # Recovery — limited requests allowed


class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open and rejects a call."""


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_exceptions: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))


class CircuitBreaker:
    """Async circuit breaker for protecting external service calls.

    Usage as decorator:
        cb = CircuitBreaker()

        @cb
        async def call_external_service():
            ...

    Usage as context:
        cb = CircuitBreaker()
        result = await cb.call(some_async_func, arg1, arg2)
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        logger.info(
            "circuit_breaker_state_change",
            from_state=old_state.value,
            to_state=new_state.value,
            failure_count=self._failure_count,
        )

    async def _handle_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _handle_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._config.failure_threshold
            ):
                await self._transition_to(CircuitState.OPEN)

    async def _check_state(self) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._config.recovery_timeout:
                    await self._transition_to(CircuitState.HALF_OPEN)
                else:
                    remaining = self._config.recovery_timeout - elapsed
                    raise CircuitBreakerError(f"Circuit is OPEN. Retry after {remaining:.1f}s")

    async def call(self, func: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any) -> R:
        """Execute a function through the circuit breaker."""
        await self._check_state()
        try:
            result = await func(*args, **kwargs)
        except self._config.expected_exceptions:
            await self._handle_failure()
            raise
        else:
            await self._handle_success()
            return result

    def __call__(self, func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        """Use as a decorator on async functions."""

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await self.call(func, *args, **kwargs)

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
