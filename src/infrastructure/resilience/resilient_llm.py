"""Resilient LLM adapter — Decorator wrapping any LLMPort with circuit breaker."""

from collections.abc import AsyncIterator
from typing import Any, TypeVar

from pydantic import BaseModel

from src.domain.exceptions import LLMError
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)

T = TypeVar("T", bound=BaseModel)

# Default: trip after 5 LLM failures, recover after 30s
DEFAULT_LLM_CB_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exceptions=(LLMError,),
)


class ResilientLLMPort(LLMPort):
    """Decorator that wraps an LLMPort with circuit breaker protection.

    All three LLM operations (generate, generate_structured, stream) are
    protected. When the failure threshold is reached the circuit opens and
    subsequent calls fail fast with ``CircuitBreakerError`` instead of
    waiting for the downstream provider to time out.
    """

    def __init__(
        self,
        inner: LLMPort,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self._inner = inner
        self._cb = CircuitBreaker(config or DEFAULT_LLM_CB_CONFIG)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Expose CB for monitoring / health checks."""
        return self._cb

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._cb.call(self._inner.generate, prompt, **kwargs)

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        return await self._cb.call(
            self._inner.generate_structured,
            prompt,
            schema,
            **kwargs,
        )

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens with circuit breaker protection.

        The CB state is checked before the stream begins.  Failures during
        iteration count toward the failure threshold.  A full successful
        iteration resets the failure counter.
        """
        # Fail fast if circuit is open
        await self._cb._check_state()  # noqa: SLF001
        try:
            async for token in self._inner.stream(prompt, **kwargs):
                yield token
        except tuple(self._cb._config.expected_exceptions):  # noqa: SLF001
            await self._cb._handle_failure()  # noqa: SLF001
            raise
        else:
            await self._cb._handle_success()  # noqa: SLF001
