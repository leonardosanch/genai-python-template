"""LiteLLM adapter — implements LLMPort for 100+ LLM providers via LiteLLM.

Uses litellm as a unified API gateway. Supports OpenAI, Anthropic, Cohere,
Gemini, local models, and more through a single interface.
"""

from collections.abc import AsyncIterator
from typing import Any, TypeVar

from pydantic import BaseModel

from src.domain.exceptions import LLMRateLimitError, LLMTimeoutError
from src.domain.ports.llm_port import LLMPort

T = TypeVar("T", bound=BaseModel)


class LiteLLMAdapter(LLMPort):
    """Concrete LLM adapter using LiteLLM unified API.

    All errors are translated to domain exceptions.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        import litellm
        from litellm.exceptions import RateLimitError, Timeout

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._api_key,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )
            return str(response.choices[0].message.content or "")
        except Timeout as e:
            raise LLMTimeoutError(f"LiteLLM request timed out: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"LiteLLM rate limit exceeded: {e}") from e

    async def generate_structured(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        import instructor
        import litellm
        from litellm.exceptions import RateLimitError, Timeout

        client = instructor.from_litellm(litellm.acompletion)
        try:
            return await client(  # type: ignore[no-any-return]
                model=self._model,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._api_key,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )
        except Timeout as e:
            raise LLMTimeoutError(f"LiteLLM request timed out: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"LiteLLM rate limit exceeded: {e}") from e

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        import litellm
        from litellm.exceptions import RateLimitError, Timeout

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._api_key,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Timeout as e:
            raise LLMTimeoutError(f"LiteLLM stream timed out: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"LiteLLM rate limit exceeded: {e}") from e
