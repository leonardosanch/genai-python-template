"""OpenAI adapter — implements LLMPort for OpenAI models.

This is a reference implementation showing:
- Port implementation (Adapter pattern)
- Async client usage
- Structured output with Instructor
- Error handling with domain exceptions
- Observability (tracing attributes)
"""

from collections.abc import AsyncIterator
from typing import Any, TypeVar

import instructor
from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from src.domain.exceptions import LLMRateLimitError, LLMTimeoutError
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.config import get_settings

T = TypeVar("T", bound=BaseModel)


class OpenAIAdapter(LLMPort):
    """Concrete LLM adapter for OpenAI API.

    Uses instructor for structured output extraction.
    All errors are translated to domain exceptions.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        settings = get_settings()
        api_key = settings.llm.OPENAI_API_KEY

        raw_client = client or AsyncOpenAI(api_key=api_key)
        self._client = instructor.from_openai(raw_client)
        self._raw_client = raw_client
        self._model = model or settings.llm.OPENAI_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        try:
            response = await self._raw_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )
            return response.choices[0].message.content or ""
        except APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI request timed out: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}") from e

    async def generate_structured(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        try:
            return await self._client.chat.completions.create(
                model=self._model,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )
        except APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI request timed out: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}") from e

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        try:
            stream = await self._raw_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI stream timed out: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}") from e
