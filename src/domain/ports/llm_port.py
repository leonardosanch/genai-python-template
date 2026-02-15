"""Port for LLM providers. Infrastructure adapters implement this interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMPort(ABC):
    """Abstract interface for LLM interactions.

    All LLM providers (OpenAI, Anthropic, Gemini, local models)
    must implement this port. Domain and application layers
    depend only on this interface, never on concrete providers.
    """

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response from the LLM."""
        ...

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        """Generate a structured response validated against a Pydantic schema."""
        ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens from the LLM as an async iterator."""
        ...
