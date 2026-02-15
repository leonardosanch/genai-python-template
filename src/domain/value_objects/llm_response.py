"""LLM Response value object — wraps raw LLM output with metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Immutable LLM response with usage metadata.

    Always track token usage and cost for observability
    and budget management.
    """

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None = None
