# src/domain/value_objects/context_budget.py
"""Context budget value object for managing LLM context window usage."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    """Immutable budget for tracking context window token usage.

    Used by context assemblers to decide how much content to include.
    """

    max_tokens: int
    used_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.used_tokens < 0:
            raise ValueError(f"used_tokens must be non-negative, got {self.used_tokens}")
        if self.used_tokens > self.max_tokens:
            raise ValueError(
                f"used_tokens ({self.used_tokens}) cannot exceed max_tokens ({self.max_tokens})"
            )

    @property
    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens

    @property
    def utilization(self) -> float:
        """Return utilization as a fraction between 0.0 and 1.0."""
        return self.used_tokens / self.max_tokens

    def consume(self, tokens: int) -> "ContextBudget":
        """Return a new budget with tokens consumed.

        Raises:
            ValueError: If consuming would exceed the budget.
        """
        new_used = self.used_tokens + tokens
        if new_used > self.max_tokens:
            raise ValueError(f"Cannot consume {tokens} tokens: only {self.remaining} remaining")
        return ContextBudget(max_tokens=self.max_tokens, used_tokens=new_used)
