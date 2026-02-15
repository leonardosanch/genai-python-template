# src/domain/ports/cost_tracker_port.py
"""Port for LLM cost tracking and analytics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class LLMUsageRecord:
    """Immutable record of a single LLM call's usage and cost."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    use_case: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class CostTrackerPort(ABC):
    """Abstract interface for LLM cost tracking."""

    @abstractmethod
    async def record(self, usage: LLMUsageRecord) -> None:
        """Record an LLM usage entry."""
        ...

    @abstractmethod
    async def get_summary(
        self,
        days: int = 30,
        group_by: str = "model",
    ) -> list[dict[str, Any]]:
        """Get usage summary grouped by a field.

        Args:
            days: Number of days to look back.
            group_by: Field to group by (model, use_case, day).
        """
        ...

    @abstractmethod
    async def get_total_cost(self, days: int = 30) -> float:
        """Get total cost in USD for the period."""
        ...
