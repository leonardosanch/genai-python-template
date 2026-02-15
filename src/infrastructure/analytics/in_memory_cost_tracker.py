# src/infrastructure/analytics/in_memory_cost_tracker.py
"""In-memory cost tracker implementation."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from src.domain.ports.cost_tracker_port import CostTrackerPort, LLMUsageRecord


class InMemoryCostTracker(CostTrackerPort):
    """In-memory LLM cost tracker for development and testing.

    For production, use a database-backed implementation.
    """

    def __init__(self) -> None:
        self._records: list[LLMUsageRecord] = []

    async def record(self, usage: LLMUsageRecord) -> None:
        self._records.append(usage)

    def _filter_by_days(self, days: int) -> list[LLMUsageRecord]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        return [r for r in self._records if r.timestamp >= cutoff]

    async def get_summary(
        self,
        days: int = 30,
        group_by: str = "model",
    ) -> list[dict[str, Any]]:
        records = self._filter_by_days(days)
        groups: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"total_cost": 0.0, "total_tokens": 0, "call_count": 0}
        )

        for r in records:
            if group_by == "model":
                key = r.model
            elif group_by == "use_case":
                key = r.use_case
            elif group_by == "day":
                key = r.timestamp.strftime("%Y-%m-%d")
            else:
                key = r.model

            groups[key]["total_cost"] += r.cost_usd
            groups[key]["total_tokens"] += r.total_tokens
            groups[key]["call_count"] += 1

        return [
            {
                "group": key,
                "total_cost": round(float(v["total_cost"]), 6),
                "total_tokens": int(v["total_tokens"]),
                "call_count": int(v["call_count"]),
            }
            for key, v in sorted(groups.items())
        ]

    async def get_total_cost(self, days: int = 30) -> float:
        records = self._filter_by_days(days)
        return round(sum(r.cost_usd for r in records), 6)
