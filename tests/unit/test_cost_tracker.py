# tests/unit/test_cost_tracker.py
"""Tests for in-memory cost tracker."""

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.ports.cost_tracker_port import LLMUsageRecord
from src.infrastructure.analytics.in_memory_cost_tracker import InMemoryCostTracker


@pytest.fixture
def tracker() -> InMemoryCostTracker:
    return InMemoryCostTracker()


def _make_usage(
    model: str = "gpt-4o",
    cost_usd: float = 0.01,
    use_case: str = "chat",
    total_tokens: int = 500,
    timestamp: datetime | None = None,
) -> LLMUsageRecord:
    return LLMUsageRecord(
        model=model,
        prompt_tokens=200,
        completion_tokens=300,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        use_case=use_case,
        timestamp=timestamp or datetime.now(UTC),
    )


class TestInMemoryCostTracker:
    async def test_record_and_total(self, tracker: InMemoryCostTracker) -> None:
        await tracker.record(_make_usage(cost_usd=0.01))
        await tracker.record(_make_usage(cost_usd=0.02))
        total = await tracker.get_total_cost(days=30)
        assert total == pytest.approx(0.03, abs=1e-6)

    async def test_total_excludes_old_records(self, tracker: InMemoryCostTracker) -> None:
        old = _make_usage(
            cost_usd=0.5,
            timestamp=datetime.now(UTC) - timedelta(days=60),
        )
        recent = _make_usage(cost_usd=0.01)
        await tracker.record(old)
        await tracker.record(recent)
        total = await tracker.get_total_cost(days=30)
        assert total == pytest.approx(0.01, abs=1e-6)

    async def test_summary_by_model(self, tracker: InMemoryCostTracker) -> None:
        await tracker.record(_make_usage(model="gpt-4o", cost_usd=0.01))
        await tracker.record(_make_usage(model="gpt-4o", cost_usd=0.02))
        await tracker.record(_make_usage(model="claude-3", cost_usd=0.05))
        summary = await tracker.get_summary(days=30, group_by="model")
        assert len(summary) == 2
        gpt = next(s for s in summary if s["group"] == "gpt-4o")
        assert gpt["call_count"] == 2
        assert gpt["total_cost"] == pytest.approx(0.03, abs=1e-6)

    async def test_summary_by_use_case(self, tracker: InMemoryCostTracker) -> None:
        await tracker.record(_make_usage(use_case="chat"))
        await tracker.record(_make_usage(use_case="rag"))
        summary = await tracker.get_summary(group_by="use_case")
        use_cases = {s["group"] for s in summary}
        assert use_cases == {"chat", "rag"}

    async def test_summary_by_day(self, tracker: InMemoryCostTracker) -> None:
        await tracker.record(_make_usage())
        summary = await tracker.get_summary(group_by="day")
        assert len(summary) == 1
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert summary[0]["group"] == today

    async def test_empty_tracker(self, tracker: InMemoryCostTracker) -> None:
        total = await tracker.get_total_cost()
        assert total == 0.0
        summary = await tracker.get_summary()
        assert summary == []
