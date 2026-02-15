# tests/unit/test_context_budget.py
"""Tests for context budget value object."""

import pytest

from src.domain.value_objects.context_budget import ContextBudget


class TestContextBudget:
    def test_create_valid(self) -> None:
        budget = ContextBudget(max_tokens=4096)
        assert budget.max_tokens == 4096
        assert budget.used_tokens == 0
        assert budget.remaining == 4096

    def test_utilization(self) -> None:
        budget = ContextBudget(max_tokens=1000, used_tokens=250)
        assert budget.utilization == pytest.approx(0.25)

    def test_consume(self) -> None:
        budget = ContextBudget(max_tokens=1000)
        new_budget = budget.consume(300)
        assert new_budget.used_tokens == 300
        assert new_budget.remaining == 700
        # Original is unchanged (immutable)
        assert budget.used_tokens == 0

    def test_consume_chain(self) -> None:
        budget = ContextBudget(max_tokens=1000)
        budget = budget.consume(200).consume(300)
        assert budget.used_tokens == 500

    def test_consume_exceeds_raises(self) -> None:
        budget = ContextBudget(max_tokens=100, used_tokens=90)
        with pytest.raises(ValueError, match="Cannot consume"):
            budget.consume(20)

    def test_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ContextBudget(max_tokens=0)

    def test_negative_used_tokens(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ContextBudget(max_tokens=100, used_tokens=-1)

    def test_used_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ContextBudget(max_tokens=100, used_tokens=200)

    def test_frozen(self) -> None:
        budget = ContextBudget(max_tokens=1000)
        with pytest.raises(AttributeError):
            budget.max_tokens = 2000  # type: ignore[misc]
