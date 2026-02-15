# tests/unit/test_context_assembler.py
"""Tests for context assembler service."""

import pytest

from src.application.services.context_assembler import ContextAssembler
from src.domain.entities.document import Document
from src.domain.value_objects.context_budget import ContextBudget


@pytest.fixture
def assembler() -> ContextAssembler:
    return ContextAssembler()


def _make_docs(count: int, content_size: int = 100) -> list[Document]:
    return [Document(content="x" * content_size, id=f"d{i}", metadata={}) for i in range(count)]


class TestContextAssembler:
    async def test_basic_assembly(self, assembler: ContextAssembler) -> None:
        docs = _make_docs(2, content_size=50)
        budget = ContextBudget(max_tokens=500)
        result, updated_budget = await assembler.assemble("What?", docs, budget)
        assert "What?" in result
        assert "Context:" in result
        assert updated_budget.used_tokens > 0

    async def test_with_system_prompt(self, assembler: ContextAssembler) -> None:
        docs = _make_docs(1, content_size=50)
        budget = ContextBudget(max_tokens=500)
        result, _ = await assembler.assemble(
            "What?", docs, budget, system_prompt="You are helpful."
        )
        assert "You are helpful." in result

    async def test_budget_limits_documents(self, assembler: ContextAssembler) -> None:
        # Each doc ~1000 chars = ~250 tokens; budget only allows 1-2
        docs = _make_docs(5, content_size=1000)
        budget = ContextBudget(max_tokens=300)
        result, updated_budget = await assembler.assemble("What?", docs, budget)
        # Not all 5 docs should be included
        doc_count = result.count("---")
        assert doc_count < 5

    async def test_empty_documents(self, assembler: ContextAssembler) -> None:
        budget = ContextBudget(max_tokens=500)
        result, _ = await assembler.assemble("What?", [], budget)
        assert "What?" in result
        assert "Context:" in result

    async def test_budget_consumed(self, assembler: ContextAssembler) -> None:
        docs = _make_docs(3, content_size=100)
        budget = ContextBudget(max_tokens=1000)
        _, updated_budget = await assembler.assemble("Query", docs, budget)
        assert updated_budget.used_tokens > 0
        assert updated_budget.remaining < budget.max_tokens
