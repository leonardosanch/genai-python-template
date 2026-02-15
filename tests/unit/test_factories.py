# tests/unit/test_factories.py
"""Tests for test factories."""

from tests.factories import DocumentFactory, LLMResponseFactory, VerificationResultFactory


class TestDocumentFactory:
    def test_create_returns_document(self) -> None:
        doc = DocumentFactory.create()
        assert doc.content
        assert doc.id is not None
        assert doc.created_at is not None

    def test_create_with_overrides(self) -> None:
        doc = DocumentFactory.create(content="custom", score=0.9)
        assert doc.content == "custom"
        assert doc.score == 0.9

    def test_create_batch(self) -> None:
        docs = DocumentFactory.create_batch(5)
        assert len(docs) == 5
        ids = [d.id for d in docs]
        assert len(set(ids)) == 5  # All unique IDs

    def test_create_batch_zero(self) -> None:
        docs = DocumentFactory.create_batch(0)
        assert docs == []


class TestLLMResponseFactory:
    def test_create_defaults(self) -> None:
        resp = LLMResponseFactory.create()
        assert resp.content == "Generated response text."
        assert resp.total_tokens == 150
        assert resp.cost_usd == 0.001

    def test_create_with_overrides(self) -> None:
        resp = LLMResponseFactory.create(content="custom", model="claude-3")
        assert resp.content == "custom"
        assert resp.model == "claude-3"


class TestVerificationResultFactory:
    def test_create_grounded(self) -> None:
        result = VerificationResultFactory.create()
        assert result.is_grounded is True
        assert result.faithfulness_score == 0.95

    def test_create_not_grounded(self) -> None:
        result = VerificationResultFactory.create(
            is_grounded=False,
            faithfulness_score=0.3,
            unsupported_claims=["Unsupported claim"],
        )
        assert result.is_grounded is False
        assert len(result.unsupported_claims) == 1
