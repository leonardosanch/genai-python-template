"""Tests for domain value objects — pure logic, no mocks, no I/O."""

import pytest

from src.domain.value_objects.llm_response import LLMResponse
from src.domain.value_objects.pipeline_result import PipelineResult
from src.domain.value_objects.prompt import Prompt
from src.domain.value_objects.verification_result import VerificationResult


class TestLLMResponse:
    """Tests for the LLMResponse value object."""

    def test_create_with_all_fields(self) -> None:
        r = LLMResponse(
            content="hello",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
        )
        assert r.content == "hello"
        assert r.model == "gpt-4"
        assert r.total_tokens == 15
        assert r.cost_usd == 0.001

    def test_cost_defaults_to_none(self) -> None:
        r = LLMResponse(
            content="x",
            model="m",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
        assert r.cost_usd is None

    def test_frozen(self) -> None:
        r = LLMResponse(
            content="x",
            model="m",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
        with pytest.raises(AttributeError):
            r.content = "changed"  # type: ignore[misc]


class TestPipelineResult:
    """Tests for the PipelineResult value object."""

    def test_create_success(self) -> None:
        r = PipelineResult(
            status="success",
            records_processed=100,
            records_failed=0,
            duration_seconds=1.5,
            errors=[],
        )
        assert r.status == "success"
        assert r.records_processed == 100

    def test_create_with_errors(self) -> None:
        r = PipelineResult(
            status="partial",
            records_processed=80,
            records_failed=20,
            duration_seconds=2.0,
            errors=["row 5 invalid"],
        )
        assert r.records_failed == 20
        assert len(r.errors) == 1

    def test_negative_records_processed_raises(self) -> None:
        with pytest.raises(ValueError, match="records_processed must be non-negative"):
            PipelineResult(
                status="failed",
                records_processed=-1,
                records_failed=0,
                duration_seconds=0,
                errors=[],
            )

    def test_negative_records_failed_raises(self) -> None:
        with pytest.raises(ValueError, match="records_failed must be non-negative"):
            PipelineResult(
                status="failed",
                records_processed=0,
                records_failed=-1,
                duration_seconds=0,
                errors=[],
            )

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds must be non-negative"):
            PipelineResult(
                status="failed",
                records_processed=0,
                records_failed=0,
                duration_seconds=-1.0,
                errors=[],
            )

    def test_frozen(self) -> None:
        r = PipelineResult(
            status="success",
            records_processed=10,
            records_failed=0,
            duration_seconds=0.1,
            errors=[],
        )
        with pytest.raises(AttributeError):
            r.status = "failed"  # type: ignore[misc]


class TestPrompt:
    """Tests for the Prompt value object."""

    def test_render_simple(self) -> None:
        p = Prompt(template="Hello {name}", version="1.0")
        assert p.render(name="World") == "Hello World"

    def test_render_multiple_vars(self) -> None:
        p = Prompt(template="{greeting} {name}!", version="1.0")
        assert p.render(greeting="Hi", name="Bob") == "Hi Bob!"

    def test_render_missing_var_raises(self) -> None:
        p = Prompt(template="Hello {name}", version="1.0")
        with pytest.raises(KeyError):
            p.render()

    def test_version_tracked(self) -> None:
        p = Prompt(template="t", version="2.1.0")
        assert p.version == "2.1.0"

    def test_frozen(self) -> None:
        p = Prompt(template="t", version="1.0")
        with pytest.raises(AttributeError):
            p.template = "new"  # type: ignore[misc]


class TestVerificationResult:
    """Tests for the VerificationResult value object."""

    def test_grounded_result(self) -> None:
        v = VerificationResult(
            is_grounded=True,
            faithfulness_score=0.95,
            unsupported_claims=[],
            citations=["source1"],
        )
        assert v.is_grounded is True
        assert v.faithfulness_score == 0.95

    def test_hallucinated_result(self) -> None:
        v = VerificationResult(
            is_grounded=False,
            faithfulness_score=0.3,
            unsupported_claims=["claim1", "claim2"],
        )
        assert v.is_grounded is False
        assert len(v.unsupported_claims) == 2

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="faithfulness_score must be between"):
            VerificationResult(is_grounded=False, faithfulness_score=-0.1)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="faithfulness_score must be between"):
            VerificationResult(is_grounded=True, faithfulness_score=1.1)

    def test_boundary_scores_valid(self) -> None:
        v0 = VerificationResult(is_grounded=False, faithfulness_score=0.0)
        v1 = VerificationResult(is_grounded=True, faithfulness_score=1.0)
        assert v0.faithfulness_score == 0.0
        assert v1.faithfulness_score == 1.0

    def test_default_empty_lists(self) -> None:
        v = VerificationResult(is_grounded=True, faithfulness_score=0.8)
        assert v.unsupported_claims == []
        assert v.citations == []

    def test_frozen(self) -> None:
        v = VerificationResult(is_grounded=True, faithfulness_score=0.9)
        with pytest.raises(AttributeError):
            v.is_grounded = False  # type: ignore[misc]
