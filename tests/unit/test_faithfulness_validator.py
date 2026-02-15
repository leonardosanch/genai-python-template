"""Unit tests for faithfulness validator and VerificationResult."""

import pytest

from src.application.guards.faithfulness_validator import faithfulness_validator
from src.application.guards.output_validator import OutputGuard
from src.domain.value_objects.verification_result import VerificationResult


class TestVerificationResult:
    def test_grounded_result(self) -> None:
        result = VerificationResult(
            is_grounded=True,
            faithfulness_score=0.95,
            unsupported_claims=[],
            citations=["source A"],
        )
        assert result.is_grounded is True
        assert result.faithfulness_score == 0.95

    def test_frozen_immutability(self) -> None:
        result = VerificationResult(is_grounded=True, faithfulness_score=0.8)
        with pytest.raises(AttributeError):
            result.is_grounded = False  # type: ignore[misc]

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            VerificationResult(is_grounded=False, faithfulness_score=-0.1)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            VerificationResult(is_grounded=True, faithfulness_score=1.1)

    def test_boundary_zero(self) -> None:
        result = VerificationResult(is_grounded=False, faithfulness_score=0.0)
        assert result.faithfulness_score == 0.0

    def test_boundary_one(self) -> None:
        result = VerificationResult(is_grounded=True, faithfulness_score=1.0)
        assert result.faithfulness_score == 1.0


class TestFaithfulnessValidator:
    def test_passes_above_threshold(self) -> None:
        verification = VerificationResult(is_grounded=True, faithfulness_score=0.85)
        validator = faithfulness_validator(verification, threshold=0.7)
        is_safe, text, violation = validator("some answer")
        assert is_safe is True
        assert violation is None

    def test_fails_below_threshold(self) -> None:
        verification = VerificationResult(
            is_grounded=False,
            faithfulness_score=0.5,
            unsupported_claims=["claim X"],
        )
        validator = faithfulness_validator(verification, threshold=0.7)
        is_safe, text, violation = validator("some answer")
        assert is_safe is False
        assert "0.50" in violation  # type: ignore[operator]
        assert "claim X" in violation  # type: ignore[operator]

    def test_exact_threshold_passes(self) -> None:
        verification = VerificationResult(is_grounded=True, faithfulness_score=0.7)
        validator = faithfulness_validator(verification, threshold=0.7)
        is_safe, _, _ = validator("answer")
        assert is_safe is True

    def test_just_below_threshold_fails(self) -> None:
        verification = VerificationResult(is_grounded=False, faithfulness_score=0.69)
        validator = faithfulness_validator(verification, threshold=0.7)
        is_safe, _, _ = validator("answer")
        assert is_safe is False

    def test_text_preserved_on_failure(self) -> None:
        verification = VerificationResult(is_grounded=False, faithfulness_score=0.3)
        validator = faithfulness_validator(verification, threshold=0.7)
        _, text, _ = validator("original text")
        assert text == "original text"


class TestFaithfulnessValidatorWithOutputGuard:
    def test_pluggable_into_output_guard(self) -> None:
        verification = VerificationResult(
            is_grounded=False,
            faithfulness_score=0.4,
            unsupported_claims=["invented fact"],
        )

        guard = OutputGuard()
        guard.add_validator(faithfulness_validator(verification, threshold=0.7))
        result = guard.validate("some answer with invented fact")

        assert result.is_safe is False
        assert len(result.violations) == 1
        assert "Faithfulness" in result.violations[0]

    def test_guard_passes_when_grounded(self) -> None:
        verification = VerificationResult(is_grounded=True, faithfulness_score=0.95)

        guard = OutputGuard()
        guard.add_validator(faithfulness_validator(verification, threshold=0.7))
        result = guard.validate("grounded answer")

        assert result.is_safe is True
        assert len(result.violations) == 0
