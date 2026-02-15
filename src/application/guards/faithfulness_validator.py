"""Faithfulness validator — pluggable into OutputGuard chain."""

from src.application.guards.output_validator import ValidatorFunc
from src.domain.value_objects.verification_result import VerificationResult


def faithfulness_validator(
    verification: VerificationResult, threshold: float = 0.7
) -> ValidatorFunc:
    """Factory for a faithfulness validator based on a verification result.

    Follows the same pattern as max_length_validator() — returns a ValidatorFunc
    that can be plugged into OutputGuard.add_validator().

    Args:
        verification: Pre-computed verification result from HallucinationCheckerPort.
        threshold: Minimum faithfulness score to pass.
    """

    def validator(text: str) -> tuple[bool, str, str | None]:
        if verification.faithfulness_score < threshold:
            violation = (
                f"Faithfulness score {verification.faithfulness_score:.2f} "
                f"below threshold {threshold:.2f}. "
                f"Unsupported claims: {verification.unsupported_claims}"
            )
            return False, text, violation
        return True, text, None

    return validator
