"""Verification result value object — immutable hallucination check outcome."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VerificationResult:
    """Immutable result of a hallucination verification check.

    Attributes:
        is_grounded: Whether the answer is fully supported by the context.
        faithfulness_score: Score between 0.0 (hallucinated) and 1.0 (grounded).
        unsupported_claims: Claims in the answer not supported by context.
        citations: Context passages that support the answer.
    """

    is_grounded: bool
    faithfulness_score: float
    unsupported_claims: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate invariants."""
        if not 0.0 <= self.faithfulness_score <= 1.0:
            raise ValueError(
                f"faithfulness_score must be between 0.0 and 1.0, got {self.faithfulness_score}"
            )
