"""Data quality result value object — immutable validation outcome."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataQualityResult:
    """Immutable result of a data quality validation.

    Attributes:
        is_valid: Whether all records passed validation.
        total_records: Total number of records evaluated.
        valid_records: Number of records that passed validation.
        invalid_records: Number of records that failed validation.
        errors: List of validation error messages.
        metadata: Additional context about the validation run.
    """

    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.total_records < 0:
            raise ValueError("total_records must be non-negative")
        if self.valid_records < 0:
            raise ValueError("valid_records must be non-negative")
        if self.invalid_records < 0:
            raise ValueError("invalid_records must be non-negative")
        if self.valid_records + self.invalid_records != self.total_records:
            raise ValueError("valid_records + invalid_records must equal total_records")
