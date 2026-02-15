"""Pipeline result value object — immutable execution result."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineResult:
    """Immutable result of a pipeline execution.

    Attributes:
        status: Execution status (success, failed, or partial)
        records_processed: Number of records successfully processed
        records_failed: Number of records that failed processing
        duration_seconds: Total execution time in seconds
        errors: List of error messages encountered during execution
    """

    status: Literal["success", "failed", "partial"]
    records_processed: int
    records_failed: int
    duration_seconds: float
    errors: list[str]

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.records_processed < 0:
            raise ValueError("records_processed must be non-negative")
        if self.records_failed < 0:
            raise ValueError("records_failed must be non-negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
