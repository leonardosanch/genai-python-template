"""Dataset metadata value object — immutable descriptive information."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class DatasetMetadata:
    """Immutable metadata describing a dataset.

    Attributes:
        name: Human-readable dataset name.
        record_count: Number of records in the dataset.
        schema_fields: Tuple of field names in the dataset.
        source: Origin of the dataset (URI, path, etc.).
        created_at: When the metadata was captured.
        size_bytes: Size of the dataset in bytes (if known).
        format: Data format (csv, json, parquet, etc.).
    """

    name: str
    record_count: int
    schema_fields: tuple[str, ...] = field(default_factory=tuple)
    source: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    size_bytes: int | None = None
    format: str = ""
