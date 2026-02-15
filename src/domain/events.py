from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DomainEvent(BaseModel):
    """Base class for all domain events."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(frozen=True)


class DataIngestedEvent(DomainEvent):
    """Emitted when data has been successfully ingested from a source."""

    dataset_name: str
    record_count: int
    source_uri: str


class DataValidatedEvent(DomainEvent):
    """Emitted when a dataset has been validated."""

    dataset_name: str
    is_valid: bool
    valid_records: int
    invalid_records: int


class DataTransformCompletedEvent(DomainEvent):
    """Emitted when a data transformation completes."""

    dataset_name: str
    input_records: int
    output_records: int


class SchemaEvolutionEvent(DomainEvent):
    """Emitted when a schema change is detected."""

    dataset_name: str
    old_version: str
    new_version: str
    changes: list[str]
