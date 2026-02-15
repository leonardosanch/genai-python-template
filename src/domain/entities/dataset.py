"""Dataset entity — represents a named dataset in the system."""

from dataclasses import dataclass
from datetime import datetime

from src.domain.value_objects.dataset_metadata import DatasetMetadata
from src.domain.value_objects.schema_definition import SchemaDefinition


@dataclass(frozen=True)
class Dataset:
    """Immutable dataset entity.

    Represents a logical dataset with metadata and optional schema.
    Frozen to ensure immutability — domain entities should not be
    mutated after creation.
    """

    id: str
    name: str
    metadata: DatasetMetadata
    schema: SchemaDefinition | None = None
    created_at: datetime | None = None
