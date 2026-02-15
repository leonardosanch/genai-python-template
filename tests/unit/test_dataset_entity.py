"""Unit tests for Dataset entity."""

from dataclasses import FrozenInstanceError

import pytest

from src.domain.entities.dataset import Dataset
from src.domain.value_objects.dataset_metadata import DatasetMetadata
from src.domain.value_objects.schema_definition import (
    FieldDefinition,
    SchemaDefinition,
)


class TestDatasetCreation:
    def test_minimal(self) -> None:
        meta = DatasetMetadata(name="test", record_count=10)
        ds = Dataset(id="ds-1", name="test", metadata=meta)
        assert ds.id == "ds-1"
        assert ds.name == "test"
        assert ds.metadata.record_count == 10
        assert ds.schema is None
        assert ds.created_at is None

    def test_with_schema(self) -> None:
        meta = DatasetMetadata(name="users", record_count=100)
        schema = SchemaDefinition(
            name="users",
            version="1.0.0",
            fields=(FieldDefinition(name="id", data_type="int"),),
        )
        ds = Dataset(id="ds-2", name="users", metadata=meta, schema=schema)
        assert ds.schema is not None
        assert ds.schema.name == "users"


class TestDatasetImmutability:
    def test_frozen(self) -> None:
        meta = DatasetMetadata(name="test", record_count=0)
        ds = Dataset(id="ds-1", name="test", metadata=meta)
        with pytest.raises(FrozenInstanceError):
            ds.name = "changed"  # type: ignore[misc]
