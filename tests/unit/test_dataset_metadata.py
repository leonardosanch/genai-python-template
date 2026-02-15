"""Unit tests for DatasetMetadata value object."""

from dataclasses import FrozenInstanceError

import pytest

from src.domain.value_objects.dataset_metadata import DatasetMetadata


class TestDatasetMetadataCreation:
    def test_minimal(self) -> None:
        meta = DatasetMetadata(name="test", record_count=100)
        assert meta.name == "test"
        assert meta.record_count == 100
        assert meta.schema_fields == ()
        assert meta.source == ""
        assert meta.size_bytes is None
        assert meta.format == ""

    def test_full(self) -> None:
        meta = DatasetMetadata(
            name="sales",
            record_count=1000,
            schema_fields=("id", "amount", "date"),
            source="s3://bucket/sales.csv",
            size_bytes=4096,
            format="csv",
        )
        assert meta.schema_fields == ("id", "amount", "date")
        assert meta.source == "s3://bucket/sales.csv"
        assert meta.size_bytes == 4096
        assert meta.format == "csv"

    def test_created_at_auto_set(self) -> None:
        meta = DatasetMetadata(name="test", record_count=0)
        assert meta.created_at is not None


class TestDatasetMetadataImmutability:
    def test_frozen(self) -> None:
        meta = DatasetMetadata(name="test", record_count=0)
        with pytest.raises(FrozenInstanceError):
            meta.name = "changed"  # type: ignore[misc]
