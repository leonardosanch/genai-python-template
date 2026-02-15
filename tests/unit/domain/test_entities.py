"""Tests for domain entities — pure logic, no mocks, no I/O."""

from datetime import datetime

import pytest

from src.domain.entities.audit_record import AuditRecord
from src.domain.entities.dataset import Dataset
from src.domain.value_objects.dataset_metadata import DatasetMetadata


class TestAuditRecord:
    """Tests for the AuditRecord entity."""

    def test_create_with_required_fields(self) -> None:
        r = AuditRecord(action="create", actor="user1", resource="doc/1")
        assert r.action == "create"
        assert r.actor == "user1"
        assert r.resource == "doc/1"

    def test_auto_generated_id(self) -> None:
        r1 = AuditRecord(action="a", actor="b", resource="c")
        r2 = AuditRecord(action="a", actor="b", resource="c")
        assert r1.id != r2.id

    def test_auto_generated_timestamp(self) -> None:
        r = AuditRecord(action="a", actor="b", resource="c")
        assert isinstance(r.timestamp, datetime)

    def test_optional_fields(self) -> None:
        r = AuditRecord(
            action="delete",
            actor="admin",
            resource="user/5",
            details={"reason": "GDPR"},
            tenant_id="tenant-1",
        )
        assert r.details == {"reason": "GDPR"}
        assert r.tenant_id == "tenant-1"

    def test_frozen(self) -> None:
        r = AuditRecord(action="a", actor="b", resource="c")
        with pytest.raises(AttributeError):
            r.action = "modified"  # type: ignore[misc]

    def test_defaults_empty_details(self) -> None:
        r = AuditRecord(action="a", actor="b", resource="c")
        assert r.details == {}
        assert r.tenant_id is None


class TestDataset:
    """Tests for the Dataset entity."""

    def test_create_dataset(self) -> None:
        meta = DatasetMetadata(
            name="sales_meta",
            source="s3://bucket/data.csv",
            format="csv",
            record_count=1000,
        )
        ds = Dataset(id="ds-1", name="sales", metadata=meta)
        assert ds.id == "ds-1"
        assert ds.name == "sales"
        assert ds.metadata.source == "s3://bucket/data.csv"

    def test_optional_schema(self) -> None:
        meta = DatasetMetadata(name="empty_meta", source="local", format="json", record_count=0)
        ds = Dataset(id="1", name="empty", metadata=meta)
        assert ds.schema is None
        assert ds.created_at is None

    def test_frozen(self) -> None:
        meta = DatasetMetadata(name="m", source="x", format="csv", record_count=1)
        ds = Dataset(id="1", name="n", metadata=meta)
        with pytest.raises(AttributeError):
            ds.name = "changed"  # type: ignore[misc]
