"""Tests for domain events — immutability, auto-fields, schema correctness."""

from datetime import datetime

from src.domain.events import (
    DataIngestedEvent,
    DataTransformCompletedEvent,
    DataValidatedEvent,
    DomainEvent,
    SchemaEvolutionEvent,
)


class TestDomainEvent:
    """Tests for the base DomainEvent."""

    def test_auto_generated_id(self) -> None:
        e1 = DomainEvent()
        e2 = DomainEvent()
        assert e1.event_id != e2.event_id

    def test_auto_generated_timestamp(self) -> None:
        e = DomainEvent()
        assert isinstance(e.timestamp, datetime)

    def test_frozen(self) -> None:
        e = DomainEvent()
        try:
            e.event_id = "changed"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # Pydantic frozen model raises ValidationError


class TestDataIngestedEvent:
    """Tests for DataIngestedEvent."""

    def test_create(self) -> None:
        e = DataIngestedEvent(
            dataset_name="sales",
            record_count=100,
            source_uri="s3://data",
        )
        assert e.dataset_name == "sales"
        assert e.record_count == 100
        assert e.source_uri == "s3://data"
        assert e.event_id  # auto-generated


class TestDataValidatedEvent:
    """Tests for DataValidatedEvent."""

    def test_create(self) -> None:
        e = DataValidatedEvent(
            dataset_name="users",
            is_valid=True,
            valid_records=95,
            invalid_records=5,
        )
        assert e.is_valid is True
        assert e.valid_records == 95
        assert e.invalid_records == 5


class TestDataTransformCompletedEvent:
    """Tests for DataTransformCompletedEvent."""

    def test_create(self) -> None:
        e = DataTransformCompletedEvent(
            dataset_name="logs",
            input_records=1000,
            output_records=950,
        )
        assert e.input_records == 1000
        assert e.output_records == 950


class TestSchemaEvolutionEvent:
    """Tests for SchemaEvolutionEvent."""

    def test_create(self) -> None:
        e = SchemaEvolutionEvent(
            dataset_name="orders",
            old_version="1.0",
            new_version="2.0",
            changes=["added column: discount"],
        )
        assert e.old_version == "1.0"
        assert e.new_version == "2.0"
        assert len(e.changes) == 1
