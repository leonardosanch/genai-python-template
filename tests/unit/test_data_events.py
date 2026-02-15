"""Unit tests for data engineering domain events."""

from src.domain.events import (
    DataIngestedEvent,
    DataTransformCompletedEvent,
    DataValidatedEvent,
    DomainEvent,
    SchemaEvolutionEvent,
)


class TestDataIngestedEvent:
    def test_inherits_domain_event(self) -> None:
        event = DataIngestedEvent(
            dataset_name="sales", record_count=100, source_uri="s3://bucket/sales.csv"
        )
        assert isinstance(event, DomainEvent)
        assert event.dataset_name == "sales"
        assert event.record_count == 100
        assert event.event_id  # auto-generated

    def test_frozen(self) -> None:
        event = DataIngestedEvent(dataset_name="x", record_count=0, source_uri="file.csv")
        try:
            event.dataset_name = "y"  # type: ignore[misc]
            assert False, "Should raise"
        except Exception:
            pass


class TestDataValidatedEvent:
    def test_fields(self) -> None:
        event = DataValidatedEvent(
            dataset_name="users",
            is_valid=False,
            valid_records=8,
            invalid_records=2,
        )
        assert event.is_valid is False
        assert event.valid_records == 8
        assert event.invalid_records == 2


class TestDataTransformCompletedEvent:
    def test_fields(self) -> None:
        event = DataTransformCompletedEvent(
            dataset_name="orders", input_records=100, output_records=95
        )
        assert event.input_records == 100
        assert event.output_records == 95


class TestSchemaEvolutionEvent:
    def test_fields(self) -> None:
        event = SchemaEvolutionEvent(
            dataset_name="products",
            old_version="1.0.0",
            new_version="2.0.0",
            changes=["added field 'color'"],
        )
        assert event.old_version == "1.0.0"
        assert event.new_version == "2.0.0"
        assert len(event.changes) == 1
