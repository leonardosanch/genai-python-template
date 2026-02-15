"""Tests for InMemoryEventBus — publish/subscribe event system."""

from unittest.mock import AsyncMock

import pytest

from src.domain.events import DataIngestedEvent, DataValidatedEvent, DomainEvent
from src.infrastructure.events.in_memory_event_bus import InMemoryEventBus


@pytest.fixture()
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


class TestInMemoryEventBus:
    """Tests for the in-memory event bus."""

    @pytest.mark.asyncio()
    async def test_publish_without_subscribers(self, bus: InMemoryEventBus) -> None:
        event = DataIngestedEvent(
            dataset_name="test",
            record_count=10,
            source_uri="s3://x",
        )
        # Should not raise
        await bus.publish(event)

    @pytest.mark.asyncio()
    async def test_subscriber_receives_event(self, bus: InMemoryEventBus) -> None:
        handler = AsyncMock()
        bus.subscribe(DataIngestedEvent, handler)

        event = DataIngestedEvent(
            dataset_name="sales",
            record_count=100,
            source_uri="s3://data",
        )
        await bus.publish(event)

        handler.assert_called_once_with(event)

    @pytest.mark.asyncio()
    async def test_multiple_subscribers(self, bus: InMemoryEventBus) -> None:
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        bus.subscribe(DataIngestedEvent, handler1)
        bus.subscribe(DataIngestedEvent, handler2)

        event = DataIngestedEvent(
            dataset_name="x",
            record_count=1,
            source_uri="y",
        )
        await bus.publish(event)

        handler1.assert_called_once()
        handler2.assert_called_once()

    @pytest.mark.asyncio()
    async def test_subscriber_only_receives_correct_event_type(self, bus: InMemoryEventBus) -> None:
        ingested_handler = AsyncMock()
        validated_handler = AsyncMock()
        bus.subscribe(DataIngestedEvent, ingested_handler)
        bus.subscribe(DataValidatedEvent, validated_handler)

        event = DataIngestedEvent(
            dataset_name="x",
            record_count=1,
            source_uri="y",
        )
        await bus.publish(event)

        ingested_handler.assert_called_once()
        validated_handler.assert_not_called()

    @pytest.mark.asyncio()
    async def test_concurrent_handler_execution(self, bus: InMemoryEventBus) -> None:
        """Handlers execute concurrently via asyncio.gather."""
        call_order: list[str] = []

        async def handler_a(event: DomainEvent) -> None:
            call_order.append("a")

        async def handler_b(event: DomainEvent) -> None:
            call_order.append("b")

        bus.subscribe(DataIngestedEvent, handler_a)
        bus.subscribe(DataIngestedEvent, handler_b)

        event = DataIngestedEvent(
            dataset_name="x",
            record_count=1,
            source_uri="y",
        )
        await bus.publish(event)

        assert set(call_order) == {"a", "b"}
