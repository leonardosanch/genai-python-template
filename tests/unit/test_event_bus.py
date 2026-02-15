# tests/unit/test_event_bus.py

import pytest

from src.domain.events import DomainEvent
from src.infrastructure.events.in_memory_event_bus import InMemoryEventBus


class MockEvent(DomainEvent):
    """Test event payload."""

    payload: str


@pytest.mark.asyncio
async def test_publish_subscribe() -> None:
    """Test that subscribed handlers receive events."""
    bus = InMemoryEventBus()
    received_events: list[MockEvent] = []

    async def handler(event: DomainEvent) -> None:
        if isinstance(event, MockEvent):
            received_events.append(event)

    bus.subscribe(MockEvent, handler)

    event = MockEvent(payload="hello")
    await bus.publish(event)

    assert len(received_events) == 1
    assert received_events[0].payload == "hello"
    assert received_events[0].event_id is not None


@pytest.mark.asyncio
async def test_multiple_subscribers() -> None:
    """Test multiple handlers for same event."""
    bus = InMemoryEventBus()
    counter = {"count": 0}

    async def handler1(event: DomainEvent) -> None:
        counter["count"] += 1

    async def handler2(event: DomainEvent) -> None:
        counter["count"] += 2

    bus.subscribe(MockEvent, handler1)
    bus.subscribe(MockEvent, handler2)

    await bus.publish(MockEvent(payload="test"))

    assert counter["count"] == 3


@pytest.mark.asyncio
async def test_no_subscribers() -> None:
    """Test publishing with no subscribers works (no-op)."""
    bus = InMemoryEventBus()
    # Should not raise exception
    await bus.publish(MockEvent(payload="test"))
