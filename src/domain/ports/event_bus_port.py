# src/domain/ports/event_bus_port.py
from collections.abc import Awaitable, Callable
from typing import Protocol

from src.domain.events import DomainEvent


class EventBusPort(Protocol):
    """Port for event bus implementation."""

    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        ...

    def subscribe(
        self, event_type: type[DomainEvent], handler: Callable[[DomainEvent], Awaitable[None]]
    ) -> None:
        """Subscribe a handler to an event type."""
        ...
