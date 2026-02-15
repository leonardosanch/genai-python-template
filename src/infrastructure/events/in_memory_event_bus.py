# src/infrastructure/events/in_memory_event_bus.py
import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

import structlog

from src.domain.events import DomainEvent
from src.domain.ports.event_bus_port import EventBusPort

logger = structlog.get_logger()


class InMemoryEventBus(EventBusPort):
    """In-memory implementation of EventBusPort."""

    def __init__(self) -> None:
        self._subscribers: defaultdict[
            type[DomainEvent], list[Callable[[DomainEvent], Awaitable[None]]]
        ] = defaultdict(list)

    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        event_type = type(event)
        handlers = self._subscribers[event_type]

        logger.debug(
            "publishing_event",
            event_type=event_type.__name__,
            event_id=event.event_id,
            handler_count=len(handlers),
        )

        if not handlers:
            return

        # Execute all handlers concurrently
        await asyncio.gather(*[handler(event) for handler in handlers])

    def subscribe(
        self, event_type: type[DomainEvent], handler: Callable[[DomainEvent], Awaitable[None]]
    ) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(
            "subscriber_registered", event_type=event_type.__name__, handler=handler.__name__
        )
