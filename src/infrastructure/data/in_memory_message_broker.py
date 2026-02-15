"""In-memory message broker — for testing and development only."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from src.domain.ports.message_broker_port import MessageBrokerPort


class InMemoryMessageBroker(MessageBrokerPort):
    """MessageBrokerPort implementation using asyncio.Queue.

    For testing and local development only — not for production.
    """

    def __init__(self) -> None:
        self._topics: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)
        self._closed = False

    async def publish(self, topic: str, message: dict[str, Any], key: str | None = None) -> None:
        """Publish a message to an in-memory topic queue."""
        if self._closed:
            raise RuntimeError("Broker is closed")
        await self._topics[topic].put(message)

    async def subscribe(self, topic: str, group_id: str) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to a topic and yield messages from the queue."""
        queue = self._topics[topic]
        while not self._closed:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield message
            except TimeoutError:
                continue

    async def close(self) -> None:
        """Mark the broker as closed."""
        self._closed = True
