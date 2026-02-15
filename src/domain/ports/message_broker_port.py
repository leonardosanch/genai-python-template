"""Port for message brokers. Infrastructure adapters implement this interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class MessageBrokerPort(ABC):
    """Abstract interface for message broker interactions.

    Implementations handle Kafka, RabbitMQ, Redis Streams, etc.
    Separate from EventBusPort — this is for external streaming,
    not internal domain event dispatch.
    """

    @abstractmethod
    async def publish(self, topic: str, message: dict[str, Any], key: str | None = None) -> None:
        """Publish a message to a topic.

        Args:
            topic: Target topic/queue name.
            message: Message payload.
            key: Optional partition key.
        """
        ...

    @abstractmethod
    def subscribe(self, topic: str, group_id: str) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to a topic and yield messages.

        Args:
            topic: Topic/queue to subscribe to.
            group_id: Consumer group identifier.

        Returns:
            Async iterator of message payloads.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the broker connection."""
        ...
