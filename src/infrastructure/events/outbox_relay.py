# src/infrastructure/events/outbox_relay.py
"""Outbox relay — polls unpublished events and publishes to broker."""

import asyncio
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.domain.ports.message_broker_port import MessageBrokerPort
from src.infrastructure.database.outbox_model import OutboxRecord

logger = structlog.get_logger(__name__)


class OutboxRelay:
    """Polls the outbox table for unpublished events and publishes them.

    Runs as a background process. Each cycle:
    1. SELECT unpublished records (published_at IS NULL) with batch limit
    2. Publish each to the broker
    3. UPDATE published_at for successfully published records
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        broker: MessageBrokerPort,
        batch_size: int = 100,
        poll_interval: float = 1.0,
    ) -> None:
        self._session_factory = session_factory
        self._broker = broker
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._running = False

    async def relay_once(self) -> int:
        """Process one batch of unpublished events. Returns count published."""
        published_count = 0
        async with self._session_factory() as session:
            stmt = (
                select(OutboxRecord)
                .where(OutboxRecord.published_at.is_(None))
                .order_by(OutboxRecord.created_at)
                .limit(self._batch_size)
            )
            result = await session.execute(stmt)
            records = list(result.scalars().all())

            if not records:
                return 0

            published_ids: list[str] = []
            for record in records:
                try:
                    topic = f"{record.aggregate_type}.{record.event_type}"
                    await self._broker.publish(
                        topic=topic,
                        message=record.payload,
                        key=record.id,
                    )
                    published_ids.append(record.id)
                    published_count += 1
                except Exception:
                    logger.error(
                        "outbox_publish_failed",
                        record_id=record.id,
                        event_type=record.event_type,
                    )

            if published_ids:
                update_stmt = (
                    update(OutboxRecord)
                    .where(OutboxRecord.id.in_(published_ids))
                    .values(published_at=datetime.now(UTC))
                )
                await session.execute(update_stmt)
                await session.commit()

            logger.info(
                "outbox_relay_batch",
                total=len(records),
                published=published_count,
            )

        return published_count

    async def run(self) -> None:
        """Run the relay loop continuously."""
        self._running = True
        logger.info("outbox_relay_started", poll_interval=self._poll_interval)
        while self._running:
            try:
                await self.relay_once()
            except Exception:
                logger.error("outbox_relay_error")
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """Signal the relay to stop."""
        self._running = False
        logger.info("outbox_relay_stopped")
