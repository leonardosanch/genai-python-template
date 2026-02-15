# tests/unit/test_outbox_relay.py
"""Tests for outbox relay."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from src.infrastructure.events.outbox_relay import OutboxRelay


class _FakeSession:
    """Fake async session that supports async context manager."""

    def __init__(self, records: list[object]) -> None:
        self._records = records
        self.execute = AsyncMock()
        self.commit = AsyncMock()

        # Setup execute to return mock result with records
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = records
        self.execute.return_value = mock_result


def _make_session_factory(records: list[object]) -> object:
    """Create a callable that returns an async context manager yielding a FakeSession."""
    session = _FakeSession(records)

    @asynccontextmanager
    async def factory():  # type: ignore[no-untyped-def]
        yield session

    factory._session = session  # type: ignore[attr-defined]
    return factory


class TestOutboxRelay:
    async def test_relay_once_no_records(self) -> None:
        factory = _make_session_factory([])
        broker = AsyncMock()
        relay = OutboxRelay(
            session_factory=factory,  # type: ignore[arg-type]
            broker=broker,
            batch_size=10,
        )
        count = await relay.relay_once()
        assert count == 0
        broker.publish.assert_not_called()

    async def test_relay_once_publishes_records(self) -> None:
        record = MagicMock()
        record.id = "r1"
        record.aggregate_type = "document"
        record.event_type = "created"
        record.payload = {"doc_id": "123"}

        factory = _make_session_factory([record])
        broker = AsyncMock()

        relay = OutboxRelay(
            session_factory=factory,  # type: ignore[arg-type]
            broker=broker,
        )
        count = await relay.relay_once()
        assert count == 1
        broker.publish.assert_called_once_with(
            topic="document.created",
            message={"doc_id": "123"},
            key="r1",
        )

    async def test_relay_once_handles_publish_failure(self) -> None:
        record = MagicMock()
        record.id = "r1"
        record.aggregate_type = "document"
        record.event_type = "created"
        record.payload = {}

        factory = _make_session_factory([record])
        broker = AsyncMock()
        broker.publish.side_effect = RuntimeError("broker down")

        relay = OutboxRelay(
            session_factory=factory,  # type: ignore[arg-type]
            broker=broker,
        )
        count = await relay.relay_once()
        assert count == 0

    def test_stop(self) -> None:
        relay = OutboxRelay(
            session_factory=AsyncMock(),  # type: ignore[arg-type]
            broker=AsyncMock(),
        )
        relay._running = True
        relay.stop()
        assert relay._running is False
