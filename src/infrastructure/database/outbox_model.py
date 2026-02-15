# src/infrastructure/database/outbox_model.py
"""Outbox pattern model for reliable event publishing."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database.models import Base


class OutboxRecord(Base):
    """SQLAlchemy model for the outbox table.

    Stores events that need to be published to the message broker.
    A relay process polls unpublished records and publishes them.
    """

    __tablename__ = "outbox"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    aggregate_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)  # type: ignore[type-arg]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None, index=True
    )
