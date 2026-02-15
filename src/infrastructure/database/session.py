"""SQLAlchemy async session factory.

Reference implementation for database connection management.
Uses asyncpg driver for PostgreSQL.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.infrastructure.config import get_settings


def create_engine() -> "AsyncEngine":
    """Create async SQLAlchemy engine with connection pooling."""
    settings = get_settings()
    return create_async_engine(
        settings.database.URL,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        echo=settings.database.ECHO,
    )


def create_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create session factory for dependency injection."""
    engine = create_engine()
    return async_sessionmaker(engine, expire_on_commit=False)
