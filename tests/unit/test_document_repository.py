# tests/unit/test_document_repository.py
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.entities.document import Document
from src.infrastructure.database.document_repository import SQLAlchemyDocumentRepository
from src.infrastructure.database.models import Base


@pytest.fixture
async def sqlite_session_factory():
    """Create an in-memory SQLite session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_get(sqlite_session_factory):
    """Test saving and retrieving a document."""
    repo = SQLAlchemyDocumentRepository(sqlite_session_factory)
    doc = Document(content="Test content", metadata={"key": "value"})

    saved_id = await repo.save(doc)
    assert saved_id is not None

    retrieved = await repo.get_by_id(saved_id)
    assert retrieved is not None
    assert retrieved.content == "Test content"
    assert retrieved.metadata == {"key": "value"}
    assert retrieved.id == saved_id


@pytest.mark.asyncio
async def test_list(sqlite_session_factory):
    """Test listing documents."""
    repo = SQLAlchemyDocumentRepository(sqlite_session_factory)
    await repo.save(Document(content="Doc 1"))
    await repo.save(Document(content="Doc 2"))

    docs = await repo.list()
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_delete(sqlite_session_factory):
    """Test deleting a document."""
    repo = SQLAlchemyDocumentRepository(sqlite_session_factory)
    doc = Document(content="To delete")
    saved_id = await repo.save(doc)

    await repo.delete(saved_id)
    retrieved = await repo.get_by_id(saved_id)
    assert retrieved is None
