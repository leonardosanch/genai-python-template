# src/infrastructure/database/document_repository.py
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.domain.entities.document import Document
from src.domain.ports.repository_port import RepositoryPort
from src.infrastructure.database.models import DocumentRecord


class SQLAlchemyDocumentRepository(RepositoryPort[Document]):
    """SQLAlchemy implementation of DocumentRepository."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def save(self, entity: Document) -> str:
        """Save a document to the database."""
        async with self._session_factory() as session:
            record = DocumentRecord(
                id=entity.id or str(uuid4()),
                content=entity.content,
                metadata_=entity.metadata,
                created_at=entity.created_at,
            )
            session.add(record)
            await session.commit()
            return record.id

    async def get_by_id(self, id: str) -> Document | None:
        """Get a document by ID."""
        async with self._session_factory() as session:
            record = await session.get(DocumentRecord, id)
            if not record:
                return None
            return self._to_entity(record)

    async def list(self, offset: int = 0, limit: int = 20) -> list[Document]:
        """List all documents."""
        async with self._session_factory() as session:
            stmt = select(DocumentRecord).offset(offset).limit(limit)
            result = await session.execute(stmt)
            return [self._to_entity(r) for r in result.scalars()]

    async def delete(self, id: str) -> None:
        """Delete a document by ID."""
        async with self._session_factory() as session:
            stmt = delete(DocumentRecord).where(DocumentRecord.id == id)
            await session.execute(stmt)
            await session.commit()

    def _to_entity(self, record: DocumentRecord) -> Document:
        """Convert record to entity."""
        return Document(
            id=record.id,
            content=record.content,
            metadata=record.metadata_,
            created_at=record.created_at,
        )
