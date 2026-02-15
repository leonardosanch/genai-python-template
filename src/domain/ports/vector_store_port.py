"""Port for vector store operations."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Document


class VectorStorePort(ABC):
    """Abstract interface for vector database operations.

    Implementations: Pinecone, Weaviate, Qdrant, ChromaDB, pgvector.
    """

    @abstractmethod
    async def upsert(self, documents: list[Document]) -> None:
        """Insert or update documents in the vector store."""
        ...

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5, filters: dict[str, str] | None = None
    ) -> list[Document]:
        """Search for similar documents using vector similarity."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        ...
