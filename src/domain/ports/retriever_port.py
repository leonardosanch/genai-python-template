"""Port for document retrieval. Used by RAG pipelines."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Document


class RetrieverPort(ABC):
    """Abstract interface for retrieving relevant documents given a query."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve the most relevant documents for a query."""
        ...
