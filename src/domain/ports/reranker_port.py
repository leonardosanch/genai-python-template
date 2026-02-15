# src/domain/ports/reranker_port.py
"""Port for document reranking in RAG pipelines."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Document


class RerankerPort(ABC):
    """Abstract interface for reranking retrieved documents.

    Rerankers refine the initial retrieval by scoring documents
    against the query with a more sophisticated model.
    """

    @abstractmethod
    async def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank documents by relevance to the query.

        Args:
            query: The user query.
            documents: Initially retrieved documents.
            top_k: Number of top documents to return.

        Returns:
            Reranked documents sorted by relevance (highest first).
        """
        ...
