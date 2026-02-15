"""ChromaDB adapter — implements VectorStorePort and RetrieverPort.

Reference implementation showing:
- Single adapter satisfying multiple ports
- In-memory vector store (no external infrastructure)
- Error translation to domain exceptions
"""

from typing import Any

import chromadb
from chromadb.api import ClientAPI

from src.domain.entities.document import Document
from src.domain.exceptions import RetrievalError
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.ports.vector_store_port import VectorStorePort


class ChromaDBAdapter(VectorStorePort, RetrieverPort):
    """Vector store and retriever backed by ChromaDB.

    Uses ChromaDB's built-in embedding function (all-MiniLM-L6-v2)
    for simplicity. For production, inject a custom embedding function.
    """

    def __init__(
        self,
        client: ClientAPI | None = None,
        collection_name: str = "documents",
    ) -> None:
        self._client = client or chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    async def upsert(self, documents: list[Document]) -> None:
        ids = [doc.id or str(i) for i, doc in enumerate(documents)]
        self._collection.upsert(
            ids=ids,
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata or {} for doc in documents],
        )

    async def search(
        self, query: str, top_k: int = 5, filters: dict[str, str] | None = None
    ) -> list[Document]:
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,  # type: ignore[arg-type]
            )
            return self._to_documents(results)
        except Exception as e:
            raise RetrievalError(f"ChromaDB search failed: {e}") from e

    async def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """RetrieverPort implementation — delegates to search."""
        return await self.search(query, top_k=top_k)

    @staticmethod
    def _to_documents(results: Any) -> list[Document]:
        documents: list[Document] = []
        ids = results.get("ids", [[]])[0]
        contents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for id_, content, metadata, distance in zip(
            ids, contents, metadatas, distances, strict=True
        ):
            documents.append(
                Document(
                    id=id_,
                    content=content,
                    metadata=metadata or {},
                    score=1.0 - distance,
                )
            )
        return documents
