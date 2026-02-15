# src/infrastructure/rag/llm_reranker.py
"""LLM-based document reranker using structured output."""

import structlog
from pydantic import BaseModel, Field

from src.domain.entities.document import Document
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.reranker_port import RerankerPort

logger = structlog.get_logger(__name__)


class RelevanceScore(BaseModel):
    """Structured output for document relevance scoring."""

    score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    reasoning: str = Field(description="Brief explanation of relevance")


class LLMReranker(RerankerPort):
    """Reranker that uses an LLM to score document relevance.

    Sends each document + query to the LLM for scoring,
    then sorts by score and returns top_k.
    """

    PROMPT_TEMPLATE = (
        "Rate the relevance of the following document to the query.\n\n"
        "Query: {query}\n\n"
        "Document:\n{document}\n\n"
        "Provide a relevance score between 0.0 (completely irrelevant) "
        "and 1.0 (perfectly relevant)."
    )

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        if not documents:
            return []

        scored: list[tuple[float, Document]] = []
        for doc in documents:
            try:
                prompt = self.PROMPT_TEMPLATE.format(query=query, document=doc.content)
                result = await self._llm.generate_structured(prompt, RelevanceScore)
                scored.append((result.score, doc))
            except Exception:
                logger.warning("reranker_scoring_failed", doc_id=doc.id)
                scored.append((0.0, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            Document(
                content=doc.content,
                metadata=doc.metadata,
                id=doc.id,
                score=score,
                created_at=doc.created_at,
            )
            for score, doc in scored[:top_k]
        ]
