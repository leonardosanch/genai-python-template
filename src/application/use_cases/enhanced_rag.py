# src/application/use_cases/enhanced_rag.py
"""Enhanced RAG use case with optional reranking and semantic caching."""

from dataclasses import dataclass

import structlog

from src.domain.entities.document import Document
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.reranker_port import RerankerPort
from src.domain.ports.retriever_port import RetrieverPort
from src.infrastructure.cache.semantic_cache import SemanticCache

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class EnhancedRAGResult:
    """Result of the enhanced RAG pipeline."""

    answer: str
    sources: list[Document]
    cache_hit: bool


class EnhancedRAGUseCase:
    """RAG pipeline with optional reranking and semantic caching.

    Pipeline: cache check → retrieve → rerank (optional) → generate → cache set.
    """

    PROMPT_TEMPLATE = (
        "Answer the question based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        llm: LLMPort,
        retriever: RetrieverPort,
        reranker: RerankerPort | None = None,
        cache: SemanticCache | None = None,
        top_k: int = 5,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._reranker = reranker
        self._cache = cache
        self._top_k = top_k

    async def execute(self, query: str) -> EnhancedRAGResult:
        """Execute the enhanced RAG pipeline."""
        # 1. Check semantic cache
        if self._cache:
            cached = await self._cache.get(query)
            if cached:
                logger.info("enhanced_rag_cache_hit", query=query[:50])
                return EnhancedRAGResult(answer=cached, sources=[], cache_hit=True)

        # 2. Retrieve — fetch more if reranker is available
        retrieve_k = self._top_k * 2 if self._reranker else self._top_k
        documents = await self._retriever.retrieve(query, top_k=retrieve_k)

        # 3. Rerank (optional)
        if self._reranker and documents:
            documents = await self._reranker.rerank(query, documents, top_k=self._top_k)
        else:
            documents = documents[: self._top_k]

        # 4. Generate answer
        context = "\n\n---\n\n".join(doc.content for doc in documents)
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)
        answer = await self._llm.generate(prompt)

        # 5. Cache result
        if self._cache:
            await self._cache.set(query, answer)

        logger.info(
            "enhanced_rag_complete",
            query=query[:50],
            num_sources=len(documents),
        )
        return EnhancedRAGResult(answer=answer, sources=documents, cache_hit=False)
