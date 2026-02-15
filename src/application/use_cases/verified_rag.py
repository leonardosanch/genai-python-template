"""Use case: Verified RAG — RAG with automatic hallucination checking.

Composes QueryRAGUseCase with HallucinationCheckerPort to verify
that generated answers are grounded in the retrieved context.
"""

import structlog

from src.application.use_cases.query_rag import QueryRAGUseCase, RAGAnswer
from src.domain.exceptions import HallucinationError
from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.value_objects.verification_result import VerificationResult

logger = structlog.get_logger()


class VerifiedRAGUseCase:
    """Execute a RAG query with automatic faithfulness verification.

    If the answer fails verification after retries, raises HallucinationError.
    """

    def __init__(
        self,
        llm: LLMPort,
        retriever: RetrieverPort,
        hallucination_checker: HallucinationCheckerPort,
        faithfulness_threshold: float = 0.7,
        max_retries: int = 1,
    ) -> None:
        self._rag = QueryRAGUseCase(llm=llm, retriever=retriever)
        self._retriever = retriever
        self._checker = hallucination_checker
        self._threshold = faithfulness_threshold
        self._max_retries = max_retries

    async def execute(self, query: str, top_k: int = 5) -> tuple[RAGAnswer, VerificationResult]:
        """Execute RAG query and verify the answer is grounded.

        Returns:
            Tuple of (RAGAnswer, VerificationResult) if verification passes.

        Raises:
            HallucinationError: If answer fails verification after all retries.
        """
        last_answer: RAGAnswer | None = None
        last_verification: VerificationResult | None = None

        for attempt in range(1 + self._max_retries):
            answer = await self._rag.execute(query, top_k=top_k)
            last_answer = answer

            # Build context string from the retriever for verification
            docs = await self._retriever.retrieve(query, top_k=top_k)
            context = "\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.content}" for doc in docs
            )

            verification = await self._checker.verify(
                answer=answer.answer,
                context=context,
                query=query,
            )
            last_verification = verification

            logger.info(
                "verified_rag_attempt",
                attempt=attempt + 1,
                faithfulness_score=verification.faithfulness_score,
                is_grounded=verification.is_grounded,
                threshold=self._threshold,
            )

            if verification.faithfulness_score >= self._threshold:
                return answer, verification

        assert last_answer is not None
        assert last_verification is not None

        raise HallucinationError(
            f"Answer failed faithfulness verification after {1 + self._max_retries} attempts. "
            f"Score: {last_verification.faithfulness_score:.2f}, "
            f"threshold: {self._threshold:.2f}. "
            f"Unsupported claims: {last_verification.unsupported_claims}"
        )
