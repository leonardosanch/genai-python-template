"""Port for hallucination checking. Used by verified RAG pipelines."""

from abc import ABC, abstractmethod

from src.domain.value_objects.verification_result import VerificationResult


class HallucinationCheckerPort(ABC):
    """Abstract interface for verifying LLM answers against source context."""

    @abstractmethod
    async def verify(self, answer: str, context: str, query: str) -> VerificationResult:
        """Verify that an answer is grounded in the provided context."""
        ...
