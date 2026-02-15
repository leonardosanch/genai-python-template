# tests/factories.py
"""Test factories for creating domain objects with sensible defaults."""

from datetime import UTC, datetime
from uuid import uuid4

from src.domain.entities.document import Document
from src.domain.value_objects.llm_response import LLMResponse
from src.domain.value_objects.verification_result import VerificationResult


class DocumentFactory:
    """Factory for creating Document test fixtures."""

    _counter = 0

    @classmethod
    def create(
        cls,
        content: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        id: str | None = None,
        score: float | None = None,
    ) -> Document:
        cls._counter += 1
        return Document(
            content=content or f"Test document content #{cls._counter}",
            metadata=metadata or {"source": "test", "index": cls._counter},
            id=id or str(uuid4()),
            score=score,
            created_at=datetime.now(UTC),
        )

    @classmethod
    def create_batch(cls, count: int, **kwargs: object) -> list[Document]:
        return [cls.create(**kwargs) for _ in range(count)]  # type: ignore[arg-type]


class LLMResponseFactory:
    """Factory for creating LLMResponse test fixtures."""

    @classmethod
    def create(
        cls,
        content: str = "Generated response text.",
        model: str = "gpt-4o-mini",
        prompt_tokens: int = 50,
        completion_tokens: int = 100,
        total_tokens: int | None = None,
        cost_usd: float | None = 0.001,
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            cost_usd=cost_usd,
        )


class VerificationResultFactory:
    """Factory for creating VerificationResult test fixtures."""

    @classmethod
    def create(
        cls,
        is_grounded: bool = True,
        faithfulness_score: float = 0.95,
        unsupported_claims: list[str] | None = None,
        citations: list[str] | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            is_grounded=is_grounded,
            faithfulness_score=faithfulness_score,
            unsupported_claims=unsupported_claims or [],
            citations=citations or ["Source passage supporting the claim."],
        )
