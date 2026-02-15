"""Tests for domain exceptions — hierarchy and raising."""

import pytest

from src.domain.exceptions import (
    DataQualityError,
    DataSinkError,
    DataSourceError,
    DomainError,
    ExtractionError,
    HallucinationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LoadError,
    PipelineError,
    RetrievalError,
    SchemaEvolutionError,
    SchemaValidationError,
    TokenBudgetExceededError,
    TransformationError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Verify the exception inheritance tree."""

    def test_llm_errors_are_domain_errors(self) -> None:
        assert issubclass(LLMError, DomainError)
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_pipeline_errors_are_domain_errors(self) -> None:
        assert issubclass(PipelineError, DomainError)
        assert issubclass(ExtractionError, PipelineError)
        assert issubclass(TransformationError, PipelineError)
        assert issubclass(LoadError, PipelineError)
        assert issubclass(DataSourceError, PipelineError)
        assert issubclass(DataSinkError, PipelineError)

    def test_data_quality_errors(self) -> None:
        assert issubclass(DataQualityError, PipelineError)
        assert issubclass(SchemaValidationError, DataQualityError)

    def test_other_domain_errors(self) -> None:
        assert issubclass(RetrievalError, DomainError)
        assert issubclass(ValidationError, DomainError)
        assert issubclass(TokenBudgetExceededError, DomainError)
        assert issubclass(HallucinationError, DomainError)
        assert issubclass(SchemaEvolutionError, DomainError)


class TestExceptionMessages:
    """Verify exceptions carry messages correctly."""

    def test_llm_timeout_message(self) -> None:
        with pytest.raises(LLMTimeoutError, match="timed out"):
            raise LLMTimeoutError("Request timed out after 30s")

    def test_hallucination_message(self) -> None:
        with pytest.raises(HallucinationError, match="failed verification"):
            raise HallucinationError("Answer failed verification")

    def test_data_sink_message(self) -> None:
        with pytest.raises(DataSinkError, match="write failed"):
            raise DataSinkError("write failed for s3://bucket")

    def test_catch_as_domain_error(self) -> None:
        """All domain exceptions can be caught as DomainError."""
        exceptions = [
            LLMTimeoutError("t"),
            LLMRateLimitError("r"),
            RetrievalError("re"),
            ValidationError("v"),
            HallucinationError("h"),
            ExtractionError("e"),
            DataSinkError("d"),
        ]
        for exc in exceptions:
            with pytest.raises(DomainError):
                raise exc
