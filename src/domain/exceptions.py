"""Domain exceptions — business-level errors.

These exceptions are raised by domain and application layers.
Infrastructure and interface layers catch and translate them
into appropriate responses (HTTP errors, CLI messages, etc.).
"""


class DomainError(Exception):
    """Base exception for all domain errors."""


class LLMError(DomainError):
    """Error during LLM interaction."""


class LLMTimeoutError(LLMError):
    """LLM call timed out."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""


class RetrievalError(DomainError):
    """Error during document retrieval."""


class ValidationError(DomainError):
    """Domain validation error."""


class TokenBudgetExceededError(DomainError):
    """Token budget for the period has been exceeded."""


class PipelineError(DomainError):
    """Base exception for all pipeline errors."""


class ExtractionError(PipelineError):
    """Error during pipeline extraction phase."""


class TransformationError(PipelineError):
    """Error during pipeline transformation phase."""


class LoadError(PipelineError):
    """Error during pipeline load phase."""


class DataSourceError(PipelineError):
    """Error reading from a data source."""


class DataSinkError(PipelineError):
    """Error writing to a data destination."""


class DataQualityError(PipelineError):
    """Data validation failed."""


class SchemaValidationError(DataQualityError):
    """Schema mismatch during validation."""


class SchemaEvolutionError(DomainError):
    """Incompatible schema change detected."""


class HallucinationError(DomainError):
    """LLM answer failed hallucination verification."""
