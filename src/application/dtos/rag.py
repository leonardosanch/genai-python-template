"""DTOs for RAG query endpoints.

Reference implementation showing:
- Request/response separation from domain models
- Pydantic validation at the boundary
- Explicit field descriptions for OpenAPI docs
"""

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Incoming RAG query from the API."""

    query: str = Field(min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Documents to retrieve")


class SourceDocument(BaseModel):
    """A retrieved document included in the response."""

    content: str
    source: str
    score: float | None = None


class RAGQueryResponse(BaseModel):
    """Structured response from a RAG query."""

    answer: str = Field(description="Generated answer grounded in retrieved context")
    sources: list[SourceDocument] = Field(default_factory=list)
    model: str = Field(description="LLM model used for generation")
    is_verified: bool = Field(default=False, description="Whether hallucination check was run")
    faithfulness_score: float | None = Field(
        default=None, description="Faithfulness score from verification (0.0-1.0)"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not supported by context"
    )
