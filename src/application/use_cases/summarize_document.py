"""Use case: Summarize a document using RAG.

This is a reference implementation showing:
- Dependency on ports (not concrete implementations)
- Structured output with Pydantic
- Single responsibility
"""

from pydantic import BaseModel, Field

from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort


class Summary(BaseModel):
    """Structured output schema for document summarization."""

    title: str = Field(description="Brief title of the summary")
    key_points: list[str] = Field(description="Main points extracted")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    sources: list[str] = Field(default_factory=list, description="Cited sources")


SUMMARIZE_PROMPT = """You are a document summarization expert.

## Context
{context}

## Query
{query}

## Instructions
- Extract key points (max 5)
- Include a confidence score (0-1)
- Cite sources when available
- Respond in JSON matching the provided schema
"""


class SummarizeDocumentUseCase:
    """Summarize documents retrieved by a RAG pipeline.

    Dependencies are injected via constructor — this use case
    depends only on ports, never on concrete implementations.
    """

    def __init__(self, llm: LLMPort, retriever: RetrieverPort) -> None:
        self._llm = llm
        self._retriever = retriever

    async def execute(self, query: str) -> Summary:
        docs = await self._retriever.retrieve(query, top_k=5)
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.content}" for doc in docs
        )
        prompt = SUMMARIZE_PROMPT.format(context=context, query=query)
        return await self._llm.generate_structured(prompt, schema=Summary)
