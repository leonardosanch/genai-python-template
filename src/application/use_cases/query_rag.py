"""Use case: RAG query — retrieve context and generate an answer.

Reference implementation showing:
- Full RAG pipeline as a use case
- Structured LLM output with Pydantic
- Dependency on ports only (LLMPort, RetrieverPort)
"""

from pydantic import BaseModel, Field

from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort


class RAGAnswer(BaseModel):
    """Schema for structured LLM output in RAG queries."""

    answer: str = Field(description="Answer grounded in the provided context")
    model: str = Field(default="", description="Model used (populated after generation)")


RAG_QUERY_PROMPT = """You are a helpful assistant. Answer the user's question
using ONLY the provided context. If the context does not contain enough
information, say so explicitly.

## Context
{context}

## Question
{question}

## Instructions
- Base your answer strictly on the context above
- Cite sources when possible
- Be concise and direct
"""


class QueryRAGUseCase:
    """Execute a RAG query: retrieve documents, generate grounded answer.

    Dependencies injected via constructor — depends only on ports.
    """

    def __init__(self, llm: LLMPort, retriever: RetrieverPort) -> None:
        self._llm = llm
        self._retriever = retriever

    async def execute(self, query: str, top_k: int = 5) -> RAGAnswer:
        docs = await self._retriever.retrieve(query, top_k=top_k)

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.content}" for doc in docs
        )

        prompt = RAG_QUERY_PROMPT.format(context=context, question=query)
        return await self._llm.generate_structured(prompt, schema=RAGAnswer)
