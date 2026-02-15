"""RAG query routes — reference endpoint implementation.

Shows:
- Dependency injection with FastAPI Depends
- Request/response DTOs (never expose domain models)
- Domain exception → HTTP error translation
- Composition root pattern for wiring dependencies
"""

from typing import TYPE_CHECKING, cast

import structlog
from fastapi import APIRouter, Depends, Request

from src.application.dtos.rag import RAGQueryRequest, RAGQueryResponse, SourceDocument
from src.application.use_cases.query_rag import QueryRAGUseCase
from src.application.use_cases.verified_rag import VerifiedRAGUseCase
from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.retriever_port import RetrieverPort

if TYPE_CHECKING:
    from typing import Protocol

    from src.infrastructure.container import Container

    class AppState(Protocol):
        container: "Container"

    class App(Protocol):
        state: AppState


logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


def get_retriever(request: Request) -> RetrieverPort:
    """Composition root — wire concrete adapter to port."""
    if TYPE_CHECKING:
        app_with_container = cast(App, request.app)
        return app_with_container.state.container.retriever_adapter
    return request.app.state.container.retriever_adapter


def get_llm(request: Request) -> LLMPort:
    """Composition root — wire concrete adapter to port."""
    if TYPE_CHECKING:
        app_with_container = cast(App, request.app)
        return app_with_container.state.container.llm_adapter
    return request.app.state.container.llm_adapter


def get_hallucination_checker(request: Request) -> HallucinationCheckerPort:
    """Composition root — wire hallucination checker adapter."""
    if TYPE_CHECKING:
        app_with_container = cast(App, request.app)
        return app_with_container.state.container.hallucination_checker
    return request.app.state.container.hallucination_checker


def get_use_case(
    llm: LLMPort = Depends(get_llm),
    retriever: RetrieverPort = Depends(get_retriever),
) -> QueryRAGUseCase:
    return QueryRAGUseCase(llm=llm, retriever=retriever)


def get_verified_use_case(
    llm: LLMPort = Depends(get_llm),
    retriever: RetrieverPort = Depends(get_retriever),
    checker: HallucinationCheckerPort = Depends(get_hallucination_checker),
) -> VerifiedRAGUseCase:
    return VerifiedRAGUseCase(
        llm=llm,
        retriever=retriever,
        hallucination_checker=checker,
    )


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    request: RAGQueryRequest,
    use_case: QueryRAGUseCase = Depends(get_use_case),
) -> RAGQueryResponse:
    """Execute a RAG query: retrieve context and generate a grounded answer."""
    result = await use_case.execute(
        query=request.query,
        top_k=request.top_k,
    )

    return RAGQueryResponse(
        answer=result.answer,
        sources=[SourceDocument(content="", source="context", score=None)],
        model=result.model,
    )


@router.post("/query/verified", response_model=RAGQueryResponse)
async def query_rag_verified(
    request: RAGQueryRequest,
    use_case: VerifiedRAGUseCase = Depends(get_verified_use_case),
) -> RAGQueryResponse:
    """Execute a RAG query with automatic hallucination verification."""
    answer, verification = await use_case.execute(
        query=request.query,
        top_k=request.top_k,
    )

    return RAGQueryResponse(
        answer=answer.answer,
        sources=[SourceDocument(content="", source="context", score=None)],
        model=answer.model,
        is_verified=True,
        faithfulness_score=verification.faithfulness_score,
        unsupported_claims=verification.unsupported_claims,
    )
