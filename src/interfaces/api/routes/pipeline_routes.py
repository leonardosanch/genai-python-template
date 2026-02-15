"""Pipeline API routes."""

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, Request

from src.application.dtos.pipeline import (
    AsyncTaskResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    TaskStatusResponse,
)
from src.application.pipelines.document_ingestion import DocumentIngestionPipeline
from src.domain.ports.storage_port import StoragePort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.role import Role
from src.infrastructure.container import Container
from src.infrastructure.tasks import celery_app, run_pipeline_task
from src.interfaces.api.dependencies.rbac import require_roles

router = APIRouter(prefix="/api/v1/pipelines", tags=["pipelines"])


def get_container(request: Request) -> Container:
    """Get dependency injection container from app state."""
    container: Container = request.app.state.container
    return container


def get_storage(container: Container = Depends(get_container)) -> StoragePort:
    """Get storage adapter from container."""
    storage: StoragePort = container.storage_adapter
    return storage


def get_vector_store(container: Container = Depends(get_container)) -> VectorStorePort:
    """Get vector store adapter from container."""
    return container.vector_store_adapter


@router.post(
    "/ingest",
    response_model=PipelineRunResponse,
    dependencies=[Depends(require_roles(Role.OPERATOR, Role.ADMIN))],
)
async def ingest_documents(
    request: PipelineRunRequest,
    storage: StoragePort = Depends(get_storage),
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> PipelineRunResponse:
    """Run document ingestion pipeline.

    Reads text files from storage, chunks them, and loads them into the vector store.

    Args:
        request: Pipeline configuration
        storage: Storage adapter (injected)
        vector_store: Vector store adapter (injected)

    Returns:
        Pipeline execution result with metrics
    """
    # Create and run pipeline
    pipeline = DocumentIngestionPipeline(
        storage=storage,
        vector_store=vector_store,
        source_prefix=request.source_path,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    result = await pipeline.run()

    # Convert PipelineResult to response DTO
    return PipelineRunResponse(
        status=result.status,
        records_processed=result.records_processed,
        records_failed=result.records_failed,
        duration_seconds=result.duration_seconds,
        errors=result.errors,
    )


@router.post(
    "/ingest/async",
    response_model=AsyncTaskResponse,
    dependencies=[Depends(require_roles(Role.OPERATOR, Role.ADMIN))],
)
async def ingest_documents_async(
    request: PipelineRunRequest,
) -> AsyncTaskResponse:
    """Trigger document ingestion pipeline asynchronously.

    Returns immediately with a task ID.
    """
    # Convert Pydantic model to dict for Celery
    params = request.model_dump()

    # Trigger task
    task = run_pipeline_task.delay(
        pipeline_name="document_ingestion",
        params=params,
    )

    return AsyncTaskResponse(
        task_id=task.id,
        status="pending",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Check status of a background task."""
    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        status=result.status.lower(),
        result=None,
    )

    if result.ready():
        if result.successful():
            response.result = result.result
        else:
            # If failed, result.result contains exception
            response.status = "failed"
            response.result = {"error": str(result.result)}

    return response
