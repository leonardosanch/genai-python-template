# src/infrastructure/tasks/worker.py
import asyncio
from typing import Any

import structlog

from src.application.pipelines.document_ingestion import DocumentIngestionPipeline
from src.infrastructure.config.settings import get_settings
from src.infrastructure.storage.local_storage import LocalStorage
from src.infrastructure.tasks.celery_app import celery_app
from src.infrastructure.vectorstores.chromadb_adapter import ChromaDBAdapter

logger = structlog.get_logger()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)  # type: ignore
def run_pipeline_task(self: Any, pipeline_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a pipeline as a background task.

    Args:
        pipeline_name: Name of the pipeline to run (e.g., "document_ingestion")
        params: Parameters for the pipeline (e.g., source_path, chunk_size)

    Returns:
        Serialization of PipelineResult
    """
    logger.info("task_started", task_id=self.request.id, pipeline=pipeline_name)

    async def _run() -> dict[str, Any]:
        settings = get_settings()

        # NOTE: In a real production app with multiple pipelines,
        # use a Factory or Registry pattern to pick the right pipeline.
        # For now, we only have one reference pipeline.
        if pipeline_name != "document_ingestion":
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Initialize adapters
        # Note: We create fresh adapters for the worker process
        storage_path = getattr(settings, "STORAGE_PATH", "./data")
        storage = LocalStorage(base_path=storage_path)
        vector_store = ChromaDBAdapter()

        pipeline = DocumentIngestionPipeline(
            storage=storage,
            vector_store=vector_store,
            source_prefix=params.get("source_path", ""),
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 200),
        )

        result = await pipeline.run()

        # Return dict serialization of the dataclass
        return {
            "status": result.status,
            "records_processed": result.records_processed,
            "records_failed": result.records_failed,
            "duration_seconds": result.duration_seconds,
            "errors": result.errors,
        }

    try:
        result = asyncio.run(_run())
        logger.info("task_succeeded", task_id=self.request.id, result=result)
        return result
    except Exception as e:
        logger.error("task_failed", task_id=self.request.id, error=str(e))
        # Retry logic handled by Celery
        raise self.retry(exc=e)
