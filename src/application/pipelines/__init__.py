"""Pipeline abstractions for ETL workflows."""

from src.application.pipelines.base import Pipeline
from src.application.pipelines.document_ingestion import DocumentIngestionPipeline
from src.application.pipelines.spark_pipeline import SparkPipeline

__all__ = ["DocumentIngestionPipeline", "Pipeline", "SparkPipeline"]
