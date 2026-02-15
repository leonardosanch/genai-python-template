"""Cloud-specific deployment examples."""

from src.examples.cloud.aws_lambda_rag import lambda_handler
from src.examples.cloud.azure_container_apps import app as azure_app
from src.examples.cloud.gcp_cloud_run_agent import app as gcp_app

__all__ = ["lambda_handler", "gcp_app", "azure_app"]
