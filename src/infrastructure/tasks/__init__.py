# src/infrastructure/tasks/__init__.py
from .celery_app import celery_app
from .worker import run_pipeline_task

__all__ = ["celery_app", "run_pipeline_task"]
