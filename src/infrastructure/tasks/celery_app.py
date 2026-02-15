# src/infrastructure/tasks/celery_app.py
from celery import Celery

from src.infrastructure.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "genai_tasks",
    broker=settings.redis.URL,
    backend=settings.redis.URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=3600,
    # Auto-discover tasks in the worker module
    include=["src.infrastructure.tasks.worker"],
)
