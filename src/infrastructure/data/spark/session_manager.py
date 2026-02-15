"""Thread-safe singleton manager for SparkSession lifecycle."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from src.infrastructure.config.settings import SparkSettings


class SparkSessionManager:
    """Singleton SparkSession factory.

    SparkSession creation is expensive (~seconds). This manager ensures
    a single session is reused across all pipelines in the process.
    """

    _instance: SparkSession | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_create(cls, settings: SparkSettings) -> SparkSession:
        """Get existing or create new SparkSession from settings."""
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            # Double-checked locking
            if cls._instance is not None:
                return cls._instance

            from pyspark.sql import SparkSession as _SparkSession

            builder: Any = (
                _SparkSession.builder.appName(settings.APP_NAME)
                .master(settings.MASTER)
                .config("spark.sql.shuffle.partitions", str(settings.SHUFFLE_PARTITIONS))
                .config("spark.driver.memory", settings.DRIVER_MEMORY)
                .config("spark.executor.memory", settings.EXECUTOR_MEMORY)
                .config("spark.executor.cores", str(settings.EXECUTOR_CORES))
                .config("spark.sql.warehouse.dir", settings.WAREHOUSE_DIR)
            )

            if settings.DELTA_ENABLED:
                builder = builder.config(
                    "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
                ).config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                )

            for key, value in settings.EXTRA_CONF.items():
                builder = builder.config(key, value)

            cls._instance = builder.getOrCreate()
            return cls._instance

    @classmethod
    def stop(cls) -> None:
        """Stop the SparkSession and release resources."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None
