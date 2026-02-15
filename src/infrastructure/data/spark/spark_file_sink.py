"""File-based data sink adapter using PySpark."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.ports.data_sink_port import DataSinkPort

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = structlog.get_logger(__name__)

_FORMAT_MAP: dict[str, str] = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".json": "json",
    ".delta": "delta",
    ".orc": "orc",
}


def _infer_format(uri: str) -> str:
    """Infer Spark write format from file extension."""
    for ext, fmt in _FORMAT_MAP.items():
        if uri.endswith(ext) or uri.endswith(f"/{ext.lstrip('.')}"):
            return fmt
    return "parquet"


class SparkFileSink(DataSinkPort):
    """Writes Parquet, CSV, JSON, Delta files via PySpark."""

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    async def write_records(
        self, uri: str, records: list[dict[str, object]], **options: Any
    ) -> int:
        """Write records to file sink. Runs Spark I/O in a thread."""
        return await asyncio.to_thread(self._write_sync, uri, records, **options)

    def _write_sync(self, uri: str, records: list[dict[str, object]], **options: Any) -> int:
        fmt = options.pop("format", None) or _infer_format(uri)
        mode = options.pop("mode", "overwrite")
        partition_by: list[str] | None = options.pop("partition_by", None)

        logger.info("spark.file.write", uri=uri, format=fmt, mode=mode, count=len(records))

        df = self._spark.createDataFrame(records)
        writer = df.write.format(fmt).mode(mode)

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        for key, value in options.items():
            writer = writer.option(key, str(value))

        writer.save(uri)
        return len(records)
