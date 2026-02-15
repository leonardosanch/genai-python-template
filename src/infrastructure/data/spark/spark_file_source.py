"""File-based data source adapter using PySpark."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from src.domain.ports.data_source_port import DataSourcePort
from src.domain.value_objects.schema_definition import FieldDefinition, SchemaDefinition

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
    """Infer Spark read format from file extension."""
    for ext, fmt in _FORMAT_MAP.items():
        if uri.endswith(ext) or f"/{ext.lstrip('.')}/" in uri:
            return fmt
    # Check if path looks like a delta table (directory without extension)
    if "/delta/" in uri or uri.endswith("/delta"):
        return "delta"
    return "parquet"


class SparkFileSource(DataSourcePort):
    """Reads Parquet, CSV, JSON, Delta files via PySpark."""

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    async def read_records(self, uri: str, **options: Any) -> list[dict[str, object]]:
        """Read records from file source. Runs Spark I/O in a thread."""
        return await asyncio.to_thread(self._read_sync, uri, **options)

    def _read_sync(self, uri: str, **options: Any) -> list[dict[str, object]]:
        fmt = options.pop("format", None) or _infer_format(uri)
        logger.info("spark.file.read", uri=uri, format=fmt)

        reader = self._spark.read.format(fmt)
        if fmt == "csv":
            reader = reader.option("header", options.pop("header", "true"))
            reader = reader.option("inferSchema", options.pop("infer_schema", "true"))

        for key, value in options.items():
            reader = reader.option(key, str(value))

        df = reader.load(uri)
        rows = df.collect()
        columns = df.columns
        return [{col: row[col] for col in columns} for row in rows]

    async def read_schema(self, uri: str) -> SchemaDefinition | None:
        """Infer schema from file source."""
        return await asyncio.to_thread(self._read_schema_sync, uri)

    def _read_schema_sync(self, uri: str) -> SchemaDefinition | None:
        fmt = _infer_format(uri)
        df = self._spark.read.format(fmt).load(uri)
        fields = tuple(
            FieldDefinition(
                name=f.name,
                data_type=str(f.dataType),
                nullable=f.nullable,
            )
            for f in df.schema.fields
        )
        return SchemaDefinition(name=uri.split("/")[-1], version="inferred", fields=fields)
