"""JDBC data sink adapter using PySpark."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import structlog

from src.domain.ports.data_sink_port import DataSinkPort

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = structlog.get_logger(__name__)


class SparkJDBCSink(DataSinkPort):
    """Writes data to JDBC destinations via PySpark.

    URI format: jdbc:postgresql://host:5432/db?table=schema.table_name
    """

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    def _parse_jdbc_uri(self, uri: str) -> tuple[str, str]:
        """Extract JDBC URL and table name from URI."""
        parsed = urlparse(uri.replace("jdbc:", "", 1))
        params = parse_qs(parsed.query)
        table = params.get("table", [""])[0]
        clean_query = "&".join(f"{k}={v[0]}" for k, v in params.items() if k != "table")
        base = f"jdbc:{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            base += f":{parsed.port}"
        base += parsed.path
        if clean_query:
            base += f"?{clean_query}"
        return base, table

    async def write_records(
        self, uri: str, records: list[dict[str, object]], **options: Any
    ) -> int:
        """Write records to JDBC sink. Runs Spark I/O in a thread."""
        return await asyncio.to_thread(self._write_sync, uri, records, **options)

    def _write_sync(self, uri: str, records: list[dict[str, object]], **options: Any) -> int:
        jdbc_url, table = self._parse_jdbc_uri(uri)
        mode = options.pop("mode", "append")

        writer_options: dict[str, Any] = {
            "url": jdbc_url,
            "dbtable": table,
            "driver": options.pop("driver", "org.postgresql.Driver"),
        }
        if options.get("user"):
            writer_options["user"] = options.pop("user")
        if options.get("password"):
            writer_options["password"] = options.pop("password")

        logger.info("spark.jdbc.write", table=table, mode=mode, count=len(records))

        df = self._spark.createDataFrame(records)
        df.write.format("jdbc").options(**writer_options).mode(mode).save()
        return len(records)
