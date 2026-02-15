"""JDBC data source adapter using PySpark."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import structlog

from src.domain.ports.data_source_port import DataSourcePort
from src.domain.value_objects.schema_definition import FieldDefinition, SchemaDefinition

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = structlog.get_logger(__name__)


class SparkJDBCSource(DataSourcePort):
    """Reads data from JDBC sources via PySpark.

    URI format: jdbc:postgresql://host:5432/db?table=schema.table_name
    Supports parallel reads via partition_column/num_partitions options.
    """

    def __init__(self, spark: SparkSession, fetch_size: int = 10_000) -> None:
        self._spark = spark
        self._fetch_size = fetch_size

    def _parse_jdbc_uri(self, uri: str) -> tuple[str, str]:
        """Extract JDBC URL and table name from URI."""
        parsed = urlparse(uri.replace("jdbc:", "", 1))
        params = parse_qs(parsed.query)
        table = params.get("table", [""])[0]
        # Reconstruct clean JDBC URL without table param
        clean_query = "&".join(f"{k}={v[0]}" for k, v in params.items() if k != "table")
        base = f"jdbc:{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            base += f":{parsed.port}"
        base += parsed.path
        if clean_query:
            base += f"?{clean_query}"
        return base, table

    async def read_records(self, uri: str, **options: Any) -> list[dict[str, object]]:
        """Read records from JDBC source. Runs Spark I/O in a thread."""
        return await asyncio.to_thread(self._read_records_sync, uri, **options)

    def _read_records_sync(self, uri: str, **options: Any) -> list[dict[str, object]]:
        jdbc_url, table = self._parse_jdbc_uri(uri)
        query: str | None = options.get("query")

        reader_options: dict[str, Any] = {
            "url": jdbc_url,
            "driver": options.get("driver", "org.postgresql.Driver"),
            "fetchsize": str(options.get("fetch_size", self._fetch_size)),
        }

        if options.get("user"):
            reader_options["user"] = options["user"]
        if options.get("password"):
            reader_options["password"] = options["password"]

        if query:
            reader_options["dbtable"] = f"({query}) AS subq"
        else:
            reader_options["dbtable"] = table

        # Parallel read support
        partition_column = options.get("partition_column")
        if partition_column:
            reader_options["partitionColumn"] = partition_column
            reader_options["numPartitions"] = str(options.get("num_partitions", 4))
            reader_options["lowerBound"] = str(options.get("lower_bound", 0))
            reader_options["upperBound"] = str(options.get("upper_bound", 1_000_000))

        logger.info("spark.jdbc.read", table=reader_options.get("dbtable"), url=jdbc_url)
        df = self._spark.read.format("jdbc").options(**reader_options).load()

        rows = df.collect()
        columns = df.columns
        return [{col: row[col] for col in columns} for row in rows]

    async def read_schema(self, uri: str) -> SchemaDefinition | None:
        """Infer schema from JDBC source without reading data."""
        return await asyncio.to_thread(self._read_schema_sync, uri)

    def _read_schema_sync(self, uri: str) -> SchemaDefinition | None:
        jdbc_url, table = self._parse_jdbc_uri(uri)
        df = (
            self._spark.read.format("jdbc")
            .options(url=jdbc_url, dbtable=f"(SELECT * FROM {table} WHERE 1=0) AS schema_q")
            .load()
        )
        fields = tuple(
            FieldDefinition(
                name=f.name,
                data_type=str(f.dataType),
                nullable=f.nullable,
            )
            for f in df.schema.fields
        )
        return SchemaDefinition(name=table, version="inferred", fields=fields)
