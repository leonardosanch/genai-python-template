"""Data cleaning pipeline — ETL for data quality improvements."""

from dataclasses import dataclass, field

import structlog

from src.application.pipelines.base import Pipeline
from src.domain.exceptions import DataSinkError, DataSourceError, LoadError
from src.domain.ports.data_sink_port import DataSinkPort
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.value_objects.schema_definition import SchemaDefinition

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for data cleaning operations."""

    strip_whitespace: bool = True
    drop_nulls: bool = False
    drop_duplicates: bool = False
    null_columns: tuple[str, ...] = field(default_factory=tuple)
    dedup_columns: tuple[str, ...] = field(default_factory=tuple)


class DataCleaningPipeline(Pipeline):
    """Pipeline that reads, cleans, validates, and writes data.

    Extends the base Pipeline template method with concrete
    data cleaning logic using injected ports.
    """

    def __init__(
        self,
        source: DataSourcePort,
        sink: DataSinkPort,
        validator: DataValidatorPort,
        source_uri: str,
        sink_uri: str,
        config: CleaningConfig | None = None,
        schema: SchemaDefinition | None = None,
    ) -> None:
        self._source = source
        self._sink = sink
        self._validator = validator
        self._source_uri = source_uri
        self._sink_uri = sink_uri
        self._config = config or CleaningConfig()
        self._schema = schema

    async def extract(self) -> list[dict[str, object]]:
        """Read records from the data source."""
        try:
            records = await self._source.read_records(self._source_uri)
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(f"Failed to read from {self._source_uri}: {e}") from e
        logger.info("cleaning.extract", records=len(records), source=self._source_uri)
        return records

    async def transform(self, data: list[dict[str, object]]) -> list[dict[str, object]]:
        """Apply cleaning transformations."""
        result = list(data)
        cfg = self._config

        if cfg.strip_whitespace:
            result = [
                {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
                for row in result
            ]

        if cfg.drop_nulls:
            columns = cfg.null_columns or tuple(result[0].keys()) if result else ()
            result = [row for row in result if all(row.get(c) is not None for c in columns)]

        if cfg.drop_duplicates and result:
            columns = cfg.dedup_columns or tuple(result[0].keys())
            seen: set[tuple[object, ...]] = set()
            deduped: list[dict[str, object]] = []
            for row in result:
                key = tuple(row.get(c) for c in columns)
                if key not in seen:
                    seen.add(key)
                    deduped.append(row)
            result = deduped

        logger.info(
            "cleaning.transform",
            input_records=len(data),
            output_records=len(result),
        )
        return result

    async def load(self, data: list[dict[str, object]]) -> None:
        """Validate and write records to the data sink."""
        if self._schema:
            quality = await self._validator.validate(data, self._schema)
            if not quality.is_valid:
                logger.warning("cleaning.validation_failed", errors=quality.errors)

        try:
            written = await self._sink.write_records(self._sink_uri, data)
        except DataSinkError:
            raise
        except Exception as e:
            raise LoadError(f"Failed to write to {self._sink_uri}: {e}") from e
        logger.info("cleaning.load", records_written=written, sink=self._sink_uri)
