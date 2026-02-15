"""Local file data source — reads CSV/JSON from the local filesystem."""

import csv
import json
from io import StringIO
from typing import Any

import aiofiles

from src.domain.exceptions import DataSourceError
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.value_objects.schema_definition import (
    FieldDefinition,
    SchemaDefinition,
)


class LocalFileSource(DataSourcePort):
    """DataSourcePort implementation for local CSV and JSON files."""

    async def read_records(self, uri: str, **options: Any) -> list[dict[str, object]]:
        """Read records from a local CSV or JSON file.

        Args:
            uri: File path. Extension determines format (.csv or .json).
            **options: Passed to csv.DictReader or json.load.

        Returns:
            List of records as dictionaries.

        Raises:
            DataSourceError: If the file cannot be read or parsed.
        """
        try:
            async with aiofiles.open(uri, encoding="utf-8") as f:
                content = await f.read()
        except FileNotFoundError as e:
            raise DataSourceError(f"File not found: {uri}") from e
        except OSError as e:
            raise DataSourceError(f"Cannot read file: {uri}: {e}") from e

        lower = uri.lower()
        try:
            if lower.endswith(".json"):
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                raise DataSourceError(f"JSON file must contain a list: {uri}")
            elif lower.endswith(".csv"):
                reader = csv.DictReader(StringIO(content))
                return [dict(row) for row in reader]
            else:
                raise DataSourceError(f"Unsupported file format: {uri}")
        except (json.JSONDecodeError, csv.Error) as e:
            raise DataSourceError(f"Failed to parse {uri}: {e}") from e

    async def read_schema(self, uri: str) -> SchemaDefinition | None:
        """Infer schema from the first record of a file.

        Returns None if the file is empty or unreadable.
        """
        try:
            records = await self.read_records(uri)
        except DataSourceError:
            return None

        if not records:
            return None

        first = records[0]
        fields = tuple(
            FieldDefinition(
                name=key,
                data_type=type(value).__name__ if value is not None else "str",
                nullable=value is None,
            )
            for key, value in first.items()
        )
        return SchemaDefinition(name=uri, version="1.0.0", fields=fields)
