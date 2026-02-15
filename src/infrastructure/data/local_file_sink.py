"""Local file data sink — writes CSV/JSON to the local filesystem."""

import csv
import json
from io import StringIO
from typing import Any

import aiofiles

from src.domain.exceptions import DataSinkError
from src.domain.ports.data_sink_port import DataSinkPort


class LocalFileSink(DataSinkPort):
    """DataSinkPort implementation for local CSV and JSON files."""

    async def write_records(
        self, uri: str, records: list[dict[str, object]], **options: Any
    ) -> int:
        """Write records to a local CSV or JSON file.

        Args:
            uri: File path. Extension determines format (.csv or .json).
            records: Records to write.
            **options: Additional options (unused).

        Returns:
            Number of records written.

        Raises:
            DataSinkError: If writing fails.
        """
        if not records:
            return 0

        lower = uri.lower()
        try:
            if lower.endswith(".json"):
                content = json.dumps(records, default=str, indent=2)
            elif lower.endswith(".csv"):
                output = StringIO()
                writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
                content = output.getvalue()
            else:
                raise DataSinkError(f"Unsupported file format: {uri}")
        except (TypeError, ValueError) as e:
            raise DataSinkError(f"Failed to serialize data for {uri}: {e}") from e

        try:
            async with aiofiles.open(uri, mode="w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as e:
            raise DataSinkError(f"Failed to write to {uri}: {e}") from e

        return len(records)
