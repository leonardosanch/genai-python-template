"""Unit tests for LocalFileSource infrastructure adapter."""

import json

import pytest

from src.domain.exceptions import DataSourceError
from src.infrastructure.data.local_file_source import LocalFileSource


class TestLocalFileSourceCSV:
    async def test_read_csv(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "data.csv"
        p.write_text("name,age\nAlice,30\nBob,25\n")

        source = LocalFileSource()
        records = await source.read_records(str(p))

        assert len(records) == 2
        assert records[0]["name"] == "Alice"
        assert records[0]["age"] == "30"  # CSV returns strings

    async def test_read_csv_empty(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "empty.csv"
        p.write_text("name,age\n")

        source = LocalFileSource()
        records = await source.read_records(str(p))
        assert records == []


class TestLocalFileSourceJSON:
    async def test_read_json(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "data.json"
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        p.write_text(json.dumps(data))

        source = LocalFileSource()
        records = await source.read_records(str(p))

        assert len(records) == 2
        assert records[0]["name"] == "Alice"
        assert records[0]["age"] == 30


class TestLocalFileSourceErrors:
    async def test_file_not_found(self) -> None:
        source = LocalFileSource()
        with pytest.raises(DataSourceError, match="File not found"):
            await source.read_records("/nonexistent/file.csv")

    async def test_unsupported_format(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "data.xml"
        p.write_text("<data/>")

        source = LocalFileSource()
        with pytest.raises(DataSourceError, match="Unsupported"):
            await source.read_records(str(p))


class TestLocalFileSourceSchema:
    async def test_infer_schema(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "data.json"
        p.write_text(json.dumps([{"id": 1, "name": "Alice"}]))

        source = LocalFileSource()
        schema = await source.read_schema(str(p))

        assert schema is not None
        assert len(schema.fields) == 2
        field_names = {f.name for f in schema.fields}
        assert "id" in field_names
        assert "name" in field_names

    async def test_schema_returns_none_for_missing(self) -> None:
        source = LocalFileSource()
        schema = await source.read_schema("/nonexistent/file.csv")
        assert schema is None
