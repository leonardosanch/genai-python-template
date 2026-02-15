"""Tests for LocalFileSink — CSV/JSON file writer."""

import json

import pytest

from src.domain.exceptions import DataSinkError
from src.infrastructure.data.local_file_sink import LocalFileSink


@pytest.fixture()
def sink() -> LocalFileSink:
    return LocalFileSink()


class TestLocalFileSink:
    """Tests for the local file data sink."""

    @pytest.mark.asyncio()
    async def test_write_json(self, sink: LocalFileSink, tmp_path: object) -> None:
        path = f"{tmp_path}/output.json"
        records = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        count = await sink.write_records(path, records)

        assert count == 2
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"

    @pytest.mark.asyncio()
    async def test_write_csv(self, sink: LocalFileSink, tmp_path: object) -> None:
        path = f"{tmp_path}/output.csv"
        records = [{"id": 1, "value": "x"}, {"id": 2, "value": "y"}]
        count = await sink.write_records(path, records)

        assert count == 2
        with open(path) as f:
            content = f.read()
        assert "id,value" in content
        assert "1,x" in content

    @pytest.mark.asyncio()
    async def test_empty_records_returns_zero(self, sink: LocalFileSink) -> None:
        count = await sink.write_records("any.json", [])
        assert count == 0

    @pytest.mark.asyncio()
    async def test_unsupported_format_raises(self, sink: LocalFileSink, tmp_path: object) -> None:
        path = f"{tmp_path}/output.parquet"
        with pytest.raises(DataSinkError, match="Unsupported file format"):
            await sink.write_records(path, [{"a": 1}])

    @pytest.mark.asyncio()
    async def test_write_to_bad_path_raises(self, sink: LocalFileSink) -> None:
        with pytest.raises(DataSinkError, match="Failed to write"):
            await sink.write_records("/nonexistent/dir/file.json", [{"a": 1}])
