# tests/unit/test_cli_commands.py
"""Tests for CLI commands."""

from typer.testing import CliRunner

from src.interfaces.cli.main import app

runner = CliRunner()


class TestIngestCommand:
    def test_dry_run(self) -> None:
        result = runner.invoke(app, ["ingest", "--source-path", "/data", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_missing_source_path(self) -> None:
        result = runner.invoke(app, ["ingest"])
        assert result.exit_code != 0


class TestETLCommand:
    def test_dry_run(self) -> None:
        result = runner.invoke(
            app, ["etl", "--source", "s3://bucket/data", "--sink", "local://out", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_missing_args(self) -> None:
        result = runner.invoke(app, ["etl"])
        assert result.exit_code != 0


class TestHealthCommand:
    def test_health_help(self) -> None:
        result = runner.invoke(app, ["health", "--help"])
        assert result.exit_code == 0
        assert "health" in result.output.lower()


class TestQueryCommand:
    def test_query_help(self) -> None:
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "question" in result.output.lower()


class TestCostCommand:
    def test_cost_help(self) -> None:
        result = runner.invoke(app, ["cost", "--help"])
        assert result.exit_code == 0
        assert "days" in result.output.lower()


class TestValidateCommand:
    def test_validate_help(self) -> None:
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()
