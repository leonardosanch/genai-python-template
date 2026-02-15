"""Unit tests for DataQualityResult value object."""

from dataclasses import FrozenInstanceError

import pytest

from src.domain.value_objects.data_quality_result import DataQualityResult


class TestDataQualityResultCreation:
    def test_valid_result(self) -> None:
        result = DataQualityResult(
            is_valid=True,
            total_records=10,
            valid_records=10,
            invalid_records=0,
        )
        assert result.is_valid is True
        assert result.total_records == 10
        assert result.valid_records == 10
        assert result.invalid_records == 0
        assert result.errors == []
        assert result.metadata == {}

    def test_invalid_result_with_errors(self) -> None:
        result = DataQualityResult(
            is_valid=False,
            total_records=5,
            valid_records=3,
            invalid_records=2,
            errors=["bad row 1", "bad row 2"],
        )
        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_with_metadata(self) -> None:
        result = DataQualityResult(
            is_valid=True,
            total_records=1,
            valid_records=1,
            invalid_records=0,
            metadata={"schema": "v1"},
        )
        assert result.metadata["schema"] == "v1"


class TestDataQualityResultImmutability:
    def test_frozen(self) -> None:
        result = DataQualityResult(
            is_valid=True, total_records=0, valid_records=0, invalid_records=0
        )
        with pytest.raises(FrozenInstanceError):
            result.is_valid = False  # type: ignore[misc]


class TestDataQualityResultValidation:
    def test_negative_total_records(self) -> None:
        with pytest.raises(ValueError, match="total_records"):
            DataQualityResult(is_valid=True, total_records=-1, valid_records=0, invalid_records=0)

    def test_negative_valid_records(self) -> None:
        with pytest.raises(ValueError, match="valid_records"):
            DataQualityResult(is_valid=True, total_records=0, valid_records=-1, invalid_records=0)

    def test_negative_invalid_records(self) -> None:
        with pytest.raises(ValueError, match="invalid_records"):
            DataQualityResult(is_valid=True, total_records=0, valid_records=0, invalid_records=-1)

    def test_counts_must_sum_to_total(self) -> None:
        with pytest.raises(ValueError, match="must equal total_records"):
            DataQualityResult(is_valid=True, total_records=10, valid_records=5, invalid_records=3)
