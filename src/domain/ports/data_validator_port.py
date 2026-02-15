"""Port for data validation. Infrastructure adapters implement this interface."""

from abc import ABC, abstractmethod

from src.domain.value_objects.data_quality_result import DataQualityResult
from src.domain.value_objects.schema_definition import SchemaDefinition


class DataValidatorPort(ABC):
    """Abstract interface for validating data records.

    Implementations handle schema validation, type checks, constraints, etc.
    Domain and application layers depend only on this interface.
    """

    @abstractmethod
    async def validate(
        self,
        records: list[dict[str, object]],
        schema: SchemaDefinition | None = None,
    ) -> DataQualityResult:
        """Validate records against an optional schema.

        Args:
            records: List of records to validate.
            schema: Optional schema to validate against.

        Returns:
            DataQualityResult with validation outcome.
        """
        ...
