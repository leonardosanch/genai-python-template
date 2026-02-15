"""Pydantic-based data validator — validates records against schema definitions."""

from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.value_objects.data_quality_result import DataQualityResult
from src.domain.value_objects.schema_definition import SchemaDefinition

# Mapping from schema data_type strings to Python types
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


class PydanticDataValidator(DataValidatorPort):
    """DataValidatorPort implementation using pure Python type checks.

    Validates records against a SchemaDefinition by checking
    field presence and type compatibility.
    """

    async def validate(
        self,
        records: list[dict[str, object]],
        schema: SchemaDefinition | None = None,
    ) -> DataQualityResult:
        """Validate records against an optional schema.

        If no schema is provided, all records are considered valid.
        """
        total = len(records)

        if not schema or total == 0:
            return DataQualityResult(
                is_valid=True,
                total_records=total,
                valid_records=total,
                invalid_records=0,
            )

        errors: list[str] = []
        invalid_count = 0

        for i, record in enumerate(records):
            record_valid = True
            for field_def in schema.fields:
                value = record.get(field_def.name)

                # Check missing non-nullable fields
                if value is None and not field_def.nullable:
                    errors.append(f"Record {i}: field '{field_def.name}' is null but not nullable")
                    record_valid = False
                    continue

                # Check type if value is present
                if value is not None:
                    expected_type = _TYPE_MAP.get(field_def.data_type)
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(
                            f"Record {i}: field '{field_def.name}' expected "
                            f"{field_def.data_type}, got {type(value).__name__}"
                        )
                        record_valid = False

            if not record_valid:
                invalid_count += 1

        valid_count = total - invalid_count
        return DataQualityResult(
            is_valid=invalid_count == 0,
            total_records=total,
            valid_records=valid_count,
            invalid_records=invalid_count,
            errors=errors,
        )
