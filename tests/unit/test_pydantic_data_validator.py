"""Unit tests for PydanticDataValidator infrastructure adapter."""

from src.domain.value_objects.schema_definition import FieldDefinition, SchemaDefinition
from src.infrastructure.data.pydantic_data_validator import PydanticDataValidator


class TestPydanticDataValidator:
    async def test_no_schema_all_valid(self) -> None:
        validator = PydanticDataValidator()
        result = await validator.validate([{"id": 1}, {"id": 2}])

        assert result.is_valid is True
        assert result.total_records == 2
        assert result.valid_records == 2

    async def test_empty_records(self) -> None:
        validator = PydanticDataValidator()
        schema = SchemaDefinition(name="test", version="1.0.0", fields=())
        result = await validator.validate([], schema)

        assert result.is_valid is True
        assert result.total_records == 0

    async def test_valid_records_with_schema(self) -> None:
        schema = SchemaDefinition(
            name="users",
            version="1.0.0",
            fields=(
                FieldDefinition(name="id", data_type="int"),
                FieldDefinition(name="name", data_type="str"),
            ),
        )
        records: list[dict[str, object]] = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

        validator = PydanticDataValidator()
        result = await validator.validate(records, schema)

        assert result.is_valid is True
        assert result.valid_records == 2

    async def test_type_mismatch(self) -> None:
        schema = SchemaDefinition(
            name="test",
            version="1.0.0",
            fields=(FieldDefinition(name="age", data_type="int"),),
        )
        records: list[dict[str, object]] = [{"age": "not_a_number"}]

        validator = PydanticDataValidator()
        result = await validator.validate(records, schema)

        assert result.is_valid is False
        assert result.invalid_records == 1
        assert "expected int" in result.errors[0]

    async def test_null_non_nullable(self) -> None:
        schema = SchemaDefinition(
            name="test",
            version="1.0.0",
            fields=(FieldDefinition(name="id", data_type="int", nullable=False),),
        )
        records: list[dict[str, object]] = [{"id": None}]

        validator = PydanticDataValidator()
        result = await validator.validate(records, schema)

        assert result.is_valid is False
        assert "null but not nullable" in result.errors[0]

    async def test_nullable_field_allows_none(self) -> None:
        schema = SchemaDefinition(
            name="test",
            version="1.0.0",
            fields=(FieldDefinition(name="email", data_type="str", nullable=True),),
        )
        records: list[dict[str, object]] = [{"email": None}]

        validator = PydanticDataValidator()
        result = await validator.validate(records, schema)

        assert result.is_valid is True
        assert result.valid_records == 1
