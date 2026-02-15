"""Unit tests for SchemaDefinition and FieldDefinition value objects."""

from dataclasses import FrozenInstanceError

import pytest

from src.domain.value_objects.schema_definition import FieldDefinition, SchemaDefinition


class TestFieldDefinition:
    def test_defaults(self) -> None:
        field = FieldDefinition(name="id", data_type="int")
        assert field.name == "id"
        assert field.data_type == "int"
        assert field.nullable is False
        assert field.description == ""

    def test_nullable_with_description(self) -> None:
        field = FieldDefinition(
            name="email", data_type="str", nullable=True, description="User email"
        )
        assert field.nullable is True
        assert field.description == "User email"

    def test_frozen(self) -> None:
        field = FieldDefinition(name="id", data_type="int")
        with pytest.raises(FrozenInstanceError):
            field.name = "changed"  # type: ignore[misc]


class TestSchemaDefinition:
    def test_creation(self) -> None:
        fields = (
            FieldDefinition(name="id", data_type="int"),
            FieldDefinition(name="name", data_type="str"),
        )
        schema = SchemaDefinition(name="users", version="1.0.0", fields=fields)
        assert schema.name == "users"
        assert schema.version == "1.0.0"
        assert len(schema.fields) == 2

    def test_frozen(self) -> None:
        schema = SchemaDefinition(name="test", version="1.0.0", fields=())
        with pytest.raises(FrozenInstanceError):
            schema.name = "changed"  # type: ignore[misc]

    def test_empty_fields(self) -> None:
        schema = SchemaDefinition(name="empty", version="0.1.0", fields=())
        assert schema.fields == ()
