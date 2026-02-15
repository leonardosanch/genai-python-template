"""Schema definition value objects — immutable schema descriptions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldDefinition:
    """Definition of a single field in a schema.

    Attributes:
        name: Field name.
        data_type: Expected data type (str, int, float, bool, etc.).
        nullable: Whether the field accepts None values.
        description: Optional human-readable description.
    """

    name: str
    data_type: str
    nullable: bool = False
    description: str = ""


@dataclass(frozen=True)
class SchemaDefinition:
    """Immutable schema describing the structure of a dataset.

    Attributes:
        name: Schema identifier.
        version: Schema version string.
        fields: Tuple of field definitions (tuple for immutability).
    """

    name: str
    version: str
    fields: tuple[FieldDefinition, ...]
