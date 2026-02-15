# src/domain/value_objects/prompt_template.py
"""Prompt template value object with Jinja2 rendering and versioning."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
from pydantic import BaseModel


class PromptStatus(Enum):
    """Lifecycle status of a prompt template."""

    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class PromptMetadata:
    """Immutable metadata for a prompt template."""

    name: str
    version: str
    author: str
    status: PromptStatus
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt template with Jinja2 rendering.

    Attributes:
        template: Jinja2 template string.
        metadata: Versioning and lifecycle metadata.
        input_schema: Optional Pydantic model to validate render inputs.
    """

    template: str
    metadata: PromptMetadata
    input_schema: type[BaseModel] | None = None

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables.

        Raises:
            ValueError: If input validation fails or template has undefined variables.
            jinja2.TemplateSyntaxError: If template syntax is invalid.
        """
        if self.input_schema is not None:
            try:
                self.input_schema(**kwargs)
            except Exception as e:
                raise ValueError(f"Input validation failed: {e}") from e

        env = Environment(undefined=StrictUndefined, autoescape=False)
        try:
            jinja_template = env.from_string(self.template)
            return jinja_template.render(**kwargs)
        except UndefinedError as e:
            raise ValueError(f"Missing template variable: {e}") from e
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error: {e}") from e
