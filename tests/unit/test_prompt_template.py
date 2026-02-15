# tests/unit/test_prompt_template.py
"""Tests for PromptTemplate value object."""

import pytest
from pydantic import BaseModel

from src.domain.value_objects.prompt_template import (
    PromptMetadata,
    PromptStatus,
    PromptTemplate,
)


def _make_metadata(**overrides: object) -> PromptMetadata:
    defaults = {
        "name": "test_prompt",
        "version": "1.0.0",
        "author": "test",
        "status": PromptStatus.PRODUCTION,
    }
    defaults.update(overrides)
    return PromptMetadata(**defaults)  # type: ignore[arg-type]


class TestPromptMetadata:
    def test_frozen(self) -> None:
        meta = _make_metadata()
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]

    def test_tags_default_empty(self) -> None:
        meta = _make_metadata()
        assert meta.tags == ()


class TestPromptTemplate:
    def test_render_simple(self) -> None:
        tpl = PromptTemplate(
            template="Hello, {{ name }}!",
            metadata=_make_metadata(),
        )
        assert tpl.render(name="World") == "Hello, World!"

    def test_render_missing_variable_raises(self) -> None:
        tpl = PromptTemplate(
            template="Hello, {{ name }}!",
            metadata=_make_metadata(),
        )
        with pytest.raises(ValueError, match="Missing template variable"):
            tpl.render()

    def test_render_with_schema_valid(self) -> None:
        class InputSchema(BaseModel):
            query: str
            top_k: int

        tpl = PromptTemplate(
            template="Search: {{ query }} (top {{ top_k }})",
            metadata=_make_metadata(),
            input_schema=InputSchema,
        )
        result = tpl.render(query="test", top_k=5)
        assert result == "Search: test (top 5)"

    def test_render_with_schema_invalid_raises(self) -> None:
        class InputSchema(BaseModel):
            query: str

        tpl = PromptTemplate(
            template="Search: {{ query }}",
            metadata=_make_metadata(),
            input_schema=InputSchema,
        )
        with pytest.raises(ValueError, match="Input validation failed"):
            tpl.render(wrong_field="oops")

    def test_frozen(self) -> None:
        tpl = PromptTemplate(template="test", metadata=_make_metadata())
        with pytest.raises(AttributeError):
            tpl.template = "changed"  # type: ignore[misc]

    def test_multiline_template(self) -> None:
        tpl = PromptTemplate(
            template="System: {{ role }}\nUser: {{ question }}",
            metadata=_make_metadata(),
        )
        result = tpl.render(role="assistant", question="How?")
        assert "System: assistant" in result
        assert "User: How?" in result

    def test_conditional_template(self) -> None:
        tpl = PromptTemplate(
            template="{% if verbose %}Detailed: {% endif %}{{ answer }}",
            metadata=_make_metadata(),
        )
        assert tpl.render(verbose=True, answer="yes") == "Detailed: yes"
        assert tpl.render(verbose=False, answer="yes") == "yes"
