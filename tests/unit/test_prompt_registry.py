# tests/unit/test_prompt_registry.py
"""Tests for PromptRegistry service."""

import pytest

from src.application.services.prompt_registry import PromptRegistry
from src.domain.value_objects.prompt_template import (
    PromptMetadata,
    PromptStatus,
    PromptTemplate,
)


def _make_template(
    name: str = "qa",
    version: str = "1.0.0",
    status: PromptStatus = PromptStatus.DRAFT,
) -> PromptTemplate:
    return PromptTemplate(
        template="Q: {{ question }}",
        metadata=PromptMetadata(name=name, version=version, author="test", status=status),
    )


class TestPromptRegistry:
    def test_register_and_get(self) -> None:
        registry = PromptRegistry()
        tpl = _make_template()
        registry.register(tpl)
        assert registry.get("qa", "1.0.0") is tpl

    def test_register_duplicate_raises(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_template())

    def test_get_nonexistent_name_raises(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(KeyError, match="No templates"):
            registry.get("missing", "1.0.0")

    def test_get_nonexistent_version_raises(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template())
        with pytest.raises(KeyError, match="Version"):
            registry.get("qa", "9.9.9")

    def test_get_production(self) -> None:
        registry = PromptRegistry()
        draft = _make_template(version="0.1.0", status=PromptStatus.DRAFT)
        prod = _make_template(version="1.0.0", status=PromptStatus.PRODUCTION)
        registry.register(draft)
        registry.register(prod)
        assert registry.get_production("qa") is prod

    def test_get_production_none_available_raises(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template(status=PromptStatus.DRAFT))
        with pytest.raises(KeyError, match="No production"):
            registry.get_production("qa")

    def test_get_production_name_missing_raises(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get_production("missing")

    def test_list_versions(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template(version="1.0.0"))
        registry.register(_make_template(version="2.0.0"))
        versions = registry.list_versions("qa")
        assert set(versions) == {"1.0.0", "2.0.0"}

    def test_list_versions_missing_raises(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.list_versions("missing")

    def test_list_names(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template(name="qa"))
        registry.register(_make_template(name="summary"))
        names = registry.list_names()
        assert set(names) == {"qa", "summary"}

    def test_list_names_empty(self) -> None:
        registry = PromptRegistry()
        assert registry.list_names() == []
