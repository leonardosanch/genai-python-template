# src/application/services/prompt_registry.py
"""In-memory prompt template registry with versioning."""

from src.domain.value_objects.prompt_template import PromptStatus, PromptTemplate


class PromptRegistry:
    """Registry for managing versioned prompt templates.

    Stores templates by name and version. Supports retrieving
    the production version of a named prompt.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template.

        Raises:
            ValueError: If a template with the same name and version already exists.
        """
        name = template.metadata.name
        version = template.metadata.version
        if name not in self._store:
            self._store[name] = {}
        if version in self._store[name]:
            raise ValueError(
                f"Template '{name}' version '{version}' already registered. "
                "Use a new version instead."
            )
        self._store[name][version] = template

    def get(self, name: str, version: str) -> PromptTemplate:
        """Get a specific version of a template.

        Raises:
            KeyError: If name or version not found.
        """
        if name not in self._store:
            raise KeyError(f"No templates registered with name '{name}'")
        if version not in self._store[name]:
            raise KeyError(f"Version '{version}' not found for template '{name}'")
        return self._store[name][version]

    def get_production(self, name: str) -> PromptTemplate:
        """Get the production version of a named template.

        Raises:
            KeyError: If no production version exists.
        """
        if name not in self._store:
            raise KeyError(f"No templates registered with name '{name}'")

        for template in self._store[name].values():
            if template.metadata.status == PromptStatus.PRODUCTION:
                return template

        raise KeyError(f"No production version found for template '{name}'")

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a named template.

        Raises:
            KeyError: If name not found.
        """
        if name not in self._store:
            raise KeyError(f"No templates registered with name '{name}'")
        return list(self._store[name].keys())

    def list_names(self) -> list[str]:
        """List all registered template names."""
        return list(self._store.keys())
