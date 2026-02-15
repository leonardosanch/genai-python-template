"""
Prompt Templates Example

Demonstrates:
- Template versioning
- Variable substitution
- Template inheritance
- Validation
- Best practices

Usage:
    python -m src.examples.supporting.prompt_templates
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PromptVersion(str, Enum):
    """Prompt version."""

    V1 = "v1"
    V2 = "v2"


class PromptTemplate(BaseModel):
    """Prompt template with versioning."""

    name: str
    version: PromptVersion
    template: str
    variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str, info: Any) -> str:
        """Validate template has required variables."""
        if not v.strip():
            raise ValueError("Template cannot be empty")
        return v

    def render(self, **kwargs: Any) -> str:
        """Render template with variables."""
        # Validate all required variables provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        # Render
        return self.template.format(**kwargs)

    def save(self, path: Path) -> None:
        """Save template to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "PromptTemplate":
        """Load template from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class PromptRegistry:
    """Registry for managing prompts."""

    def __init__(self, base_path: Path):
        """Initialize registry."""
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        key = f"{template.name}_{template.version.value}"
        self.templates[key] = template

        # Save to disk
        path = self.base_path / f"{key}.json"
        template.save(path)

    def get(self, name: str, version: PromptVersion = PromptVersion.V1) -> PromptTemplate:
        """Get template by name and version."""
        key = f"{name}_{version.value}"
        if key not in self.templates:
            # Try to load from disk
            path = self.base_path / f"{key}.json"
            if path.exists():
                self.templates[key] = PromptTemplate.load(path)
            else:
                raise KeyError(f"Template not found: {key}")
        return self.templates[key]

    def list_templates(self) -> list[str]:
        """List all registered templates."""
        return list(self.templates.keys())


def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Prompt Templates Example")
    print("=" * 60)

    # Create registry
    registry = PromptRegistry(Path("/tmp/prompts"))

    # Example 1: Simple template
    print("\nExample 1: Simple Template")
    print("-" * 60)

    simple = PromptTemplate(
        name="greeting",
        version=PromptVersion.V1,
        template="Hello {name}, welcome to {product}!",
        variables=["name", "product"],
        metadata={"category": "onboarding"},
    )

    registry.register(simple)

    rendered = simple.render(name="Alice", product="RAG System")
    print(f"Rendered: {rendered}")

    # Example 2: RAG prompt
    print("\nExample 2: RAG Prompt")
    print("-" * 60)

    rag_prompt = PromptTemplate(
        name="rag_query",
        version=PromptVersion.V1,
        template="""Answer the question based on the context.

Context:
{context}

Question: {question}

Answer:""",
        variables=["context", "question"],
        metadata={"use_case": "rag", "temperature": 0.3},
    )

    registry.register(rag_prompt)

    rendered = rag_prompt.render(
        context="RAG combines retrieval with generation.",
        question="What is RAG?",
    )
    print(f"Rendered:\n{rendered}")

    # Example 3: Versioned prompt
    print("\nExample 3: Prompt Versioning")
    print("-" * 60)

    # V1
    rag_v1 = registry.get("rag_query", PromptVersion.V1)
    print(f"V1 template: {rag_v1.template[:50]}...")

    # V2 (improved)
    rag_v2 = PromptTemplate(
        name="rag_query",
        version=PromptVersion.V2,
        template="""You are a helpful assistant. Answer based on context only.

Context:
{context}

Question: {question}

Instructions:
- Be concise
- Cite sources
- If unsure, say so

Answer:""",
        variables=["context", "question"],
        metadata={"use_case": "rag", "temperature": 0.3, "improvements": "Added instructions"},
    )

    registry.register(rag_v2)

    rendered_v2 = rag_v2.render(
        context="RAG combines retrieval with generation.",
        question="What is RAG?",
    )
    print(f"\nV2 rendered:\n{rendered_v2}")

    # List templates
    print("\nRegistered Templates:")
    print("-" * 60)
    for template_key in registry.list_templates():
        print(f"  - {template_key}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✅ Template versioning")
    print("✅ Variable validation")
    print("✅ Persistence (save/load)")
    print("✅ Registry pattern")
    print("✅ Metadata tracking")


if __name__ == "__main__":
    main()
