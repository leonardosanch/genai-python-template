"""Prompt value object — versioned, immutable prompt template."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """Immutable prompt template with explicit version.

    Prompts are versioned artifacts. A change in prompt text
    is a change in system behavior and must be tracked.
    """

    template: str
    version: str

    def render(self, **kwargs: str) -> str:
        """Render the prompt template with provided variables."""
        return self.template.format(**kwargs)
