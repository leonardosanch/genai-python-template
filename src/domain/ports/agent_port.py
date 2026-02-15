# src/domain/ports/agent_port.py
"""Port for agent abstraction in multi-agent systems."""

from abc import ABC, abstractmethod
from typing import Any


class AgentPort(ABC):
    """Abstract interface for an agent.

    Each agent has a single responsibility and can be
    composed into multi-agent systems.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this agent does — used by supervisors for routing."""
        ...

    @abstractmethod
    async def execute(self, input: str, context: dict[str, Any] | None = None) -> str:
        """Execute the agent's task.

        Args:
            input: The task or query to process.
            context: Optional context dict (conversation history, metadata).

        Returns:
            The agent's response as a string.
        """
        ...

    def get_tools(self) -> list[str]:
        """List tool names available to this agent. Override if agent uses tools."""
        return []
