# src/application/services/agent_registry.py
"""Registry for managing available agents."""

from src.domain.ports.agent_port import AgentPort


class AgentRegistry:
    """Registry for discovering and retrieving agents by name."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentPort] = {}

    def register(self, agent: AgentPort) -> None:
        """Register an agent.

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentPort:
        """Get an agent by name.

        Raises:
            KeyError: If no agent with that name is registered.
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered with name '{name}'")
        return self._agents[name]

    def list_agents(self) -> list[dict[str, str]]:
        """List all registered agents with their descriptions."""
        return [
            {"name": agent.name, "description": agent.description}
            for agent in self._agents.values()
        ]
