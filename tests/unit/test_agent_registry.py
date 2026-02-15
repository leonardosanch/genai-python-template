# tests/unit/test_agent_registry.py
"""Tests for agent registry."""

from typing import Any

import pytest

from src.application.services.agent_registry import AgentRegistry
from src.domain.ports.agent_port import AgentPort


class MockAgent(AgentPort):
    def __init__(self, agent_name: str, agent_desc: str = "test") -> None:
        self._name = agent_name
        self._desc = agent_desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    async def execute(self, input: str, context: dict[str, Any] | None = None) -> str:
        return f"[{self._name}] processed: {input}"


class TestAgentRegistry:
    def test_register_and_get(self) -> None:
        registry = AgentRegistry()
        agent = MockAgent("search")
        registry.register(agent)
        assert registry.get("search") is agent

    def test_register_duplicate_raises(self) -> None:
        registry = AgentRegistry()
        registry.register(MockAgent("search"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockAgent("search"))

    def test_get_missing_raises(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="No agent"):
            registry.get("nonexistent")

    def test_list_agents(self) -> None:
        registry = AgentRegistry()
        registry.register(MockAgent("search", "Search the web"))
        registry.register(MockAgent("code", "Write code"))
        agents = registry.list_agents()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert names == {"search", "code"}

    def test_list_agents_empty(self) -> None:
        registry = AgentRegistry()
        assert registry.list_agents() == []
