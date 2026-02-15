# tests/unit/test_supervisor_agent.py
"""Tests for supervisor agent."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.application.agents.supervisor import RoutingDecision, SupervisorAgent
from src.application.services.agent_registry import AgentRegistry
from src.domain.ports.agent_port import AgentPort


class FakeWorker(AgentPort):
    def __init__(self, agent_name: str) -> None:
        self._name = agent_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Worker {self._name}"

    async def execute(self, input: str, context: dict[str, Any] | None = None) -> str:
        return f"[{self._name}] done: {input}"


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(FakeWorker("search"))
    reg.register(FakeWorker("code"))
    return reg


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock()


class TestSupervisorAgent:
    async def test_routes_to_correct_agent(
        self, mock_llm: AsyncMock, registry: AgentRegistry
    ) -> None:
        mock_llm.generate_structured.return_value = RoutingDecision(
            agent_name="search", reasoning="User wants to search"
        )
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        result = await supervisor.execute("find documents about AI")
        assert "[search] done:" in result

    async def test_retries_on_invalid_agent(
        self, mock_llm: AsyncMock, registry: AgentRegistry
    ) -> None:
        mock_llm.generate_structured.side_effect = [
            RoutingDecision(agent_name="nonexistent", reasoning="guess"),
            RoutingDecision(agent_name="code", reasoning="fallback"),
        ]
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry, max_iterations=3)
        result = await supervisor.execute("write a function")
        assert "[code] done:" in result

    async def test_max_iterations_exceeded(
        self, mock_llm: AsyncMock, registry: AgentRegistry
    ) -> None:
        mock_llm.generate_structured.return_value = RoutingDecision(
            agent_name="nonexistent", reasoning="always wrong"
        )
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry, max_iterations=2)
        result = await supervisor.execute("impossible task")
        assert "could not route" in result.lower()

    async def test_no_agents_available(self, mock_llm: AsyncMock) -> None:
        empty_registry = AgentRegistry()
        supervisor = SupervisorAgent(llm=mock_llm, registry=empty_registry)
        result = await supervisor.execute("anything")
        assert "No agents available" in result

    async def test_llm_failure(self, mock_llm: AsyncMock, registry: AgentRegistry) -> None:
        mock_llm.generate_structured.side_effect = RuntimeError("LLM down")
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        result = await supervisor.execute("task")
        assert "could not route" in result.lower()

    def test_properties(self, mock_llm: AsyncMock, registry: AgentRegistry) -> None:
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        assert supervisor.name == "supervisor"
        assert "routes" in supervisor.description.lower() or "Routes" in supervisor.description
