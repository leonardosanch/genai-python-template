"""Tests for SupervisorAgent — task routing with bounded loops."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.application.agents.supervisor import RoutingDecision, SupervisorAgent
from src.application.services.agent_registry import AgentRegistry
from src.domain.ports.agent_port import AgentPort
from src.domain.ports.llm_port import LLMPort


@pytest.fixture()
def mock_llm() -> LLMPort:
    return create_autospec(LLMPort, instance=True)


@pytest.fixture()
def registry() -> AgentRegistry:
    return AgentRegistry()


def _make_worker(name: str, description: str, response: str) -> AgentPort:
    worker = create_autospec(AgentPort, instance=True)
    worker.name = name
    worker.description = description
    worker.execute = AsyncMock(return_value=response)
    return worker


class TestSupervisorAgent:
    """Tests for the Supervisor agent."""

    @pytest.mark.asyncio()
    async def test_routes_to_correct_worker(
        self,
        mock_llm: LLMPort,
        registry: AgentRegistry,
    ) -> None:
        worker = _make_worker("analyzer", "Analyzes data", "analysis complete")
        registry.register(worker)

        mock_llm.generate_structured = AsyncMock(
            return_value=RoutingDecision(
                agent_name="analyzer",
                reasoning="Best fit",
            ),
        )

        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        result = await supervisor.execute("Analyze this dataset")

        assert result == "analysis complete"
        worker.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_no_agents_available(
        self,
        mock_llm: LLMPort,
        registry: AgentRegistry,
    ) -> None:
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        result = await supervisor.execute("Do something")
        assert "No agents available" in result

    @pytest.mark.asyncio()
    async def test_retries_on_unknown_agent(
        self,
        mock_llm: LLMPort,
        registry: AgentRegistry,
    ) -> None:
        worker = _make_worker("writer", "Writes docs", "done")
        registry.register(worker)

        # First call returns unknown agent, second returns correct one
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                RoutingDecision(agent_name="nonexistent", reasoning="guess"),
                RoutingDecision(agent_name="writer", reasoning="correct"),
            ],
        )

        supervisor = SupervisorAgent(llm=mock_llm, registry=registry, max_iterations=3)
        result = await supervisor.execute("Write a report")
        assert result == "done"

    @pytest.mark.asyncio()
    async def test_bounded_loop_exhausted(
        self,
        mock_llm: LLMPort,
        registry: AgentRegistry,
    ) -> None:
        worker = _make_worker("real", "Real agent", "ok")
        registry.register(worker)

        # Always returns non-existent agent
        mock_llm.generate_structured = AsyncMock(
            return_value=RoutingDecision(agent_name="ghost", reasoning="wrong"),
        )

        supervisor = SupervisorAgent(llm=mock_llm, registry=registry, max_iterations=2)
        result = await supervisor.execute("task")
        assert "could not route" in result.lower()

    @pytest.mark.asyncio()
    async def test_handles_worker_exception(
        self,
        mock_llm: LLMPort,
        registry: AgentRegistry,
    ) -> None:
        worker = _make_worker("faulty", "Broken agent", "")
        worker.execute = AsyncMock(side_effect=RuntimeError("boom"))
        registry.register(worker)

        mock_llm.generate_structured = AsyncMock(
            return_value=RoutingDecision(agent_name="faulty", reasoning="only option"),
        )

        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        result = await supervisor.execute("task")
        assert "could not route" in result.lower()

    def test_name_and_description(self, mock_llm: LLMPort, registry: AgentRegistry) -> None:
        supervisor = SupervisorAgent(llm=mock_llm, registry=registry)
        assert supervisor.name == "supervisor"
        assert "Routes" in supervisor.description
