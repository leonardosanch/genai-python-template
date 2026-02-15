# src/application/agents/supervisor.py
"""Supervisor agent — routes tasks to specialized worker agents."""

from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.application.services.agent_registry import AgentRegistry
from src.domain.ports.agent_port import AgentPort
from src.domain.ports.llm_port import LLMPort

logger = structlog.get_logger(__name__)


class RoutingDecision(BaseModel):
    """Structured output for the supervisor's routing decision."""

    agent_name: str = Field(description="Name of the agent to delegate to")
    reasoning: str = Field(description="Why this agent was chosen")


class SupervisorAgent(AgentPort):
    """Supervisor that uses an LLM to classify tasks and delegates to workers.

    Bounded loop: max_iterations prevents infinite delegation cycles.
    """

    ROUTING_PROMPT = (
        "You are a task router. Given the following task and available agents, "
        "decide which agent should handle it.\n\n"
        "Available agents:\n{agents}\n\n"
        "Task: {task}\n\n"
        "Choose the most appropriate agent."
    )

    def __init__(
        self,
        llm: LLMPort,
        registry: AgentRegistry,
        max_iterations: int = 5,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "supervisor"

    @property
    def description(self) -> str:
        return "Routes tasks to specialized worker agents"

    async def execute(self, input: str, context: dict[str, Any] | None = None) -> str:
        """Route the task to the best worker and return its result."""
        agents_info = self._registry.list_agents()
        if not agents_info:
            return "No agents available to handle this task."

        agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in agents_info)

        prompt = self.ROUTING_PROMPT.format(agents=agents_desc, task=input)

        for iteration in range(self._max_iterations):
            try:
                decision = await self._llm.generate_structured(prompt, RoutingDecision)
                worker = self._registry.get(decision.agent_name)

                logger.info(
                    "supervisor_delegating",
                    agent=decision.agent_name,
                    reasoning=decision.reasoning,
                    iteration=iteration + 1,
                )

                result = await worker.execute(input, context)
                return result

            except KeyError:
                logger.warning(
                    "supervisor_agent_not_found",
                    attempted=decision.agent_name,
                    iteration=iteration + 1,
                )
                prompt += f"\n\nNote: Agent '{decision.agent_name}' does not exist. Choose another."
                continue
            except Exception:
                logger.error("supervisor_error", iteration=iteration + 1)
                break

        return "Supervisor could not route the task after maximum attempts."
