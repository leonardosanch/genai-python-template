"""
Supervisor Pattern Example

Demonstrates:
- Supervisor agent coordinating workers
- LangGraph state machine
- Dynamic routing based on task
- Worker specialization
- State management

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.agents.supervisor_pattern
"""

import asyncio
import operator
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# State Definition


class AgentState(TypedDict):
    """State shared across agents."""

    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str
    final_answer: str


# Worker Agents


class ResearchAgent:
    """Agent specialized in research and information gathering."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize research agent."""
        self.llm = llm
        self.name = "Researcher"

    async def execute(self, state: AgentState) -> AgentState:
        """Execute research task."""
        print(f"\n🔍 {self.name} working...")

        # Get the task from messages
        task = state["messages"][-1].content

        # Research prompt
        prompt = f"""You are a research specialist.
Task: {task}

Provide detailed research findings with sources and facts.
Be thorough and cite specific information."""

        messages = [
            SystemMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "messages": [
                HumanMessage(
                    content=f"[{self.name}]: {response.content}",
                    name=self.name,
                )
            ],
            "next_agent": "supervisor",
        }


class WriterAgent:
    """Agent specialized in writing and content creation."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize writer agent."""
        self.llm = llm
        self.name = "Writer"

    async def execute(self, state: AgentState) -> AgentState:
        """Execute writing task."""
        print(f"\n✍️  {self.name} working...")

        task = state["messages"][-1].content

        prompt = f"""You are a professional writer.
Task: {task}

Create well-structured, engaging content.
Use clear language and proper formatting."""

        messages = [SystemMessage(content=prompt)]

        response = await self.llm.ainvoke(messages)

        return {
            "messages": [
                HumanMessage(
                    content=f"[{self.name}]: {response.content}",
                    name=self.name,
                )
            ],
            "next_agent": "supervisor",
        }


class AnalystAgent:
    """Agent specialized in data analysis and insights."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize analyst agent."""
        self.llm = llm
        self.name = "Analyst"

    async def execute(self, state: AgentState) -> AgentState:
        """Execute analysis task."""
        print(f"\n📊 {self.name} working...")

        task = state["messages"][-1].content

        prompt = f"""You are a data analyst.
Task: {task}

Provide analytical insights, patterns, and recommendations.
Use data-driven reasoning."""

        messages = [SystemMessage(content=prompt)]

        response = await self.llm.ainvoke(messages)

        return {
            "messages": [
                HumanMessage(
                    content=f"[{self.name}]: {response.content}",
                    name=self.name,
                )
            ],
            "next_agent": "supervisor",
        }


# Supervisor Agent


class SupervisorAgent:
    """Supervisor that routes tasks to appropriate workers."""

    def __init__(self, llm: ChatOpenAI, workers: list[str]):
        """Initialize supervisor."""
        self.llm = llm
        self.workers = workers
        self.name = "Supervisor"

    async def route(
        self, state: AgentState
    ) -> Literal["researcher", "writer", "analyst", "finish"]:
        """
        Route to next agent or finish.

        Returns:
            Name of next agent or "finish"
        """
        print(f"\n🎯 {self.name} routing...")

        # Get conversation history
        messages = state["messages"]

        # Routing prompt
        workers_str = ", ".join(self.workers)
        prompt = f"""You are a supervisor managing these workers: {workers_str}

Given the conversation, decide:
1. Which worker should act next, OR
2. If the task is complete, respond with "FINISH"

Available workers:
- researcher: For gathering information and facts
- writer: For creating content and documents
- analyst: For data analysis and insights

Respond with ONLY the worker name or "FINISH"."""

        routing_messages = [
            SystemMessage(content=prompt),
            *messages,
        ]

        response = await self.llm.ainvoke(routing_messages)
        decision = response.content.strip().lower()

        print(f"   Decision: {decision}")

        if "finish" in decision:
            return "finish"
        elif "research" in decision:
            return "researcher"
        elif "writ" in decision:
            return "writer"
        elif "analy" in decision:
            return "analyst"
        else:
            # Default to researcher
            return "researcher"


# Graph Construction


class SupervisorSystem:
    """Multi-agent system with supervisor pattern."""

    def __init__(self):
        """Initialize supervisor system."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # Initialize agents
        self.researcher = ResearchAgent(llm)
        self.writer = WriterAgent(llm)
        self.analyst = AnalystAgent(llm)

        workers = ["researcher", "writer", "analyst"]
        self.supervisor = SupervisorAgent(llm, workers)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("analyst", self._analyst_node)

        # Add edges
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("analyst", "supervisor")

        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next_agent"],
            {
                "researcher": "researcher",
                "writer": "writer",
                "analyst": "analyst",
                "finish": END,
            },
        )

        # Set entry point
        workflow.set_entry_point("supervisor")

        return workflow.compile()

    async def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor node."""
        next_agent = await self.supervisor.route(state)
        return {"next_agent": next_agent}

    async def _researcher_node(self, state: AgentState) -> AgentState:
        """Researcher node."""
        return await self.researcher.execute(state)

    async def _writer_node(self, state: AgentState) -> AgentState:
        """Writer node."""
        return await self.writer.execute(state)

    async def _analyst_node(self, state: AgentState) -> AgentState:
        """Analyst node."""
        return await self.analyst.execute(state)

    async def run(self, task: str) -> str:
        """
        Run supervisor system on task.

        Args:
            task: Task description

        Returns:
            Final result
        """
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print(f"{'=' * 60}")

        # Initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "next_agent": "",
            "final_answer": "",
        }

        # Run graph
        final_state = await self.graph.ainvoke(initial_state)

        # Extract final answer from messages
        final_messages = final_state["messages"]
        if final_messages:
            return final_messages[-1].content

        return "No response generated"


async def main() -> None:
    """Run example demonstrations."""
    print("=" * 60)
    print("Supervisor Pattern Example")
    print("=" * 60)

    system = SupervisorSystem()

    # Example 1: Research task
    print("\n\nExample 1: Research Task")
    result1 = await system.run(
        "Research the key benefits of Clean Architecture and provide 3 main points"
    )
    print(f"\n{'=' * 60}")
    print("Final Result:")
    print(f"{'=' * 60}")
    print(result1)

    # Example 2: Writing task
    print("\n\nExample 2: Writing Task")
    result2 = await system.run(
        "Write a brief introduction to RAG (Retrieval-Augmented Generation) "
        "for a technical blog post"
    )
    print(f"\n{'=' * 60}")
    print("Final Result:")
    print(f"{'=' * 60}")
    print(result2)

    # Example 3: Analysis task
    print("\n\nExample 3: Analysis Task")
    result3 = await system.run(
        "Analyze the trade-offs between monolithic and microservices architectures"
    )
    print(f"\n{'=' * 60}")
    print("Final Result:")
    print(f"{'=' * 60}")
    print(result3)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts Demonstrated:")
    print("✅ Supervisor pattern with dynamic routing")
    print("✅ Worker specialization (Research, Writing, Analysis)")
    print("✅ LangGraph state machine")
    print("✅ Conditional edges based on supervisor decisions")
    print("✅ Shared state across agents")


if __name__ == "__main__":
    asyncio.run(main())
