"""
Multi-Agent Supervisor Pattern Example

This example demonstrates:
- LangGraph supervisor pattern
- Specialized worker agents
- State checkpointing
- Error handling and retries
"""

import asyncio
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph


# Define state
class AgentState(TypedDict):
    messages: list
    next_agent: str
    task: str
    result: str


# Worker agents
async def research_agent(state: AgentState) -> AgentState:
    """Research agent - gathers information."""
    llm = ChatOpenAI(model="gpt-4")

    prompt = f"Research the following topic: {state['task']}"
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    state["messages"].append(AIMessage(content=f"Research: {response.content}"))
    state["result"] = response.content
    return state


async def writer_agent(state: AgentState) -> AgentState:
    """Writer agent - creates content."""
    llm = ChatOpenAI(model="gpt-4")

    prompt = f"Write a summary based on: {state['result']}"
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    state["messages"].append(AIMessage(content=f"Writing: {response.content}"))
    state["result"] = response.content
    return state


async def reviewer_agent(state: AgentState) -> AgentState:
    """Reviewer agent - validates quality."""
    llm = ChatOpenAI(model="gpt-4")

    prompt = f"Review and improve: {state['result']}"
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    state["messages"].append(AIMessage(content=f"Review: {response.content}"))
    state["result"] = response.content
    return state


# Supervisor
async def supervisor(state: AgentState) -> AgentState:
    """Supervisor agent - routes to appropriate worker."""
    llm = ChatOpenAI(model="gpt-4")

    prompt = f"""
    Given the task: {state["task"]}
    Current progress: {len(state["messages"])} steps completed

    Choose the next agent: research, writer, reviewer, or FINISH
    """

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    next_agent = response.content.strip().lower()

    state["next_agent"] = next_agent
    return state


def route_agent(state: AgentState) -> str:
    """Route to next agent based on supervisor decision."""
    next_agent = state.get("next_agent", "").lower()

    if "finish" in next_agent:
        return END
    elif "research" in next_agent:
        return "research"
    elif "writer" in next_agent:
        return "writer"
    elif "reviewer" in next_agent:
        return "reviewer"
    else:
        return END


async def create_supervisor_graph():
    """Create multi-agent graph with supervisor."""
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("research", research_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("reviewer", reviewer_agent)

    # Add edges
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("reviewer", "supervisor")

    # Add conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "research": "research",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END,
        },
    )

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add checkpointing
    memory = SqliteSaver.from_conn_string(":memory:")
    graph = workflow.compile(checkpointer=memory)

    return graph


async def main():
    """Run multi-agent supervisor example."""
    print("Creating supervisor graph...")
    graph = await create_supervisor_graph()

    # Initial state
    initial_state = {
        "messages": [],
        "next_agent": "",
        "task": "Explain Clean Architecture principles",
        "result": "",
    }

    # Run workflow
    print("\n=== Running Multi-Agent Workflow ===")
    config = {"configurable": {"thread_id": "example-1"}}

    async for event in graph.astream(initial_state, config):
        for node_name, node_state in event.items():
            print(f"\n--- {node_name.upper()} ---")
            if "messages" in node_state and node_state["messages"]:
                print(node_state["messages"][-1].content[:200] + "...")

    # Get final state
    final_state = await graph.aget_state(config)
    print("\n=== Final Result ===")
    print(final_state.values["result"])


if __name__ == "__main__":
    asyncio.run(main())
