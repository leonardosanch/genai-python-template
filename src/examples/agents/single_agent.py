"""
Single Agent Example

Demonstrates:
- Basic agent with tools
- ReAct (Reasoning + Acting) pattern
- Tool definition and execution
- Error handling
- Structured logging

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.agents.single_agent
"""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

# Tool Definitions
ChatMessages = list[ChatCompletionMessageParam]


class ToolCall(BaseModel):
    """A tool invocation."""

    tool_name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result of tool execution."""

    tool_name: str
    success: bool
    result: Any
    error: str | None = None


class AgentStep(BaseModel):
    """A single agent reasoning step."""

    thought: str = Field(description="Agent's reasoning")
    action: str | None = Field(None, description="Tool to use")
    action_input: dict[str, Any] | None = Field(None, description="Tool arguments")
    observation: str | None = Field(None, description="Tool result")
    final_answer: str | None = Field(None, description="Final answer if done")


# Tools


async def calculator_tool(expression: str) -> dict[str, Any]:
    """
    Evaluate mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")

    Returns:
        Calculation result
    """
    try:
        # Safe eval for simple math (production would use ast.literal_eval or similar)
        result = eval(expression, {"__builtins__": {}}, {})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_current_time_tool() -> dict[str, Any]:
    """
    Get current date and time.

    Returns:
        Current datetime
    """
    now = datetime.now()
    return {
        "success": True,
        "result": {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
        },
    }


async def search_tool(query: str) -> dict[str, Any]:
    """
    Simulate web search (mock implementation).

    Args:
        query: Search query

    Returns:
        Mock search results
    """
    # In production, this would call a real search API
    mock_results = {
        "clean architecture": (
            "Clean Architecture is a software design philosophy by Robert C. Martin "
            "that emphasizes separation of concerns and independence of frameworks."
        ),
        "python async": (
            "Python's asyncio library provides infrastructure for writing concurrent "
            "code using async/await syntax."
        ),
        "rag": (
            "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM "
            "responses by retrieving relevant context from external knowledge bases."
        ),
    }

    result = mock_results.get(query.lower(), f"No specific information found for '{query}'")

    return {"success": True, "result": result}


# Agent


class SingleAgent:
    """
    Single agent with tools using ReAct pattern.

    The agent:
    1. Thinks about what to do
    2. Chooses a tool to use
    3. Observes the result
    4. Repeats until it has the answer
    """

    def __init__(self) -> None:
        """Initialize agent."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # Available tools
        self.tools: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
            "calculator": calculator_tool,
            "get_current_time": get_current_time_tool,
            "search": search_tool,
        }

        # Tool descriptions for the LLM
        self.tool_descriptions = """
Available tools:

1. calculator(expression: str) -> result
   Evaluate mathematical expressions
   Example: calculator("25 * 4")

2. get_current_time() -> datetime info
   Get current date and time

3. search(query: str) -> information
   Search for information on a topic
   Example: search("clean architecture")
"""

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            tool_func = self.tools[tool_name]
            result = await tool_func(**arguments)

            return ToolResult(
                tool_name=tool_name,
                success=result.get("success", True),
                result=result.get("result"),
                error=result.get("error"),
            )

        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    async def run(self, task: str, max_iterations: int = 5) -> str:
        """Run agent on task."""
        print(f"\n{'=' * 60}\nTask: {task}\n{'=' * 60}\n")

        history: ChatMessages = []
        sys_prompt = f"Tools:\n{self.tool_descriptions}\n\nReAct Loop: Thought, Action, Input, Obs."

        for i in range(max_iterations):
            print(f"Step {i + 1}\n{'-' * 60}")
            messages: ChatMessages = [{"role": "system", "content": sys_prompt}]

            if i == 0:
                messages.append({"role": "user", "content": f"Task: {task}"})
            else:
                messages.extend(history)

            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3
            )
            content = response.choices[0].message.content or ""
            print(f"Agent: {content}\n")

            final_answer = await self._process_step(content, history)
            if final_answer:
                print(f"{'=' * 60}\nFinal Answer: {final_answer}\n{'=' * 60}\n")
                return final_answer

        return "Failed to complete task within maximum iterations"

    async def _process_step(self, content: str, history: ChatMessages) -> str | None:
        """Process a single step of the agent loop."""
        action, action_input, final_answer = self._parse_agent_response(content)

        history.append({"role": "assistant", "content": content})

        if final_answer:
            return final_answer

        if action and action_input is not None:
            tool_result = await self.execute_tool(action, action_input)
            observation = (
                f"Tool '{action}' result: {tool_result.result}"
                if tool_result.success
                else f"Tool '{action}' failed: {tool_result.error}"
            )
            print(f"Observation: {observation}\n")
            history.append({"role": "user", "content": f"Observation: {observation}"})

        return None

    def _parse_agent_response(
        self, content: str
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        """Parse the raw agent response text."""
        action = None
        action_input = None
        final_answer = None

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
                if action.lower() == "none":
                    action = None
            elif line.startswith("Action Input:"):
                input_str = line.replace("Action Input:", "").strip()
                if input_str and input_str.lower() != "null":
                    try:
                        action_input = json.loads(input_str)
                    except json.JSONDecodeError:
                        action_input = {"query": input_str}
            elif line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                if final_answer.lower() == "null":
                    final_answer = None

        return action, action_input, final_answer


async def main() -> None:
    """Run example demonstrations."""
    print("=" * 60)
    print("Single Agent Example")
    print("=" * 60)

    agent = SingleAgent()

    # Example 1: Math calculation
    print("\n\nExample 1: Mathematical Reasoning")
    await agent.run("What is 157 multiplied by 23?")

    # Example 2: Current time
    print("\n\nExample 2: Time Query")
    await agent.run("What day of the week is it today?")

    # Example 3: Information search
    print("\n\nExample 3: Information Retrieval")
    await agent.run("What is RAG in the context of AI?")

    # Example 4: Multi-step reasoning
    print("\n\nExample 4: Multi-step Task")
    await agent.run(
        "Calculate 50 * 3, then search for information about that number if it's greater than 100"
    )

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nKey Concepts Demonstrated:")
    print("- ReAct pattern (Reasoning + Acting)")
    print("- Tool definition and execution")
    print("- Iterative problem solving")
    print("- Error handling")
    print("- Bounded iterations (max_iterations)")


if __name__ == "__main__":
    asyncio.run(main())
