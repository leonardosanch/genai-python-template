"""
Sequential Pipeline Example

Demonstrates:
- Fixed sequence of agents
- State passing between agents
- Pipeline composition
- Error recovery
- Checkpointing

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.agents.sequential_pipeline
"""

import asyncio
import os
from typing import TypedDict

from langchain_core.messages import SystemMessage  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore

# State Definition


class PipelineState(TypedDict):
    """State passed through pipeline."""

    input: str
    research_output: str
    outline_output: str
    draft_output: str
    final_output: str
    error: str | None


# Pipeline Agents


class ResearchAgent:
    """First agent: Research the topic."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Initialize agent."""
        self.llm = llm
        self.name = "Researcher"

    async def execute(self, state: PipelineState) -> PipelineState:
        """Execute research."""
        print(f"\n📚 Step 1: {self.name}")
        print(f"   Input: {state['input'][:50]}...")

        try:
            prompt = f"""Research the following topic and provide key facts:

Topic: {state["input"]}

Provide 3-5 key facts or points."""

            messages = [SystemMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)

            output = response.content
            print(f"   Output: {len(output)} chars")

            return {
                "input": state["input"],
                "research_output": str(output),
                "outline_output": "",
                "draft_output": "",
                "final_output": "",
                "error": None,
            }

        except Exception as e:
            return {
                "input": state["input"],
                "research_output": "",
                "outline_output": "",
                "draft_output": "",
                "final_output": "",
                "error": f"Research failed: {str(e)}",
            }


class OutlineAgent:
    """Second agent: Create outline from research."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Initialize agent."""
        self.llm = llm
        self.name = "Outliner"

    async def execute(self, state: PipelineState) -> PipelineState:
        """Create outline."""
        print(f"\n📋 Step 2: {self.name}")

        if state.get("error"):
            print("   Skipped due to previous error")
            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": "",
                "draft_output": "",
                "final_output": "",
                "error": state["error"],
            }

        try:
            research = state["research_output"]

            prompt = f"""Based on this research, create a structured outline:

Research:
{research}

Create a clear outline with main sections and sub-points."""

            messages = [SystemMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)

            output = response.content
            print(f"   Output: {len(output)} chars")

            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": str(output),
                "draft_output": "",
                "final_output": "",
                "error": None,
            }

        except Exception as e:
            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": "",
                "draft_output": "",
                "final_output": "",
                "error": f"Outline failed: {str(e)}",
            }


class DraftAgent:
    """Third agent: Write draft from outline."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Initialize agent."""
        self.llm = llm
        self.name = "Drafter"

    async def execute(self, state: PipelineState) -> PipelineState:
        """Write draft."""
        print(f"\n✍️  Step 3: {self.name}")

        if state.get("error"):
            print("   Skipped due to previous error")
            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": state["outline_output"],
                "draft_output": "",
                "final_output": "",
                "error": state["error"],
            }

        try:
            outline = state["outline_output"]

            prompt = f"""Write a draft based on this outline:

Outline:
{outline}

Write clear, concise content for each section."""

            messages = [SystemMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)

            output = response.content
            print(f"   Output: {len(output)} chars")

            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": state["outline_output"],
                "draft_output": str(output),
                "final_output": "",
                "error": None,
            }

        except Exception as e:
            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": state["outline_output"],
                "draft_output": "",
                "final_output": "",
                "error": f"Draft failed: {str(e)}",
            }


class EditorAgent:
    """Fourth agent: Edit and finalize."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Initialize agent."""
        self.llm = llm
        self.name = "Editor"

    async def execute(self, state: PipelineState) -> PipelineState:
        """Edit draft."""
        print(f"\n🔍 Step 4: {self.name}")

        if state.get("error"):
            print("   Skipped due to previous error")
            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": state["outline_output"],
                "draft_output": state["draft_output"],
                "final_output": "",
                "error": state["error"],
            }

        try:
            draft = state["draft_output"]

            prompt = f"""Edit and improve this draft:

Draft:
{draft}

Improve clarity, fix errors, and polish the content."""

            messages = [SystemMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)

            output = response.content
            print(f"   Output: {len(output)} chars")

            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": state["outline_output"],
                "draft_output": state["draft_output"],
                "final_output": str(output),
                "error": None,
            }

        except Exception as e:
            return {
                "input": state["input"],
                "research_output": state["research_output"],
                "outline_output": state["outline_output"],
                "draft_output": state["draft_output"],
                "final_output": "",
                "error": f"Editing failed: {str(e)}",
            }


# Pipeline System


class SequentialPipeline:
    """Sequential agent pipeline."""

    def __init__(self) -> None:
        """Initialize pipeline."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # Initialize agents
        self.researcher = ResearchAgent(llm)
        self.outliner = OutlineAgent(llm)
        self.drafter = DraftAgent(llm)
        self.editor = EditorAgent(llm)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build sequential pipeline graph."""
        workflow = StateGraph(PipelineState)

        # Add nodes in sequence
        workflow.add_node("research", self._research_node)
        workflow.add_node("outline", self._outline_node)
        workflow.add_node("draft", self._draft_node)
        workflow.add_node("edit", self._edit_node)

        # Add sequential edges
        workflow.add_edge("research", "outline")
        workflow.add_edge("outline", "draft")
        workflow.add_edge("draft", "edit")
        workflow.add_edge("edit", END)

        # Set entry point
        workflow.set_entry_point("research")

        return workflow.compile()

    async def _research_node(self, state: PipelineState) -> PipelineState:
        """Research node."""
        return await self.researcher.execute(state)

    async def _outline_node(self, state: PipelineState) -> PipelineState:
        """Outline node."""
        return await self.outliner.execute(state)

    async def _draft_node(self, state: PipelineState) -> PipelineState:
        """Draft node."""
        return await self.drafter.execute(state)

    async def _edit_node(self, state: PipelineState) -> PipelineState:
        """Edit node."""
        return await self.editor.execute(state)

    async def run(self, topic: str) -> str:
        """
        Run pipeline on topic.

        Args:
            topic: Topic to process

        Returns:
            Final output or error message
        """
        print(f"\n{'=' * 60}")
        print(f"Pipeline: {topic}")
        print(f"{'=' * 60}")

        # Initial state
        initial_state: PipelineState = {
            "input": topic,
            "research_output": "",
            "outline_output": "",
            "draft_output": "",
            "final_output": "",
            "error": None,
        }

        # Run pipeline
        final_state = await self.graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            return f"Error: {final_state['error']}"

        return str(final_state.get("final_output", "No output generated"))


async def main() -> None:
    """Run example demonstrations."""
    print("=" * 60)
    print("Sequential Pipeline Example")
    print("=" * 60)

    pipeline = SequentialPipeline()

    # Example 1: Technical topic
    print("\n\nExample 1: Technical Article")
    result1 = await pipeline.run("The benefits of async programming in Python")
    print(f"\n{'=' * 60}")
    print("Final Output:")
    print(f"{'=' * 60}")
    print(result1[:500] + "..." if len(result1) > 500 else result1)

    # Example 2: Architecture topic
    print("\n\nExample 2: Architecture Guide")
    result2 = await pipeline.run("Introduction to microservices architecture")
    print(f"\n{'=' * 60}")
    print("Final Output:")
    print(f"{'=' * 60}")
    print(result2[:500] + "..." if len(result2) > 500 else result2)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Concepts Demonstrated:")
    print("✅ Fixed sequential pipeline")
    print("✅ State passing between agents")
    print("✅ Error propagation")
    print("✅ Multi-step processing")
    print("✅ Research → Outline → Draft → Edit flow")


if __name__ == "__main__":
    asyncio.run(main())
