---
description: Scaffold LangGraph agent (state, nodes, graph, bounded loops)
---

1. Ask the user for: agent name, purpose, and list of tools/capabilities.
2. Read [docs/skills/multi_agent_systems.md](file:///home/leo/templates/genai-python-template/docs/skills/multi_agent_systems.md) for agent patterns.
3. Read [docs/skills/genai_rag.md](file:///home/leo/templates/genai-python-template/docs/skills/genai_rag.md) if the agent involves retrieval.
4. Create the following files:

   **a. Agent State**
   - Path: `src/domain/agents/<agent_name>/state.py`
   - `TypedDict` defining the agent's state schema.
   - Include `messages`, iteration counter, and any domain-specific fields.

   **b. Agent Nodes**
   - Path: `src/domain/agents/<agent_name>/nodes.py`
   - One function per node (e.g., `process`, `decide`, `respond`).
   - Each node receives state, returns partial state update.
   - Pure functions where possible — side effects via injected ports.

   **c. Agent Graph**
   - Path: `src/application/agents/<agent_name>/graph.py`
   - `StateGraph` construction with nodes and edges.
   - Conditional edges for routing decisions.
   - **Bounded loop**: max iterations with explicit exit condition.
   - Compile with checkpointer for state persistence.

   **d. Agent Entry Point**
   - Path: `src/application/agents/<agent_name>/__init__.py`
   - Factory function `create_<agent_name>_agent()` that wires dependencies and returns compiled graph.

5. Apply these rules:
   - Every loop MUST have a maximum iteration count (default: 10).
   - LLM calls abstracted behind interfaces (never import provider directly in nodes).
   - Error handling with fallback behavior (graceful degradation).
   - All LLM calls are async.
6. Output all files ready to use with `uv run`.
