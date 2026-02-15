# Skill: Context Engineering for Multi-Agent Systems

## Description
This skill covers the discipline of structuring, managing, and governing the information that LLMs use to reason, decide, and generate. Based on the **Context Engine** architecture — a transparent, glass-box system built on multi-agent collaboration, retrieval, and policy-driven safeguards.

Reference: *Context Engineering for Multi-Agent Systems* — Denis Rothman (Packt, 2025)

## Executive Summary

**Critical context engineering rules:**
- Always operate at Context Level 3+ (goal-oriented) for production — Level 5 (semantic blueprints) for critical workflows
- Use context chaining for multi-step tasks — decompose into focused steps where each output feeds the next (never monolithic prompts)
- Dual RAG architecture mandatory — separate procedural (HOW: style guides, templates) from factual (WHAT: knowledge, data) retrieval
- Every agent gets MINIMUM necessary context — use Summarizer agent pattern for proactive context reduction (prevent token overflow)
- Glass-box over black-box — Execution Tracer logs every agent step for auditability (MCP structured messages for inter-agent communication)

**Read full skill when:** Designing multi-agent systems, implementing Context Engine architecture, managing context across agent workflows, preventing token overflow, or building auditable glass-box AI systems.

---

## Versiones de Dependencias

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| pinecone-client | >= 3.0.0 | API v3 con namespaces |
| openai | >= 1.0.0 | Client async oficial |
| tiktoken | >= 0.5.0 | Token counting |

### ⚠️ Nota sobre MCP
El protocolo MCP está en evolución. Verificar compatibilidad con la versión del SDK de MCP (`mcp >= 1.0.0`) antes de implementar comunicación inter-agente.

---

## Core Concepts

### 1. Context Engineering (not Prompt Engineering)
Context engineering is the shift from *asking* an LLM to *directing* it. Instead of hoping for good outputs, you engineer the informational environment the model operates within.

**Five Levels of Context Maturity:**

| Level | Name | Description |
|-------|------|-------------|
| 1 | Zero Context | Basic prompt, no background. LLM guesses from training data. |
| 2 | Linear Context | Added factual thread. Improves accuracy but no style/purpose. |
| 3 | Goal-Oriented Context | First *true* context level. Clear goal makes responses intentional. |
| 4 | Role-Based Context | Explicit roles (characters, relationships) add narrative intelligence. |
| 5 | Semantic Blueprint | Full structured plan with semantic roles. Reliable, repeatable engineering. |

**Rule:** Always operate at Level 3+ for production systems. Level 5 (semantic blueprints) for critical workflows.

### 2. Semantic Blueprint
A structured JSON/dict that defines:
- `scene_goal` — What the output must achieve
- `participants` — Entities with roles (Agent, Patient, Source)
- `action_to_complete` — Predicate + agent + patient
- Argument modifiers (temporal, location, manner)

Rooted in **Semantic Role Labeling (SRL)** (Tesnière → Fillmore → PropBank): Who did what to whom, when, where, and why.

**Core SRL roles:**
- **Predicate** — The central action (verb)
- **Agent (ARG0)** — Entity performing the action
- **Patient (ARG1)** — Entity affected by the action
- **Recipient (ARG2)** — Entity receiving the result
- **Modifiers (ARGM-)** — Temporal (TMP), Location (LOC), Manner (MNR)

### 3. Context Chaining
Multi-step workflows where the output of one LLM call becomes the input for the next. Transforms complex tasks into controlled, step-by-step dialogues.

**Advantages over monolithic prompts:**
- Precision: Guide the AI's thought process at each stage
- Debugging: Isolate which step produced poor results
- Building on insight: Each step refines and builds upon previous outputs

### 4. The Context Engine Architecture

**Glass-box system** — every decision is traceable, every reasoning step is visible.

**Core components:**
- **Planner** — Receives user goal, creates execution plan
- **Executor** — Runs the plan step-by-step through specialist agents
- **Execution Tracer** — Logs every step for auditability and debugging

**Specialist agents:**
- **Context Librarian** — Retrieves procedural context (style guides, templates) via RAG
- **Researcher** — Retrieves factual knowledge via RAG with source citations
- **Writer** — Generates final content from research + context
- **Summarizer** — Proactive context reduction to manage token overhead

**Agent Registry** — Central registry mapping agent names to functions. Enables dynamic agent discovery and modularity.

### 5. Dual RAG Architecture
Separates two types of retrieval:
- **Procedural RAG (Context Library)** — HOW to do things (style guides, templates, blueprints)
- **Factual RAG (Knowledge Base)** — WHAT the facts are (documents, data, research)

Both are stored in vector databases (e.g., Pinecone) with separate namespaces.

### 6. MCP for Agent Communication
All inter-agent communication uses **Model Context Protocol (MCP)** structured messages:

```python
{
    "protocol_version": "1.0",
    "sender": "ResearcherAgent",
    "content": "...",
    "metadata": {"source": "...", "task_id": "..."}
}
```

**Key MCP principles for MAS:**
- JSON-RPC 2.0 format
- UTF-8 encoded
- Transport: STDIO (same machine) or HTTP (distributed)
- Versioning and security headers required

### 7. Context Reduction (Summarizer Agent)
Proactive context management to prevent token overflow and cost explosion:
- `count_tokens` utility measures context size
- Summarizer agent compresses inter-agent payloads
- **Micro-context engineering** — each agent receives only the minimum context it needs
- Foundation for cost management in production

### 8. High-Fidelity RAG
Every retrieved fact carries source metadata for citation-backed reasoning:
- Source document name, section, page
- Confidence score from vector similarity
- Enables verifiable, auditable outputs

**Defense layers:**
- `helper_sanitize_input` — Prompt injection defense
- Data poisoning detection at ingestion time (validate sources before embedding)
- Input validation before every agent execution

### 8b. Latency Budgeting and Stochasticity
Production context engines must account for LLM operational realities:
- **Latency is inherent**: Multi-agent workflows multiply LLM call latency. Budget accordingly per step.
- **Stochasticity is expected**: LLM outputs vary between runs. Design for validation, retries, and fallbacks rather than assuming determinism.
- **The deliberate pace**: A reasoning engine is intentionally slower than a single LLM call — each step adds traceability and correctness at the cost of speed.

### 9. Production Safeguards

**Two-Stage Content Moderation:**
1. Pre-processing moderation (before agent execution)
2. Post-processing moderation (before returning to user)

**Policy-Driven Meta-Controller:**
- AI systems must continuously adapt to reality
- Automated contextual judgment has limits
- Policy is the ultimate context
- Human-in-the-loop for critical decisions

**Five Principles:**
1. AI systems must continuously adapt to reality
2. Limits of automated contextual judgment
3. New engineer's mindset (glass-box thinking)
4. Policy as the ultimate context
5. Architectural solution (not just code-level fixes)

### 10. Hardening for Production
Transform prototype to production-ready:
- **Modularization** — helpers.py, agents.py, registry.py, engine.py
- **Dependency injection** — Agents receive dependencies, not create them
- **Structured logging** — Production-level traceability
- **Proactive context management** — Token budgets per agent
- **Backward compatibility** — New capabilities don't break existing workflows. Validate with test cases from prior chapters.

### 11. Domain Independence
The Context Engine architecture is designed to be **domain-agnostic**:
- Core logic (Planner, Executor, Tracer, agents) remains unchanged across domains
- Only the **knowledge base** and **control deck** templates change per domain
- Proven in the book across: legal compliance, strategic marketing, NASA research
- Enables modular reuse: swap knowledge bases without touching engine code

### 12. Production API and Deployment
Enterprise deployment of the Context Engine:
- **Production API**: FastAPI orchestration layer exposing engine capabilities
- **Async execution**: Task queues (Celery/RQ) for long-running agent workflows
- **Centralized logging and observability**: Structured logs, OpenTelemetry traces per agent step
- **Containerization**: Docker + Kubernetes for scalable deployment
- **Secrets management**: Environment-based configuration, never hardcoded
- **Cost management**: Summarizer agent + token budgets as operational controls

---

## Decision Trees

### When to Use Context Engineering vs. Simple Prompts

```
What is your task?
|-- Single-shot Q&A, no precision needed
|   +-> Simple prompt (Level 1-2) is sufficient
|-- Needs consistent, goal-aligned output
|   +-> Goal-oriented context (Level 3+)
|-- Multi-step workflow with multiple concerns
|   +-> Context chaining with semantic blueprints (Level 5)
|-- Multi-agent system with retrieval
|   +-> Full Context Engine architecture
+-- Enterprise deployment with compliance
    +-> Context Engine + moderation + policy controller
```

### Choosing Agent Architecture

```
How complex is the task?
|-- Single concern, single output
|   +-> Single agent with good context
|-- Research + generation (two concerns)
|   +-> Researcher + Writer agents with Orchestrator
|-- Multiple knowledge sources + style requirements
|   +-> Dual RAG (Context Library + Knowledge Base) + specialist agents
|-- Enterprise with compliance, moderation, cost control
|   +-> Full Context Engine with Summarizer + Moderation + Policy
+-- Cross-domain reuse needed
    +-> Context Engine with swappable knowledge bases and control decks
```

---

## Anti-Patterns to Avoid

### Monolithic Prompts
**Problem:** Single massive prompt for complex multi-step tasks. LLM loses focus, produces muddled results.
**Solution:** Context chaining — decompose into focused steps where each output feeds the next.

### Black-Box Agents
**Problem:** No visibility into agent reasoning. Can't debug, audit, or trust outputs.
**Solution:** Glass-box architecture with Execution Tracer logging every step.

### Unbounded Context
**Problem:** Passing entire conversation history to every agent. Token explosion, cost, degraded reasoning.
**Solution:** Summarizer agent for proactive context reduction. Each agent gets minimum necessary context.

### No Input Sanitization
**Problem:** User input passed directly to agents without validation. Prompt injection risk.
**Solution:** `helper_sanitize_input` before every agent execution. Two-stage moderation.

### Hardcoded Agent Logic
**Problem:** Agent behavior embedded in code, not configurable.
**Solution:** Agent Registry + policy-driven control decks. Swap knowledge bases without changing core logic.

### Ignoring Latency and Stochasticity
**Problem:** Treating LLM responses as instant and deterministic.
**Solution:** Latency budgets, retry with backoff, accept stochastic nature, validate outputs.

---

## Architecture Reference

### Module Structure (Context Engine)
```
commons/
  helpers.py      # LLM calls, token counting, sanitization, embedding
  agents.py       # Specialist agents (Librarian, Researcher, Writer, Summarizer)
  registry.py     # AgentRegistry — maps names to agent functions
  engine.py       # ContextEngine (Planner, Executor, Tracer)
  utils.py        # Moderation, policy enforcement
```

### Context Engine Workflow
```
Phase 0: Data Ingestion Pipeline
  -> Chunk documents -> Embed -> Upsert to vector DB (context library + knowledge base)

Phase 1: Initiation
  -> User provides goal -> Engine initializes trace

Phase 2: Planning
  -> Planner agent creates execution plan (ordered list of agent steps)

Phase 3: Execution Loop
  -> For each step: select agent -> execute -> log trace -> pass output to next

Phase 4: Finalization
  -> Assemble final output -> Return with full trace
```

### Control Deck Templates
Reusable templates for different use cases:
1. **High-Fidelity RAG** — Research with source citations
2. **Context Reduction** — Summarized, cost-efficient workflows
3. **Grounded Reasoning** — Preventing hallucination with strict retrieval

---

## Code Examples

### Example 1: Semantic Blueprint Builder

```python
# src/application/context/semantic_blueprint.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SemanticRole(str, Enum):
    AGENT = "agent"       # ARG0 — entity performing the action
    PATIENT = "patient"   # ARG1 — entity affected by the action
    RECIPIENT = "recipient"  # ARG2 — entity receiving the result
    SOURCE = "source"     # source of information
    INSTRUMENT = "instrument"  # tool or method used


class Participant(BaseModel):
    name: str
    role: SemanticRole
    description: str = ""


class ArgumentModifier(BaseModel):
    temporal: str | None = None    # ARGM-TMP: when
    location: str | None = None    # ARGM-LOC: where
    manner: str | None = None      # ARGM-MNR: how
    purpose: str | None = None     # ARGM-PRP: why
    condition: str | None = None   # ARGM-ADV: under what conditions


class SemanticBlueprint(BaseModel):
    """Level 5 context structure based on SRL (Semantic Role Labeling)."""

    scene_goal: str = Field(..., description="What the output must achieve")
    participants: list[Participant] = Field(default_factory=list)
    predicate: str = Field(..., description="Central action verb")
    action_to_complete: str = Field(..., description="Full action: predicate + agent + patient")
    modifiers: ArgumentModifier = Field(default_factory=ArgumentModifier)
    constraints: list[str] = Field(default_factory=list)
    output_format: str = "text"

    def to_system_prompt(self) -> str:
        """Convert blueprint to structured system prompt."""
        parts = [
            f"## Goal\n{self.scene_goal}",
            f"\n## Action\n{self.action_to_complete}",
        ]
        if self.participants:
            roles = "\n".join(
                f"- **{p.name}** ({p.role.value}): {p.description}"
                for p in self.participants
            )
            parts.append(f"\n## Participants\n{roles}")
        if self.modifiers.temporal:
            parts.append(f"\n## Temporal Context\n{self.modifiers.temporal}")
        if self.modifiers.manner:
            parts.append(f"\n## Approach\n{self.modifiers.manner}")
        if self.constraints:
            rules = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"\n## Constraints\n{rules}")
        parts.append(f"\n## Output Format\n{self.output_format}")
        return "\n".join(parts)


# Usage
blueprint = SemanticBlueprint(
    scene_goal="Generate a technical architecture review",
    predicate="review",
    action_to_complete="Review the microservices architecture for scalability issues",
    participants=[
        Participant(name="Architect Agent", role=SemanticRole.AGENT, description="Performs the review"),
        Participant(name="System Design", role=SemanticRole.PATIENT, description="Architecture under review"),
        Participant(name="Engineering Team", role=SemanticRole.RECIPIENT, description="Receives the report"),
    ],
    modifiers=ArgumentModifier(
        manner="Focus on horizontal scaling, data consistency, and failure modes",
        purpose="Identify bottlenecks before production launch",
    ),
    constraints=[
        "Must reference concrete code patterns, not generic advice",
        "Cite specific services and their interactions",
        "Include severity ratings (Critical/High/Medium/Low)",
    ],
    output_format="Markdown report with sections: Summary, Findings, Recommendations",
)
```

### Example 2: Context Engine with Execution Tracer

```python
# src/application/context/engine.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


class AgentResult(BaseModel):
    agent_name: str
    output: str
    tokens_used: int = 0
    duration_ms: float = 0


class TraceStep(BaseModel):
    step_id: str
    agent_name: str
    input_summary: str
    output_summary: str
    tokens_used: int
    duration_ms: float
    timestamp: float


class ExecutionTrace(BaseModel):
    trace_id: str
    goal: str
    steps: list[TraceStep] = []
    total_tokens: int = 0
    total_duration_ms: float = 0

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)
        self.total_tokens += step.tokens_used
        self.total_duration_ms += step.duration_ms


class ContextAgent(Protocol):
    """Protocol for all context engine agents."""

    @property
    def name(self) -> str: ...

    async def execute(self, context: dict[str, Any]) -> AgentResult: ...


class AgentRegistry:
    """Central registry mapping agent names to implementations."""

    def __init__(self) -> None:
        self._agents: dict[str, ContextAgent] = {}

    def register(self, agent: ContextAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> ContextAgent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not registered. Available: {list(self._agents)}")
        return self._agents[name]

    @property
    def available_agents(self) -> list[str]:
        return list(self._agents.keys())


class ContextEngine:
    """Glass-box context engine with full execution tracing."""

    def __init__(self, registry: AgentRegistry, token_budget: int = 8000) -> None:
        self._registry = registry
        self._token_budget = token_budget

    async def execute_plan(
        self,
        goal: str,
        plan: list[dict[str, Any]],
    ) -> tuple[str, ExecutionTrace]:
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            goal=goal,
        )
        context: dict[str, Any] = {"goal": goal}

        for step_def in plan:
            agent_name = step_def["agent"]
            agent = self._registry.get(agent_name)

            start = time.perf_counter()
            result = await agent.execute(context)
            duration_ms = (time.perf_counter() - start) * 1000

            # Check token budget
            if trace.total_tokens + result.tokens_used > self._token_budget:
                summarizer = self._registry.get("summarizer")
                context["previous_output"] = result.output
                summary_result = await summarizer.execute(context)
                result = summary_result

            trace_step = TraceStep(
                step_id=str(uuid.uuid4()),
                agent_name=agent_name,
                input_summary=str(context.get("goal", ""))[:200],
                output_summary=result.output[:200],
                tokens_used=result.tokens_used,
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
            trace.add_step(trace_step)

            # Chain output to next step
            context[f"{agent_name}_output"] = result.output
            context["previous_output"] = result.output

        final_output = context.get("previous_output", "")
        return str(final_output), trace
```

### Example 3: Dual RAG Retriever (Procedural + Factual)

```python
# src/infrastructure/rag/dual_retriever.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalType(str, Enum):
    PROCEDURAL = "procedural"  # HOW — style guides, templates, blueprints
    FACTUAL = "factual"        # WHAT — documents, data, research


class RetrievedChunk(BaseModel):
    content: str
    source: str
    retrieval_type: RetrievalType
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DualRAGRetriever:
    """Separate retrieval for procedural (HOW) and factual (WHAT) context.

    Uses separate namespaces/collections to prevent cross-contamination
    between style/process knowledge and factual knowledge.
    """

    def __init__(
        self,
        vector_store: Any,  # VectorStorePort
        procedural_namespace: str = "context_library",
        factual_namespace: str = "knowledge_base",
        top_k: int = 5,
    ) -> None:
        self._store = vector_store
        self._procedural_ns = procedural_namespace
        self._factual_ns = factual_namespace
        self._top_k = top_k

    async def retrieve_procedural(self, query: str) -> list[RetrievedChunk]:
        """Retrieve HOW-TO context: style guides, templates, process docs."""
        results = await self._store.similarity_search(
            query=query,
            namespace=self._procedural_ns,
            top_k=self._top_k,
        )
        return [
            RetrievedChunk(
                content=r["content"],
                source=r["metadata"].get("source", "unknown"),
                retrieval_type=RetrievalType.PROCEDURAL,
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in results
        ]

    async def retrieve_factual(self, query: str) -> list[RetrievedChunk]:
        """Retrieve WHAT context: documents, data, research findings."""
        results = await self._store.similarity_search(
            query=query,
            namespace=self._factual_ns,
            top_k=self._top_k,
        )
        return [
            RetrievedChunk(
                content=r["content"],
                source=r["metadata"].get("source", "unknown"),
                retrieval_type=RetrievalType.FACTUAL,
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in results
        ]

    async def retrieve_both(self, query: str) -> dict[str, list[RetrievedChunk]]:
        """Retrieve from both namespaces in parallel."""
        import asyncio

        procedural, factual = await asyncio.gather(
            self.retrieve_procedural(query),
            self.retrieve_factual(query),
        )
        return {
            "procedural": procedural,
            "factual": factual,
        }
```

### Example 4: Context Chaining Pipeline

```python
# src/application/context/chaining.py
from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChainStep(BaseModel):
    name: str
    description: str
    input_keys: list[str] = []
    output_key: str = "output"


class ContextChain:
    """Multi-step context chaining where each step's output feeds the next.

    Replaces monolithic prompts with focused, debuggable steps.
    Each step has a clear goal and produces a specific output.
    """

    def __init__(self) -> None:
        self._steps: list[tuple[ChainStep, Callable[..., Awaitable[str]]]] = []

    def add_step(
        self,
        step: ChainStep,
        executor: Callable[..., Awaitable[str]],
    ) -> "ContextChain":
        self._steps.append((step, executor))
        return self

    async def run(self, initial_context: dict[str, Any]) -> dict[str, Any]:
        """Execute chain step by step, building context progressively."""
        context = dict(initial_context)

        for i, (step, executor) in enumerate(self._steps):
            logger.info(f"Chain step {i + 1}/{len(self._steps)}: {step.name}")

            # Extract only needed inputs for this step
            step_input = {k: context[k] for k in step.input_keys if k in context}

            try:
                result = await executor(**step_input)
                context[step.output_key] = result
                logger.info(
                    f"Step '{step.name}' completed — output key: {step.output_key} "
                    f"({len(result)} chars)"
                )
            except Exception as e:
                logger.error(f"Step '{step.name}' failed: {e}")
                context[f"{step.output_key}_error"] = str(e)
                raise

        return context


# Usage: Research → Analyze → Write pipeline
async def build_research_chain(llm_client: Any) -> ContextChain:
    chain = ContextChain()

    async def research(goal: str) -> str:
        return await llm_client.generate(
            f"Research the following topic thoroughly:\n{goal}\n\n"
            "Return key findings with sources."
        )

    async def analyze(goal: str, research_output: str) -> str:
        return await llm_client.generate(
            f"Goal: {goal}\n\nResearch findings:\n{research_output}\n\n"
            "Analyze these findings. Identify patterns, contradictions, and gaps."
        )

    async def write(goal: str, analysis_output: str) -> str:
        return await llm_client.generate(
            f"Goal: {goal}\n\nAnalysis:\n{analysis_output}\n\n"
            "Write a clear, structured report based on this analysis."
        )

    chain.add_step(
        ChainStep(name="Research", input_keys=["goal"], output_key="research_output"),
        research,
    ).add_step(
        ChainStep(name="Analyze", input_keys=["goal", "research_output"], output_key="analysis_output"),
        analyze,
    ).add_step(
        ChainStep(name="Write", input_keys=["goal", "analysis_output"], output_key="final_report"),
        write,
    )
    return chain
```

### Example 5: Token Budget Manager

```python
# src/application/context/token_budget.py
from __future__ import annotations

import tiktoken
from pydantic import BaseModel, Field


class TokenBudget(BaseModel):
    """Per-agent token budget with tracking and alerts."""

    max_tokens: int = Field(default=4000, description="Maximum tokens for this agent")
    used_tokens: int = 0
    warning_threshold: float = Field(default=0.8, description="Warn at this usage ratio")

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def usage_ratio(self) -> float:
        return self.used_tokens / self.max_tokens if self.max_tokens > 0 else 1.0

    @property
    def needs_summarization(self) -> bool:
        return self.usage_ratio >= self.warning_threshold

    def consume(self, tokens: int) -> None:
        self.used_tokens += tokens

    def would_exceed(self, tokens: int) -> bool:
        return (self.used_tokens + tokens) > self.max_tokens


class TokenCounter:
    """Count tokens using tiktoken for accurate budgeting."""

    def __init__(self, model: str = "gpt-4") -> None:
        try:
            self._encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoder.decode(tokens[:max_tokens])


class BudgetManager:
    """Manage token budgets across all agents in a context engine run."""

    def __init__(self, total_budget: int = 16000) -> None:
        self._total_budget = total_budget
        self._counter = TokenCounter()
        self._agent_budgets: dict[str, TokenBudget] = {}

    def allocate(self, agent_name: str, max_tokens: int) -> TokenBudget:
        budget = TokenBudget(max_tokens=max_tokens)
        self._agent_budgets[agent_name] = budget
        return budget

    def consume(self, agent_name: str, text: str) -> int:
        tokens = self._counter.count(text)
        if agent_name in self._agent_budgets:
            self._agent_budgets[agent_name].consume(tokens)
        return tokens

    @property
    def total_used(self) -> int:
        return sum(b.used_tokens for b in self._agent_budgets.values())

    @property
    def agents_needing_summarization(self) -> list[str]:
        return [
            name for name, budget in self._agent_budgets.items()
            if budget.needs_summarization
        ]
```

### Example 6: Specialist Agent Implementation

```python
# src/application/context/agents.py
from __future__ import annotations

from typing import Any

from src.application.context.engine import AgentResult, ContextAgent


class ResearcherAgent:
    """Retrieves factual knowledge via RAG with source citations."""

    def __init__(self, llm_client: Any, retriever: Any) -> None:
        self._llm = llm_client
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "researcher"

    async def execute(self, context: dict[str, Any]) -> AgentResult:
        goal = context.get("goal", "")

        # Retrieve factual knowledge
        chunks = await self._retriever.retrieve_factual(goal)
        sources = "\n".join(
            f"[{c.source}]: {c.content}" for c in chunks
        )

        prompt = (
            f"Research goal: {goal}\n\n"
            f"Available sources:\n{sources}\n\n"
            "Synthesize findings with inline citations [source_name]. "
            "Only use information from the provided sources."
        )
        response = await self._llm.generate(prompt)
        return AgentResult(
            agent_name=self.name,
            output=response,
            tokens_used=len(response.split()) * 2,  # Approximate
        )


class SummarizerAgent:
    """Proactive context reduction to manage token overhead."""

    def __init__(self, llm_client: Any, max_output_tokens: int = 500) -> None:
        self._llm = llm_client
        self._max_output_tokens = max_output_tokens

    @property
    def name(self) -> str:
        return "summarizer"

    async def execute(self, context: dict[str, Any]) -> AgentResult:
        text = context.get("previous_output", "")

        prompt = (
            "Summarize the following content preserving all key facts, "
            "decisions, and actionable items. Remove redundancy and filler.\n\n"
            f"Content:\n{text}\n\n"
            f"Maximum summary length: {self._max_output_tokens} tokens."
        )
        response = await self._llm.generate(prompt)
        return AgentResult(
            agent_name=self.name,
            output=response,
            tokens_used=len(response.split()) * 2,
        )


class ContextLibrarianAgent:
    """Retrieves procedural context: style guides, templates, blueprints."""

    def __init__(self, llm_client: Any, retriever: Any) -> None:
        self._llm = llm_client
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "context_librarian"

    async def execute(self, context: dict[str, Any]) -> AgentResult:
        goal = context.get("goal", "")

        # Retrieve procedural (HOW-TO) knowledge
        chunks = await self._retriever.retrieve_procedural(goal)
        guidelines = "\n".join(
            f"- [{c.source}]: {c.content}" for c in chunks
        )

        return AgentResult(
            agent_name=self.name,
            output=f"## Applicable Guidelines\n{guidelines}",
            tokens_used=len(guidelines.split()) * 2,
        )
```

---

## External Resources

- **Book**: *Context Engineering for Multi-Agent Systems* — Denis Rothman (Packt, Nov 2025)
- **Book Repository**: [github.com/Denis2054/Context-Engineering-for-Multi-Agent-Systems](https://github.com/Denis2054/Context-Engineering-for-Multi-Agent-Systems)
- **MCP Specification**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)
- **Microsoft A2A on MCP**: [developer.microsoft.com/blog/can-you-build-agent2agent-communication-on-mcp-yes](https://developer.microsoft.com/blog/can-you-build-agent2agent-communication-on-mcp-yes)
- **Pinecone (Vector DB)**: [pinecone.io](https://www.pinecone.io/)
- **OpenAI API**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **SRL / PropBank**: [aclanthology.org/J05-1004.pdf](https://aclanthology.org/J05-1004.pdf)
- **Chain-of-Thought Prompting**: [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- **Fillmore's Case Grammar**: [linguistics.berkeley.edu](https://linguistics.berkeley.edu/~syntax-circle/syntax-group/spr08/fillmore.pdf)
- **PropBank (SRL corpus)**: [aclanthology.org/J05-1004.pdf](https://aclanthology.org/J05-1004.pdf)

---

## Instructions for the Agent

1. **Context Level Enforcement**: ALWAYS operate at Context Level 3+ (goal-oriented) for production code. Use Level 5 (semantic blueprints) for critical workflows involving multiple agents or complex reasoning.

2. **Context Chaining**: Decompose multi-step tasks into focused steps where each output feeds the next. NEVER create monolithic prompts for complex workflows. Each step should have a clear goal and produce a specific output.

3. **Dual RAG Architecture**: Implement separate retrieval for:
   - **Procedural context** (HOW): style guides, templates, blueprints → Context Library namespace
   - **Factual knowledge** (WHAT): documents, data, research → Knowledge Base namespace
   - Store in separate Pinecone namespaces or collections

4. **Minimum Necessary Context**: Each agent receives ONLY the context it needs. Use Summarizer agent pattern to proactively reduce context size before passing to next agent. Prevents token overflow and cost explosion.

5. **Glass-Box Architecture**: Implement Execution Tracer that logs every agent step with:
   - Agent name and role
   - Input context (summarized if large)
   - Output produced
   - Timestamp and duration
   - Any errors or warnings

6. **MCP for Inter-Agent Communication**: All agent-to-agent messages use Model Context Protocol structured format:
   ```python
   {
       "protocol_version": "1.0",
       "sender": "AgentName",
       "content": "...",
       "metadata": {"source": "...", "task_id": "..."}
   }
   ```

7. **Agent Registry**: Maintain central registry mapping agent names to functions. Enables dynamic agent discovery and modularity. Never hardcode agent references.

8. **Semantic Blueprints**: For complex tasks, create structured plans with:
   - `scene_goal` — What must be achieved
   - `participants` — Entities with SRL roles (Agent, Patient, Recipient)
   - `action_to_complete` — Predicate + arguments
   - Modifiers (temporal, location, manner)

9. **Input Sanitization**: ALWAYS sanitize user input with `helper_sanitize_input` before agent execution. Implement two-stage moderation (pre-processing and post-processing).

10. **Token Budgeting**: Set explicit token budgets per agent. If agent exceeds budget, invoke Summarizer before continuing. Monitor token usage in production.

11. **Production Hardening**: Structure code as:
    - `helpers.py` — LLM calls, token counting, sanitization
    - `agents.py` — Specialist agents
    - `registry.py` — Agent registry
    - `engine.py` — Planner, Executor, Tracer
    - Use dependency injection, not global state

12. **Domain Independence**: Keep core engine logic unchanged across domains. Only swap knowledge bases and control deck templates. Validate backward compatibility when adding new capabilities.

---

---

## Notas de Implementación Segura

### 🛡️ Validación de Esquemas (Pydantic)
Toda la comunicación entre agentes vía MCP debe ser validada contra modelos de Pydantic. Esto evita que un agente malicioso o alucinante propague datos corruptos por la cadena.

### 🛡️ Límites de Tokens (Context Budget)
Implementar límites estrictos de tokens por cada paso de la cadena. Si un agente excede su presupuesto de contexto, el Summarizer debe intervenir antes de continuar con el siguiente paso.

### 🛡️ Sanitización de Retrieval
Los fragmentos recuperados vía RAG deben ser tratados como **untrusted code**. Nunca inyectar fragmentos directamente en scripts ejecutables sin una capa de sanitización previa.
