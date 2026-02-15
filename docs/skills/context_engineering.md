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
