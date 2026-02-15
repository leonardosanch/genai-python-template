# Skill: Multi-Agent Systems

## Description
This skill provides patterns and resources for building production-ready multi-agent systems. Use this when coordinating multiple AI agents, implementing agent workflows, or designing agentic architectures.

## Core Concepts

1.  **Stateful Orchestration**: Agents maintain state across interactions using graphs or state machines.
2.  **Coordination Patterns**: Supervisor, sequential, hierarchical, collaborative, swarm patterns.
3.  **Agent Communication**: Sync (shared state), async (message queues), A2A Protocol (HTTP).
4.  **Human-in-the-Loop**: Critical decisions require human approval with graceful degradation.

---

## External Resources

### 📚 Agent Frameworks & Documentation

#### LangGraph (Primary Recommendation)
- **LangGraph Documentation**: [langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
    - *Best for*: Stateful multi-agent workflows, graph-based orchestration
- **LangGraph Tutorials**: [langchain-ai.github.io/langgraph/tutorials/](https://langchain-ai.github.io/langgraph/tutorials/)
    - *Best for*: Agent supervisor, multi-agent collaboration, human-in-the-loop
- **LangGraph How-To Guides**: [langchain-ai.github.io/langgraph/how-tos/](https://langchain-ai.github.io/langgraph/how-tos/)
    - *Best for*: Checkpointing, streaming, subgraphs, error handling
- **LangGraph Conceptual Guide**: [langchain-ai.github.io/langgraph/concepts/](https://langchain-ai.github.io/langgraph/concepts/)
    - *Best for*: State, nodes, edges, persistence, deployment

#### CrewAI
- **CrewAI Documentation**: [docs.crewai.com](https://docs.crewai.com/)
    - *Best for*: Role-based multi-agent teams, task delegation
- **CrewAI Examples**: [github.com/joaomdmoura/crewAI-examples](https://github.com/joaomdmoura/crewAI-examples)
    - *Best for*: Real-world multi-agent applications
- **CrewAI Tools**: [docs.crewai.com/core-concepts/Tools/](https://docs.crewai.com/core-concepts/Tools/)
    - *Best for*: Agent tool integration

#### AutoGen (Microsoft)
- **AutoGen Documentation**: [microsoft.github.io/autogen/](https://microsoft.github.io/autogen/)
    - *Best for*: Conversational multi-agent systems
- **AutoGen Studio**: [microsoft.github.io/autogen/docs/autogen-studio/](https://microsoft.github.io/autogen/docs/autogen-studio/)
    - *Best for*: Visual agent workflow design
- **AutoGen Patterns**: [microsoft.github.io/autogen/docs/Use-Cases/](https://microsoft.github.io/autogen/docs/Use-Cases/)
    - *Best for*: Code generation, research, data analysis

#### Semantic Kernel (Microsoft)
- **Semantic Kernel Documentation**: [learn.microsoft.com/en-us/semantic-kernel/](https://learn.microsoft.com/en-us/semantic-kernel/)
    - *Best for*: Enterprise AI orchestration, .NET/Python/Java
- **Semantic Kernel Agents**: [learn.microsoft.com/en-us/semantic-kernel/agents/](https://learn.microsoft.com/en-us/semantic-kernel/agents/)
    - *Best for*: Agent patterns in enterprise environments

---

### 🔬 Research Papers & Advanced Concepts

#### Agent Foundations
- **ReAct: Synergizing Reasoning and Acting** (Yao et al., 2022)
    - [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
    - *Best for*: Reasoning + action pattern, tool use
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023)
    - [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
    - *Best for*: Self-reflection, learning from mistakes
- **Generative Agents** (Park et al., 2023)
    - [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
    - *Best for*: Believable agent behavior, memory systems

#### Multi-Agent Coordination
- **Communicative Agents for Software Development** (Qian et al., 2023)
    - [arxiv.org/abs/2307.07924](https://arxiv.org/abs/2307.07924)
    - *Best for*: Multi-agent collaboration patterns
- **MetaGPT** (Hong et al., 2023)
    - [arxiv.org/abs/2308.00352](https://arxiv.org/abs/2308.00352)
    - *Best for*: Role-based collaboration, software engineering agents
- **AutoGen: Enabling Next-Gen LLM Applications** (Wu et al., 2023)
    - [arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155)
    - *Best for*: Conversational agent frameworks

---

### 🎯 Coordination Patterns

#### Pattern Descriptions
- **Supervisor Pattern**: One agent routes work to specialized workers
    - *Best for*: Dynamic task routing, specialized expertise
    - *Reference*: [LangGraph Multi-Agent Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- **Sequential Pipeline**: Fixed chain of agents, each processes output of previous
    - *Best for*: Predictable workflows, staged processing
    - *Reference*: [LangGraph Sequential](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
- **Hierarchical**: Multi-level delegation (supervisor → sub-supervisors → workers)
    - *Best for*: Complex tasks requiring decomposition
    - *Reference*: [LangGraph Hierarchical Teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)
- **Collaborative/Debate**: Agents discuss and reach consensus
    - *Best for*: Decision-making, diverse perspectives
    - *Reference*: CrewAI collaborative tasks
- **Swarm**: Dynamic handoff based on capability
    - *Best for*: Adaptive workflows, emergent behavior
    - *Reference*: [OpenAI Swarm](https://github.com/openai/swarm)

---

### 🛠️ Agent Tools & Integrations

#### Tool Frameworks
- **LangChain Tools**: [python.langchain.com/docs/modules/tools/](https://python.langchain.com/docs/modules/tools/)
    - *Best for*: Pre-built tools (search, calculators, APIs)
- **Function Calling**: [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
    - *Best for*: Structured tool invocation
- **MCP (Model Context Protocol)**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)
    - *Best for*: Standardized agent-tool communication

#### Agent-to-Agent Communication
- **A2A Protocol** (Google): [github.com/google/a2a](https://github.com/google/a2a)
    - *Best for*: HTTP-based inter-agent communication
- **Message Queues**: RabbitMQ, Kafka for async agent communication
    - *Best for*: Decoupled, scalable agent systems

---

### 📊 Agent Observability & Monitoring

#### Tracing & Debugging
- **LangSmith**: [docs.smith.langchain.com](https://docs.smith.langchain.com/)
    - *Best for*: Agent tracing, debugging, evaluation
- **Phoenix** (Arize AI): [docs.arize.com/phoenix](https://docs.arize.com/phoenix)
    - *Best for*: LLM observability, agent monitoring
- **Weights & Biases**: [wandb.ai/site/solutions/llmops](https://wandb.ai/site/solutions/llmops)
    - *Best for*: Experiment tracking, agent performance

#### Evaluation
- **AgentBench**: [github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench)
    - *Best for*: Benchmarking agent performance
- **Agent Evaluation Frameworks**: Custom metrics for task completion, efficiency, cost

---

### 🏗️ Production Patterns

#### State Management
- **LangGraph Checkpointing**: [langchain-ai.github.io/langgraph/how-tos/persistence/](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
    - *Best for*: State persistence, recovery, time travel
- **Redis**: For distributed agent state
- **PostgreSQL**: For durable agent state and history

#### Error Handling & Recovery
- **Retry with Exponential Backoff**: [tenacity](https://tenacity.readthedocs.io/)
    - *Best for*: Transient failures
- **Circuit Breaker Pattern**: Prevent cascading failures
- **Fallback Agents**: Simpler agents when primary fails

#### Human-in-the-Loop
- **LangGraph Human-in-the-Loop**: [langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
    - *Best for*: Approval workflows, interrupts
- **Gradio/Streamlit**: For human interaction interfaces

---

### 📖 Books & Courses

#### Books
- **Building Multi-Agent Systems** (Chip Huyen - upcoming)
    - *Best for*: Production multi-agent architectures
- **Autonomous Agents** (various authors)
    - *Best for*: Agent theory and practice

#### Courses
- **DeepLearning.AI - AI Agents in LangGraph**
    - [deeplearning.ai/short-courses/ai-agents-in-langgraph/](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
- **DeepLearning.AI - Multi AI Agent Systems with CrewAI**
    - [deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)

---

## Instructions for the Agent

1.  **When designing multi-agent systems**:
    - Start with single agent, add complexity only when needed
    - Choose coordination pattern based on task structure
    - Reference LangGraph tutorials for implementation
    - Always implement bounded loops (max iterations)

2.  **Coordination Pattern Selection**:
    - **Supervisor**: Dynamic routing, specialized agents
    - **Sequential**: Predictable, staged workflows
    - **Hierarchical**: Complex task decomposition
    - **Collaborative**: Decision-making, consensus
    - **Swarm**: Adaptive, emergent behavior

3.  **State Management**:
    - Use LangGraph checkpointing for persistence
    - Implement state recovery for long-running workflows
    - Store agent history for debugging and auditing

4.  **Error Handling**:
    - Implement retry with exponential backoff (tenacity)
    - Use circuit breakers to prevent cascading failures
    - Provide fallback agents for graceful degradation
    - Always set maximum iteration limits

5.  **Human-in-the-Loop**:
    - Identify critical decision points
    - Implement approval workflows with LangGraph interrupts
    - Provide clear context for human decisions
    - Allow human override at any point

6.  **Observability**:
    - Use LangSmith for tracing and debugging
    - Log all agent decisions and tool calls
    - Track metrics: task completion rate, cost, latency
    - Monitor for infinite loops and stuck states

7.  **Tool Integration**:
    - Use function calling for structured tool invocation
    - Validate tool inputs and outputs
    - Implement timeouts for all tool calls
    - Log every tool execution

8.  **Agent Communication**:
    - Use shared state (LangGraph) for sync communication
    - Use message queues (RabbitMQ, Kafka) for async
    - Consider A2A Protocol for inter-agent HTTP communication
    - Always validate messages between agents

9.  **Production Considerations**:
    - Implement cost tracking per agent
    - Set rate limits and quotas
    - Use semantic caching to reduce redundant LLM calls
    - Deploy with container orchestration (Kubernetes)
    - Implement health checks and monitoring

---

## Code Examples

### Example 1: Supervisor Pattern (see examples/multi_agent_supervisor.py)

Complete implementation available in `docs/skills/examples/multi_agent_supervisor.py`

### Example 2: Agent Error Handling

```python
# src/agents/resilient_agent.py
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.runnables import RunnableConfig

class ResilientAgent:
    """Agent with retry logic and fallbacks."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def execute_with_retry(self, task: str):
        """Execute task with exponential backoff retry."""
        try:
            result = await self.llm.ainvoke(task)
            return result
        except Exception as e:
            self.logger.error(f"Agent failed: {e}")
            raise
    
    async def execute_with_fallback(self, task: str):
        """Execute with fallback to simpler model."""
        try:
            return await self.primary_llm.ainvoke(task)
        except Exception:
            self.logger.warning("Primary LLM failed, using fallback")
            return await self.fallback_llm.ainvoke(task)
```

### Example 3: Cost Tracking for Multi-Agent

```python
# src/agents/cost_tracker.py
from collections import defaultdict
from typing import Dict

class AgentCostTracker:
    """Track costs per agent in multi-agent system."""
    
    def __init__(self):
        self.costs = defaultdict(lambda: {"tokens": 0, "cost_usd": 0, "calls": 0})
    
    def record_call(self, agent_name: str, tokens: int, model: str):
        """Record agent LLM call."""
        cost = self._calculate_cost(tokens, model)
        self.costs[agent_name]["tokens"] += tokens
        self.costs[agent_name]["cost_usd"] += cost
        self.costs[agent_name]["calls"] += 1
    
    def get_report(self) -> Dict:
        """Get cost report by agent."""
        total_cost = sum(a["cost_usd"] for a in self.costs.values())
        return {
            "by_agent": dict(self.costs),
            "total_cost_usd": total_cost,
            "total_tokens": sum(a["tokens"] for a in self.costs.values())
        }
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost based on model pricing."""
        pricing = {
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.002 / 1000,
        }
        return tokens * pricing.get(model, 0.01 / 1000)
```

---

## Debugging Playbook

### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Infinite Loop** | Agents keep passing tasks | Add max_iterations limit |
| **Agent Stuck** | No progress after N steps | Implement timeout per agent |
| **Wrong Agent Selected** | Supervisor routes incorrectly | Improve supervisor prompt, add examples |
| **State Corruption** | Inconsistent state across agents | Use checkpointing, validate state |
| **High Latency** | Slow multi-agent execution | Parallelize independent agents |
| **High Cost** | Excessive LLM calls | Add caching, use cheaper models for routing |

### LangSmith Debugging

```python
# Enable LangSmith tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-debug"

# View traces at: https://smith.langchain.com/
```

---

## Anti-Patterns to Avoid

### ❌ God Agent
**Problem**: Single agent doing everything  
**Solution**: Decompose into specialized agents

### ❌ No Max Iterations
**Problem**: Infinite loops  
**Example**:
```python
# BAD: No limit
while not done:
    result = agent.run(task)
```
**Solution**: Bounded loops
```python
# GOOD: Max iterations
for i in range(max_iterations):
    result = agent.run(task)
    if is_complete(result):
        break
```

### ❌ Synchronous Agent Execution
**Problem**: Agents wait for each other unnecessarily  
**Solution**: Parallel execution with asyncio.gather()

### ❌ No Human-in-the-Loop
**Problem**: Agents make critical decisions autonomously  
**Solution**: Require human approval for high-impact actions

---

## Multi-Agent Checklist

### Design
- [ ] Each agent has single, clear responsibility
- [ ] Coordination pattern selected (supervisor, sequential, hierarchical)
- [ ] Communication protocol defined (shared state, messages)
- [ ] Max iterations configured (prevent infinite loops)
- [ ] Human-in-the-loop for critical decisions

### Implementation
- [ ] LangGraph state machine defined
- [ ] Checkpointing enabled (SqliteSaver, RedisSaver)
- [ ] Error handling with retries (tenacity)
- [ ] Fallback agents configured
- [ ] Timeouts set per agent

### Observability
- [ ] LangSmith tracing enabled
- [ ] Cost tracking per agent
- [ ] Metrics: latency, success rate, iterations
- [ ] Alerts for failures, infinite loops

### Testing
- [ ] Unit tests for each agent
- [ ] Integration tests for workflows
- [ ] Test infinite loop prevention
- [ ] Test error recovery

---

## Additional References

### LangGraph Resources
- **LangGraph Templates**: [github.com/langchain-ai/langgraph/tree/main/examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
    - *Best for*: Production-ready patterns
- **LangGraph Documentation**: [langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
    - *Best for*: API reference

### Multi-Agent Papers
- **AutoGen Paper**: [arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155)
    - *Best for*: Multi-agent conversation frameworks
- **MetaGPT Paper**: [arxiv.org/abs/2308.00352](https://arxiv.org/abs/2308.00352)
    - *Best for*: Role-based collaboration

