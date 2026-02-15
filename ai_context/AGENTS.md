# Agentes

## Patrones de Agentes

### Single Agent

Un agente con un objetivo claro, un set de tools y un prompt de sistema.

```python
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[search_tool, calculator_tool],
    state_modifier="You are a research assistant...",
)

result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
```

### Multi-Agent

Múltiples agentes coordinados para resolver tareas complejas.

**Patrones de coordinación:**

| Patrón | Uso | Ejemplo |
|--------|-----|---------|
| Supervisor | Un agente coordina a otros | Manager que delega a especialistas |
| Sequential | Cadena de agentes | Pipeline de procesamiento |
| Hierarchical | Árbol de agentes | Supervisor → sub-supervisors → workers |
| Collaborative | Agentes peer-to-peer | Debate o consenso |
| Swarm | Handoff dinámico entre agentes | Routing por capacidad |

---

## Patrones Multi-Agent en Detalle

### Supervisor Pattern

Un agente supervisor decide qué worker ejecutar en cada paso.

```python
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, SystemMessage

class SupervisorState(MessagesState):
    next_agent: str = ""

SUPERVISOR_PROMPT = """You are a supervisor managing these workers: {workers}.
Given the conversation, decide which worker should act next.
Respond with the worker name, or 'FINISH' if the task is complete."""

async def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor decides which agent to invoke next."""
    decision = await llm.generate_structured(
        prompt=SUPERVISOR_PROMPT.format(workers=", ".join(WORKERS)),
        schema=RouterDecision,
    )
    state.next_agent = decision.next
    return state

def route_to_agent(state: SupervisorState) -> str:
    if state.next_agent == "FINISH":
        return END
    return state.next_agent

# Construir grafo
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_agent, {
    "researcher": "researcher",
    "writer": "writer",
    "reviewer": "reviewer",
    END: END,
})
# Workers siempre vuelven al supervisor
for worker in ["researcher", "writer", "reviewer"]:
    graph.add_edge(worker, "supervisor")

multi_agent = graph.compile()
```

### Sequential Pattern

Cadena fija de agentes, cada uno procesa y pasa al siguiente.

```python
graph = StateGraph(PipelineState)
graph.add_node("extractor", extract_entities_agent)
graph.add_node("enricher", enrich_data_agent)
graph.add_node("validator", validate_output_agent)
graph.add_node("formatter", format_response_agent)

graph.set_entry_point("extractor")
graph.add_edge("extractor", "enricher")
graph.add_edge("enricher", "validator")
graph.add_edge("validator", "formatter")

pipeline = graph.compile()
```

### Hierarchical Pattern

Supervisores que gestionan sub-equipos.

```python
# Nivel 1: Supervisor principal
# Nivel 2: Sub-supervisores por área
# Nivel 3: Workers especializados

# Sub-equipo de investigación
research_team = StateGraph(ResearchState)
research_team.add_node("research_lead", research_supervisor)
research_team.add_node("web_searcher", web_search_agent)
research_team.add_node("db_searcher", database_search_agent)
# ... routing interno
research_subgraph = research_team.compile()

# Sub-equipo de escritura
writing_team = StateGraph(WritingState)
writing_team.add_node("writing_lead", writing_supervisor)
writing_team.add_node("drafter", draft_agent)
writing_team.add_node("editor", edit_agent)
writing_subgraph = writing_team.compile()

# Supervisor principal orquesta sub-equipos
main_graph = StateGraph(MainState)
main_graph.add_node("director", director_agent)
main_graph.add_node("research_team", research_subgraph)
main_graph.add_node("writing_team", writing_subgraph)
# ...
```

### Collaborative / Debate Pattern

Agentes que debaten para llegar a un consenso.

```python
class DebateState(BaseModel):
    topic: str
    rounds: int = 0
    max_rounds: int = 3
    positions: list[AgentPosition] = []
    consensus: str | None = None

async def advocate_node(state: DebateState) -> DebateState:
    """Agente que argumenta a favor."""
    position = await llm.generate(
        prompt=f"Argue FOR: {state.topic}\nPrevious arguments: {state.positions}"
    )
    state.positions.append(AgentPosition(agent="advocate", argument=position))
    return state

async def critic_node(state: DebateState) -> DebateState:
    """Agente que argumenta en contra."""
    position = await llm.generate(
        prompt=f"Argue AGAINST: {state.topic}\nPrevious arguments: {state.positions}"
    )
    state.positions.append(AgentPosition(agent="critic", argument=position))
    return state

async def judge_node(state: DebateState) -> DebateState:
    """Agente que evalúa y busca consenso."""
    state.rounds += 1
    judgment = await llm.generate_structured(
        prompt=f"Evaluate arguments and determine consensus:\n{state.positions}",
        schema=JudgmentResult,
    )
    if judgment.has_consensus or state.rounds >= state.max_rounds:
        state.consensus = judgment.conclusion
    return state
```

---

## Frameworks

### LangGraph

Framework principal para orquestación. Usa grafos de estado.

```python
from langgraph.graph import StateGraph, MessagesState

graph = StateGraph(MessagesState)
graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)
graph.add_edge("researcher", "writer")
graph.set_entry_point("researcher")
app = graph.compile()
```

**Features clave:**
- State machines con ciclos y condicionales
- Checkpointing nativo (persistencia de estado)
- Human-in-the-loop con interrupt
- Streaming de eventos
- Subgraphs para composición jerárquica

### CrewAI

Framework para equipos de agentes con roles y objetivos.

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive information about {topic}",
    backstory="You are an expert researcher with 20 years of experience.",
    llm=llm,
    tools=[search_tool, scrape_tool],
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write a clear, structured report based on research findings",
    backstory="You are a senior technical writer.",
    llm=llm,
)

research_task = Task(
    description="Research {topic} thoroughly",
    expected_output="Detailed research findings with sources",
    agent=researcher,
)

writing_task = Task(
    description="Write a report based on the research",
    expected_output="A structured report in markdown",
    agent=writer,
    context=[research_task],  # Depende del research
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # O Process.hierarchical
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "AI Governance 2025"})
```

### AutoGen

Framework de Microsoft para agentes conversacionales multi-turno.

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Agentes
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4o"},
    system_message="You are a helpful AI assistant.",
)

coder = AssistantAgent(
    name="coder",
    llm_config={"model": "gpt-4o"},
    system_message="You write Python code. Return only code blocks.",
)

reviewer = AssistantAgent(
    name="reviewer",
    llm_config={"model": "gpt-4o"},
    system_message="You review code for bugs, security, and best practices.",
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "output"},
)

# Group chat — agentes conversan entre sí
group_chat = GroupChat(
    agents=[user_proxy, coder, reviewer],
    messages=[],
    max_round=10,
)
manager = GroupChatManager(groupchat=group_chat, llm_config={"model": "gpt-4o"})

await user_proxy.a_initiate_chat(manager, message="Write a FastAPI health check endpoint")
```

### Semantic Kernel

SDK de Microsoft para integración de AI en aplicaciones enterprise.

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o"))

# Plugins como funciones nativas
@kernel.function(name="search_docs", description="Search internal documents")
async def search_docs(query: str) -> str:
    results = await vector_store.search(query)
    return format_results(results)

# Ejecutar con planificación automática
result = await kernel.invoke_prompt(
    "Find information about {{$topic}} and summarize it",
    topic="microservices architecture",
)
```

---

## State Management & Checkpointing

### Estado Compartido

```python
from pydantic import BaseModel

class AgentState(BaseModel):
    """Estado compartido entre agentes.

    Cada agente lee y escribe en este estado.
    El estado es la fuente de verdad del flujo.
    """
    messages: list[dict] = []
    documents: list[Document] = []
    current_step: str = ""
    results: dict = {}
    errors: list[str] = []
    metadata: dict = {}
```

### Checkpointing con LangGraph

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Persistir estado para recovery y debugging
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
    agent = graph.compile(checkpointer=saver)

    # Ejecutar con thread_id para tracking
    config = {"configurable": {"thread_id": "task-123"}}
    result = await agent.ainvoke(initial_state, config=config)

    # Recuperar estado en cualquier punto
    state = await agent.aget_state(config)
    print(state.values)  # Estado actual
    print(list(state.history))  # Historial de estados
```

---

## Human-in-the-Loop

```python
from langgraph.graph import StateGraph

# Definir punto de interrupción
graph = StateGraph(AgentState)
graph.add_node("draft", draft_agent)
graph.add_node("review", human_review)  # Pausa aquí
graph.add_node("publish", publish_agent)

agent = graph.compile(
    checkpointer=saver,
    interrupt_before=["review"],  # Pausa antes de review
)

# Ejecutar hasta el punto de interrupción
config = {"configurable": {"thread_id": "doc-456"}}
result = await agent.ainvoke(state, config=config)
# → Se detiene antes de "review"

# El humano revisa y aprueba
await agent.aupdate_state(config, {"approved": True})

# Continuar ejecución
result = await agent.ainvoke(None, config=config)
```

---

## Error Recovery & Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientAgent:
    """Agente con error recovery y retry automático."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def execute(self, state: AgentState) -> AgentState:
        try:
            result = await self._llm.generate(state.current_prompt)
            state.results[self.name] = result
            return state
        except LLMTimeoutError:
            state.errors.append(f"{self.name}: timeout, retrying...")
            raise  # Tenacity reintenta
        except LLMRateLimitError:
            state.errors.append(f"{self.name}: rate limited, backing off...")
            raise

async def fallback_node(state: AgentState) -> AgentState:
    """Fallback cuando un agente falla después de retries."""
    state.results["fallback"] = "Unable to complete. Using cached/default response."
    return state

# En el grafo
graph.add_node("primary", primary_agent)
graph.add_node("fallback", fallback_node)
graph.add_conditional_edges("primary", check_errors, {
    "success": "next_step",
    "failed": "fallback",
})
```

---

## Agent Communication Patterns

### Sync (Request-Response)

Un agente llama a otro y espera respuesta. Simple, pero puede bloquear.

```python
# Dentro de un grafo LangGraph — comunicación implícita via state
graph.add_edge("agent_a", "agent_b")  # A termina, B empieza con el state actualizado
```

### Async (Event-Driven)

Agentes publican eventos, otros reaccionan. Desacoplado, escalable.

```python
# Agente publica resultado como evento
async def agent_with_events(state: AgentState) -> AgentState:
    result = await process(state)
    await event_bus.publish("analysis.completed", {
        "agent": "analyzer",
        "result": result,
    })
    return state

# Otro agente reacciona al evento
@event_bus.subscribe("analysis.completed")
async def on_analysis_completed(event: dict):
    await reporter_agent.process(event["result"])
```

### Message Passing

Agentes se comunican via mensajes tipados.

```python
class AgentMessage(BaseModel):
    sender: str
    receiver: str
    content: str
    message_type: str  # "request", "response", "notification"
    correlation_id: str
    timestamp: datetime
```

---

## Agentic RAG

Agentes que deciden dinámicamente cómo y cuándo hacer retrieval.

```python
class AgenticRAGState(BaseModel):
    question: str
    needs_retrieval: bool = True
    retrieved_docs: list[Document] = []
    answer: str | None = None

async def router_node(state: AgenticRAGState) -> AgenticRAGState:
    """El agente decide si necesita buscar documentos o puede responder directamente."""
    decision = await llm.generate_structured(
        prompt=f"Can you answer this without additional context? Question: {state.question}",
        schema=NeedsRetrievalDecision,
    )
    state.needs_retrieval = decision.needs_retrieval
    return state

async def retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    state.retrieved_docs = await vector_store.search(state.question, top_k=5)
    return state

async def generate_node(state: AgenticRAGState) -> AgenticRAGState:
    context = format_context(state.retrieved_docs) if state.retrieved_docs else ""
    state.answer = await llm.generate(
        prompt=f"Context: {context}\n\nQuestion: {state.question}"
    )
    return state

graph = StateGraph(AgenticRAGState)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda s: "retrieve" if s.needs_retrieval else "generate")
graph.add_edge("retrieve", "generate")
```

Para GRaR (Graph-Retrieval-Augmented Reasoning) con agentes, ver [RAG.md](RAG.md).

---

## Principios de Diseño

1. **Single Responsibility**: Cada agente tiene una responsabilidad clara
2. **Contratos explícitos**: Los agentes se comunican a través de schemas definidos (Pydantic)
3. **Observabilidad**: Toda interacción entre agentes es traceable
4. **Fallos explícitos**: Si un agente falla, el sistema lo detecta y responde
5. **Idempotencia**: Las operaciones de agentes deben ser idempotentes cuando sea posible
6. **Stateless por defecto**: Estado gestionado externamente (checkpointing)
7. **Bounded loops**: Todo ciclo de agente tiene un máximo de iteraciones
8. **Human-in-the-loop**: Acciones críticas requieren aprobación humana
9. **Graceful degradation**: Si un agente falla, el sistema continúa con fallbacks

---

## Function Calling / Tool Use

Patrón fundamental para que los LLMs interactúen con sistemas externos.

```python
from pydantic import BaseModel
from instructor import from_openai

class WeatherQuery(BaseModel):
    city: str
    unit: str = "celsius"

class WeatherResult(BaseModel):
    temperature: float
    description: str

# Con Instructor
client = from_openai(AsyncOpenAI())
result = await client.chat.completions.create(
    model="gpt-4o",
    response_model=WeatherResult,
    messages=[{"role": "user", "content": "Weather in Madrid?"}],
)
```

**Reglas para tools:**
- Toda tool tiene input y output tipados (Pydantic)
- Tools son funciones puras cuando es posible
- Validar inputs antes de ejecutar
- Timeout en toda ejecución
- Logging de toda invocación

---

## A2A Protocol (Agent-to-Agent)

Protocolo de Google para comunicación estandarizada entre agentes.

**Conceptos clave:**
- **Agent Card**: Descripción pública de capacidades de un agente
- **Task**: Unidad de trabajo con lifecycle (submitted → working → completed/failed)
- **Message**: Comunicación entre agentes con parts tipadas
- **Artifact**: Outputs generados por el agente

```json
{
  "name": "research-agent",
  "description": "Researches topics using web search",
  "url": "https://agents.example.com/research",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "web-research",
      "name": "Web Research",
      "description": "Search and summarize web content"
    }
  ]
}
```

**Integración con el proyecto:**
- Agent Cards en `src/infrastructure/a2a/`
- Cada agente expone su card como endpoint
- Comunicación via HTTP/JSON estándar
- Compatible con cualquier framework de agentes

---

## Anti-patrones

- **God Agent**: Un agente que hace todo. Dividir en agentes especializados.
- **Implicit Coordination**: Agentes que se coordinan sin contrato explícito.
- **Unbounded Loops**: Agentes en loops sin condición de salida.
- **Silent Failures**: Agentes que fallan sin notificar al sistema.
- **Shared Mutable State**: Agentes modificando estado compartido sin sincronización.

Ver también: [MCP.md](MCP.md), [EVALUATION.md](EVALUATION.md)
