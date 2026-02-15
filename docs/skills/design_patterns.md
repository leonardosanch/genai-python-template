# Skill: Design Patterns for Python

## Description
This skill covers GoF and modern design patterns applied to Python backend and GenAI systems. Use this when choosing patterns for LLM abstraction, agent orchestration, infrastructure adapters, or refactoring code that violates SOLID principles.

## Executive Summary

**Critical pattern rules:**
- **Pattern must solve a real problem** — Never use patterns "to show knowledge" or for speculative future needs
- **Adapter is mandatory for external systems** — LLM providers, databases, cloud services always behind an interface
- **Strategy over if/else chains** — When selecting models, prompt templates, or processing pipelines dynamically
- **Factory for object creation with dependencies** — Never `new` concrete classes with complex setup in business logic
- **Decorator for cross-cutting concerns** — Logging, retries, caching, tracing — never mix with business logic
- **Maximum 1 Singleton per project** — Settings/config only. If you need more, reconsider your architecture

**Read full skill when:** Choosing patterns for new components, refactoring code smells, abstracting LLM providers, designing plugin systems, or reviewing architecture for pattern misuse.

---

## Versiones y Dependencias

| Concepto | Referencia | Notas |
|----------|-----------|-------|
| GoF Patterns | refactoring.guru/design-patterns | Catálogo completo con Python |
| Python Protocols | PEP 544 | Python 3.8+ structural typing |
| ABC (Abstract Base Classes) | `abc` stdlib | Nominal typing alternativo |
| `functools` | stdlib | `lru_cache`, `wraps`, `cached_property` |
| `dataclasses` | stdlib | Lightweight value objects |

### Protocol vs ABC — Cuándo usar cada uno

| Criterio | `Protocol` | `ABC` |
|----------|-----------|-------|
| Tipado | Structural (duck typing) | Nominal (herencia explícita) |
| Import necesario | Solo en type checking | Runtime |
| Cuándo usar | Ports en domain layer | Cuando necesitas enforcement en runtime |
| Rendimiento | Zero overhead | Mínimo overhead |
| **Recomendación** | **Default para Clean Architecture** | Cuando necesitas `isinstance()` checks |

---

## Decision Trees

### Decision Tree 1: Qué patrón usar

```
¿Cuál es tu problema?
├── Necesito abstraer un sistema externo (LLM, DB, API)
│   └── Adapter
│       ├── Define Protocol en domain
│       ├── Implementa adapter en infrastructure
│       └── Inyecta via constructor
├── Necesito crear objetos con configuración compleja
│   ├── ¿Varios tipos relacionados? → Abstract Factory
│   ├── ¿Un tipo con muchas variantes? → Factory Method
│   └── ¿Objeto con muchos parámetros opcionales? → Builder
├── Necesito seleccionar comportamiento dinámicamente
│   ├── ¿Selección en runtime? → Strategy
│   ├── ¿Pipeline de handlers? → Chain of Responsibility
│   └── ¿Encolar para ejecución posterior? → Command
├── Necesito agregar comportamiento sin modificar la clase
│   ├── ¿Cross-cutting (logging, retry, cache)? → Decorator
│   ├── ¿Control de acceso o lazy load? → Proxy
│   └── ¿Simplificar interfaz compleja? → Facade
├── Necesito notificar cambios a múltiples consumers
│   └── Observer / Event Bus
└── Necesito compartir estado global
    └── ¿Es configuración/settings? → Singleton (module-level)
        └── ¿Es otra cosa? → Probablemente no necesitas Singleton
```

### Decision Tree 2: Patterns para GenAI

```
¿Qué estás construyendo?
├── Abstracción de LLM provider (OpenAI, Anthropic, local)
│   └── Adapter + Strategy
│       ├── Adapter: cada provider implementa LLMPort
│       └── Strategy: selección de modelo por use case
├── Pipeline de procesamiento (RAG, ETL, agent)
│   └── Chain of Responsibility + Template Method
│       ├── Chain: cada paso puede transformar o pasar
│       └── Template: pasos fijos con variaciones
├── Tool execution en agentes
│   └── Command
│       ├── Cada tool es un Command con execute()
│       ├── Typed input/output (Pydantic)
│       └── Undo/retry nativo
├── Observabilidad (logging, tracing, metrics)
│   └── Decorator + Observer
│       ├── Decorator: wrap functions con tracing
│       └── Observer: emit events para monitoring
├── Caching de respuestas LLM
│   └── Proxy (cache-aside)
│       ├── Check cache antes de llamar LLM
│       └── Transparent para el caller
└── Orquestación multi-agent
    └── Mediator (Supervisor) + Observer
        ├── Mediator: supervisor coordina agents
        └── Observer: agents emiten eventos de estado
```

---

## External Resources

### Catálogos de Patrones
- **Refactoring Guru — Python**: [refactoring.guru/design-patterns/python](https://refactoring.guru/design-patterns/python)
    - *Best for*: Referencia visual con ejemplos Python para cada patrón
- **Python Design Patterns**: [python-patterns.guide](https://python-patterns.guide/)
    - *Best for*: Patrones idiomáticos de Python (no Java-style)
- **Source Making**: [sourcemaking.com/design_patterns](https://sourcemaking.com/design_patterns)
    - *Best for*: Explicaciones con UML y trade-offs

### Libros
- **Head First Design Patterns** (Freeman, Robson)
    - *Best for*: Aprender patrones por primera vez
- **Design Patterns: Elements of Reusable OO Software** (GoF)
    - *Best for*: Referencia definitiva
- **Architecture Patterns with Python** (Percival, Gregory)
    - *Best for*: Repository, Unit of Work, Event-Driven en Python

---

## Code Examples

### Example 1: Adapter — LLM Provider abstraction

```python
# src/domain/ports/llm_port.py
from typing import Protocol

from src.domain.models.llm import LLMRequest, LLMResponse


class LLMPort(Protocol):
    """Port for LLM providers. Domain never knows about OpenAI, Anthropic, etc."""

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    async def generate_stream(self, request: LLMRequest):  # AsyncIterator[str]
        ...
```

```python
# src/infrastructure/llm/openai_adapter.py
from openai import AsyncOpenAI

from src.domain.models.llm import LLMRequest, LLMResponse
from src.domain.ports.llm_port import LLMPort


class OpenAIAdapter:
    """Adapter: translates domain LLMPort to OpenAI-specific API."""

    def __init__(self, client: AsyncOpenAI, default_model: str = "gpt-4o"):
        self._client = client
        self._default_model = default_model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=request.model or self._default_model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
```

```python
# src/infrastructure/llm/anthropic_adapter.py
from anthropic import AsyncAnthropic

from src.domain.models.llm import LLMRequest, LLMResponse


class AnthropicAdapter:
    """Same LLMPort interface, different provider."""

    def __init__(self, client: AsyncAnthropic, default_model: str = "claude-sonnet-4-5-20250929"):
        self._client = client
        self._default_model = default_model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await self._client.messages.create(
            model=request.model or self._default_model,
            max_tokens=request.max_tokens or 1024,
            messages=[{"role": "user", "content": request.prompt}],
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
```

### Example 2: Strategy — Model selection

```python
# src/domain/ports/model_selector.py
from typing import Protocol

from src.domain.models.llm import LLMRequest


class ModelSelectorStrategy(Protocol):
    """Strategy to select the optimal model based on request characteristics."""

    def select_model(self, request: LLMRequest) -> str: ...


# src/application/strategies/model_selectors.py
from src.domain.models.llm import LLMRequest


class CostOptimizedSelector:
    """Use cheapest model that meets quality requirements."""

    def select_model(self, request: LLMRequest) -> str:
        if request.max_tokens and request.max_tokens < 100:
            return "gpt-4o-mini"
        if request.requires_reasoning:
            return "gpt-4o"
        return "gpt-4o-mini"


class QualityFirstSelector:
    """Always use best model regardless of cost."""

    def select_model(self, request: LLMRequest) -> str:
        if request.requires_reasoning:
            return "o1"
        return "gpt-4o"


class LatencyOptimizedSelector:
    """Use fastest model for real-time interactions."""

    def select_model(self, request: LLMRequest) -> str:
        return "gpt-4o-mini"  # Fastest response time


# Usage in use case:
class ChatUseCase:
    def __init__(self, llm: "LLMPort", selector: ModelSelectorStrategy):
        self._llm = llm
        self._selector = selector

    async def execute(self, request: LLMRequest) -> "LLMResponse":
        request.model = self._selector.select_model(request)
        return await self._llm.generate(request)
```

### Example 3: Decorator — Cross-cutting concerns

```python
# src/infrastructure/decorators/resilience.py
import asyncio
import functools
import time
from collections.abc import Callable
from typing import TypeVar

import structlog

logger = structlog.get_logger()

F = TypeVar("F", bound=Callable)


def with_retry(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator: retry with exponential backoff for transient failures."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = backoff_factor * (2 ** attempt)
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            wait_seconds=wait,
                            error=str(e),
                        )
                        await asyncio.sleep(wait)
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def with_logging(func: F) -> F:
    """Decorator: structured logging for function entry/exit."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.monotonic()
        logger.info("function_start", function=func.__name__)
        try:
            result = await func(*args, **kwargs)
            elapsed = (time.monotonic() - start) * 1000
            logger.info("function_end", function=func.__name__, duration_ms=round(elapsed, 2))
            return result
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "function_error",
                function=func.__name__,
                duration_ms=round(elapsed, 2),
                error=str(e),
            )
            raise

    return wrapper  # type: ignore[return-value]


def with_timeout(seconds: float):
    """Decorator: enforce timeout on async operations."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error("function_timeout", function=func.__name__, timeout=seconds)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# Usage — stack decorators:
# @with_logging
# @with_retry(max_retries=3, backoff_factor=0.5)
# @with_timeout(30.0)
# async def call_llm(request: LLMRequest) -> LLMResponse:
#     return await self._llm.generate(request)
```

### Example 4: Factory — Agent creation

```python
# src/application/factories/agent_factory.py
from typing import Protocol

from src.domain.ports.llm_port import LLMPort


class AgentPort(Protocol):
    """Base interface for all agents."""

    async def run(self, input_text: str) -> str: ...


class AgentFactory:
    """Factory Method: create agents with proper dependency wiring.

    Centralizes agent construction so callers don't need to know
    about concrete implementations or their dependencies.
    """

    def __init__(self, llm: LLMPort, vector_store: "VectorStorePort | None" = None):
        self._llm = llm
        self._vector_store = vector_store

    def create(self, agent_type: str) -> AgentPort:
        match agent_type:
            case "qa":
                from src.infrastructure.agents.qa_agent import QAAgent
                return QAAgent(llm=self._llm)
            case "rag":
                if not self._vector_store:
                    raise ValueError("RAG agent requires a vector store")
                from src.infrastructure.agents.rag_agent import RAGAgent
                return RAGAgent(llm=self._llm, store=self._vector_store)
            case "summarizer":
                from src.infrastructure.agents.summarizer_agent import SummarizerAgent
                return SummarizerAgent(llm=self._llm)
            case _:
                raise ValueError(f"Unknown agent type: {agent_type}")


# Usage:
# factory = AgentFactory(llm=openai_adapter, vector_store=chroma_store)
# agent = factory.create("rag")
# result = await agent.run("What is Clean Architecture?")
```

### Example 5: Command — Tool execution in agents

```python
# src/domain/commands/tool_command.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Base for all tool inputs — validated via Pydantic."""
    pass


class ToolOutput(BaseModel):
    """Base for all tool outputs."""
    success: bool
    result: Any = None
    error: str | None = None


class ToolCommand(ABC):
    """Command pattern: encapsulate tool execution as an object.

    Benefits:
    - Typed inputs/outputs
    - Logging/tracing per execution
    - Undo capability
    - Queue-able for async execution
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def execute(self, input_data: ToolInput) -> ToolOutput: ...


# Concrete tool:
class SearchInput(ToolInput):
    query: str
    max_results: int = 5


class SearchOutput(ToolOutput):
    documents: list[dict] = field(default_factory=list)


class SearchTool(ToolCommand):
    name = "search"
    description = "Search documents in the knowledge base"

    def __init__(self, vector_store: "VectorStorePort"):
        self._store = vector_store

    async def execute(self, input_data: SearchInput) -> SearchOutput:
        try:
            results = await self._store.search(
                query=input_data.query,
                top_k=input_data.max_results,
            )
            return SearchOutput(success=True, documents=results)
        except Exception as e:
            return SearchOutput(success=False, error=str(e))


# Tool registry:
class ToolRegistry:
    """Registry of available tools for agents."""

    def __init__(self):
        self._tools: dict[str, ToolCommand] = {}

    def register(self, tool: ToolCommand) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolCommand:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
```

### Example 6: Proxy — LLM response caching

```python
# src/infrastructure/llm/cached_llm_proxy.py
import hashlib
import json

from src.domain.models.llm import LLMRequest, LLMResponse
from src.domain.ports.llm_port import LLMPort


class CachedLLMProxy:
    """Proxy pattern: transparent caching layer for LLM calls.

    Intercepts generate() calls, checks cache first.
    Cache-aside: if miss, call real LLM, store result.
    """

    def __init__(self, llm: LLMPort, cache: "CachePort", ttl_seconds: int = 3600):
        self._llm = llm
        self._cache = cache
        self._ttl = ttl_seconds

    def _cache_key(self, request: LLMRequest) -> str:
        payload = json.dumps({
            "prompt": request.prompt,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }, sort_keys=True)
        return f"llm:cache:{hashlib.sha256(payload.encode()).hexdigest()}"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Skip cache for non-deterministic requests
        if request.temperature and request.temperature > 0:
            return await self._llm.generate(request)

        key = self._cache_key(request)
        cached = await self._cache.get(key)
        if cached:
            return LLMResponse.model_validate_json(cached)

        response = await self._llm.generate(request)
        await self._cache.set(key, response.model_dump_json(), ttl=self._ttl)
        return response

    async def generate_stream(self, request: LLMRequest):
        # Streaming bypasses cache
        async for chunk in self._llm.generate_stream(request):
            yield chunk


# Usage — transparent to callers:
# real_llm = OpenAIAdapter(client)
# cached_llm = CachedLLMProxy(llm=real_llm, cache=redis_cache, ttl_seconds=7200)
# response = await cached_llm.generate(request)  # Same interface as LLMPort
```

### Example 7: Observer — Event system for monitoring

```python
# src/domain/events/event_bus.py
import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DomainEvent:
    """Base event — all domain events inherit from this."""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallCompleted(DomainEvent):
    event_type: str = "llm.call.completed"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    cost_usd: float = 0


@dataclass
class AgentStepCompleted(DomainEvent):
    event_type: str = "agent.step.completed"
    agent_name: str = ""
    step_name: str = ""
    iteration: int = 0


EventHandler = Callable[[DomainEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Observer pattern: decouple event producers from consumers.

    Usage:
    - LLM adapter emits LLMCallCompleted
    - Cost tracker, logger, metrics collector subscribe independently
    - No coupling between producer and consumers
    """

    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: DomainEvent) -> None:
        handlers = self._handlers.get(event.event_type, [])
        if handlers:
            await asyncio.gather(
                *(handler(event) for handler in handlers),
                return_exceptions=True,
            )


# Subscribers:
async def log_llm_call(event: DomainEvent) -> None:
    e = event  # type: LLMCallCompleted
    logger.info("llm_call", model=e.model, tokens=e.input_tokens + e.output_tokens)


async def track_cost(event: DomainEvent) -> None:
    e = event  # type: LLMCallCompleted
    await metrics.increment("llm_cost_usd", e.cost_usd, tags={"model": e.model})


# Wiring:
# bus = EventBus()
# bus.subscribe("llm.call.completed", log_llm_call)
# bus.subscribe("llm.call.completed", track_cost)
```

### Example 8: Chain of Responsibility — RAG processing pipeline

```python
# src/application/pipelines/rag_chain.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class RAGContext:
    """Context object passed through the chain."""
    query: str
    documents: list[dict] = field(default_factory=list)
    reranked_documents: list[dict] = field(default_factory=list)
    response: str = ""
    metadata: dict = field(default_factory=dict)


class RAGHandler(ABC):
    """Chain of Responsibility: each handler processes or passes."""

    def __init__(self):
        self._next: RAGHandler | None = None

    def set_next(self, handler: "RAGHandler") -> "RAGHandler":
        self._next = handler
        return handler

    async def handle(self, context: RAGContext) -> RAGContext:
        context = await self.process(context)
        if self._next:
            return await self._next.handle(context)
        return context

    @abstractmethod
    async def process(self, context: RAGContext) -> RAGContext: ...


class RetrievalHandler(RAGHandler):
    def __init__(self, vector_store: "VectorStorePort"):
        super().__init__()
        self._store = vector_store

    async def process(self, context: RAGContext) -> RAGContext:
        context.documents = await self._store.search(context.query, top_k=10)
        logger.info("retrieval_done", count=len(context.documents))
        return context


class RerankHandler(RAGHandler):
    def __init__(self, reranker: "RerankerPort"):
        super().__init__()
        self._reranker = reranker

    async def process(self, context: RAGContext) -> RAGContext:
        context.reranked_documents = await self._reranker.rerank(
            query=context.query,
            documents=context.documents,
            top_k=5,
        )
        logger.info("rerank_done", count=len(context.reranked_documents))
        return context


class GenerationHandler(RAGHandler):
    def __init__(self, llm: "LLMPort"):
        super().__init__()
        self._llm = llm

    async def process(self, context: RAGContext) -> RAGContext:
        docs = context.reranked_documents or context.documents
        prompt = f"Based on these documents:\n{docs}\n\nAnswer: {context.query}"
        response = await self._llm.generate(LLMRequest(prompt=prompt))
        context.response = response.content
        return context


# Build pipeline:
# retriever = RetrievalHandler(vector_store)
# reranker = RerankHandler(cohere_reranker)
# generator = GenerationHandler(llm)
#
# retriever.set_next(reranker).set_next(generator)
#
# result = await retriever.handle(RAGContext(query="What is SOLID?"))
# print(result.response)
```

---

## Pattern Selection by Layer

| Layer | Patterns frecuentes | Por qué |
|---|---|---|
| **Domain** | Value Object, Entity, Domain Event, Protocol | Lógica pura, sin dependencias |
| **Application** | Factory, Strategy, Command, Chain of Responsibility | Orquestación de use cases |
| **Infrastructure** | Adapter, Proxy, Decorator, Repository | Integración con sistemas externos |
| **Interfaces** | Facade, Decorator (middleware) | Simplificar API surface |

---

## Anti-Patterns to Avoid

### Singleton Abuse
**Problem**: Usar Singleton para servicios, repositories, o LLM clients — oculta dependencias y dificulta testing
**Solution**: Dependency injection via constructor. El contenedor DI maneja el lifecycle.

### Pattern Obsession
**Problem**: Aplicar Strategy + Factory + Observer para un if/else de 3 líneas
**Solution**: Si el código es simple y no va a crecer, déjalo simple. Los patrones son para gestionar complejidad que **ya existe**.

### Adapter sin Protocol
**Problem**: `class OpenAIAdapter` que no implementa ninguna interfaz — no es un adapter real
**Solution**: Siempre definir el Protocol/ABC primero en domain. El adapter implementa esa interfaz.

### God Factory
**Problem**: Una factory que crea 20 tipos diferentes de objetos
**Solution**: Una factory por familia de objetos. Si crece demasiado, usar Abstract Factory o módulos separados.

### Decorator Hell
**Problem**: 5+ decorators stacked en una función — dificulta debugging y el orden importa
**Solution**: Máximo 3 decorators. Si necesitas más, considerar middleware o un pipeline explícito.

### Leaky Abstraction
**Problem**: El adapter expone detalles del provider (`openai.RateLimitError` se propaga al domain)
**Solution**: El adapter atrapa excepciones del provider y lanza domain exceptions (`LLMProviderError`).

---

## Performance Considerations

| Pattern | Overhead | Cuándo preocuparte |
|---|---|---|
| Adapter | Negligible (1 indirection) | Nunca |
| Strategy | Negligible | Nunca |
| Decorator | Mínimo per-call | Si > 3 stacked en hot path |
| Observer | `asyncio.gather` overhead | Si > 50 subscribers por evento |
| Proxy | Cache lookup latency | Si cache miss ratio > 80% |
| Factory | Object creation cost | Si creación en hot loop (usar pool) |
| Chain of Responsibility | N handler calls | Si chain > 10 handlers |

---

## Checklist

### Antes de aplicar un patrón
- [ ] ¿Existe un problema real que el patrón resuelve?
- [ ] ¿La solución sin patrón es más compleja que con patrón?
- [ ] ¿El patrón será entendido por otro developer en 5 minutos?
- [ ] ¿No estoy aplicando el patrón para "futuros requisitos"?

### Después de aplicar
- [ ] Protocol/ABC definido en domain layer (no en infrastructure)
- [ ] Tests no necesitan mocks complejos para el patrón
- [ ] El patrón no introduce más de 1 nivel de indirección
- [ ] Documentado brevemente por qué se eligió este patrón

---

## Additional References

- **Refactoring Guru — Python Patterns**: [refactoring.guru/design-patterns/python](https://refactoring.guru/design-patterns/python)
    - *Best for*: Catálogo visual completo con código Python
- **Python Patterns Guide**: [python-patterns.guide](https://python-patterns.guide/)
    - *Best for*: Patrones Pythónicos (no traducciones de Java)
- **Architecture Patterns with Python**: [cosmicpython.com](https://www.cosmicpython.com/)
    - *Best for*: Repository, Unit of Work, Event-Driven DDD en Python
- **SOLID in Python**: [realpython.com/solid-principles-python](https://realpython.com/solid-principles-python/)
    - *Best for*: SOLID aplicado a Python con ejemplos prácticos
