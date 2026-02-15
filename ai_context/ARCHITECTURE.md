# Arquitectura

## Clean Architecture

Este proyecto sigue **Clean Architecture** (Robert C. Martin) adaptada para sistemas GenAI.

La regla fundamental: **las dependencias siempre apuntan hacia adentro**. El dominio nunca conoce la infraestructura.

---

## Capas

### Domain (`src/domain/`)

Lógica de negocio pura. Sin dependencias externas, sin frameworks.

- Entidades y value objects
- Interfaces/ports (abstracciones que la infraestructura implementa)
- Reglas de negocio
- Excepciones de dominio

```python
# Ejemplo: port para LLM
from abc import ABC, abstractmethod

class LLMPort(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: type[T]) -> T: ...
```

### Application (`src/application/`)

Casos de uso y orquestación. Coordina domain e infrastructure a través de ports.

- Use cases (un caso de uso por clase)
- Orquestación de agentes
- DTOs de entrada/salida
- Application services

```python
class SummarizeDocumentUseCase:
    def __init__(self, llm: LLMPort, retriever: RetrieverPort):
        self._llm = llm
        self._retriever = retriever

    async def execute(self, query: str) -> Summary:
        context = await self._retriever.retrieve(query)
        return await self._llm.generate_structured(
            prompt=build_summary_prompt(query, context),
            schema=Summary,
        )
```

### Infrastructure (`src/infrastructure/`)

Implementaciones concretas de los ports del dominio.

- `llm/` — Adaptadores para OpenAI, Anthropic, Gemini, modelos locales
- `database/` — Repositories para PostgreSQL, MongoDB, etc. (SQLAlchemy, Django ORM)
- `cache/` — Redis cache
- `vectorstores/` — Pinecone, Weaviate, Qdrant, ChromaDB, pgvector
- `mcp/` — Clientes y servidores MCP
- `cloud/` — AWS, Azure, GCP
- `observability/` — OpenTelemetry, structured logging, métricas

```python
class OpenAIAdapter(LLMPort):
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
```

### Interfaces (`src/interfaces/`)

Puntos de entrada al sistema.

- `api/` — FastAPI endpoints (REST, SSE, WebSocket)
- `cli/` — Comandos CLI (click, typer)

---

## Flujo de Datos

```
User Request
    → Interface (API/CLI)
        → Application (Use Case)
            → Domain (Business Logic + Ports)
                → Infrastructure (LLM, DB, MCP)
            ← Response
        ← DTO
    ← API Response
```

---

## Inyección de Dependencias

Usar dependency injection explícita. Las dependencias se resuelven en el punto de entrada (composition root), no dentro de las capas.

```python
# Composition root (main.py o factory)
def create_summarize_use_case() -> SummarizeDocumentUseCase:
    llm = OpenAIAdapter(client=AsyncOpenAI(), model="gpt-4o")
    retriever = PineconeRetriever(index=pinecone_index)
    return SummarizeDocumentUseCase(llm=llm, retriever=retriever)
```

---

## Reglas Estrictas

1. **Domain** no importa nada de infrastructure, application o interfaces
2. **Application** solo importa de domain (ports e interfaces)
3. **Infrastructure** implementa ports de domain
4. **Interfaces** solo interactúa con application (use cases)
5. Sin dependencias circulares entre capas
6. Cada módulo tiene un `__init__.py` que expone su API pública

---

## Estructura de Directorios Detallada

```
src/
├── domain/
│   ├── __init__.py
│   ├── entities/           # Entidades de negocio
│   ├── ports/              # Interfaces abstractas
│   ├── value_objects/      # Value objects inmutables
│   └── exceptions.py      # Excepciones de dominio
├── application/
│   ├── __init__.py
│   ├── use_cases/          # Un archivo por caso de uso
│   ├── dtos/               # Data Transfer Objects
│   └── services/           # Application services
├── infrastructure/
│   ├── __init__.py
│   ├── llm/
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   └── gemini_adapter.py
│   ├── vectorstores/
│   │   ├── pinecone_store.py
│   │   ├── qdrant_store.py
│   │   └── chroma_store.py
│   ├── database/
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── session.py          # Engine, session factory
│   │   ├── sqlalchemy_repository.py
│   │   ├── mongo_repository.py
│   │   └── migrations/         # Alembic
│   ├── cache/
│   │   └── redis_cache.py
│   ├── mcp/
│   ├── cloud/
│   │   ├── aws/
│   │   ├── azure/
│   │   └── gcp/
│   └── observability/
│       ├── logging.py
│       ├── tracing.py
│       └── metrics.py
└── interfaces/
    ├── __init__.py
    ├── api/
    │   ├── routes/
    │   ├── middleware/
    │   └── dependencies.py
    └── cli/
        └── commands/
```

Ver también: [DECISIONS.md](DECISIONS.md) para los ADRs que justifican estas decisiones.
