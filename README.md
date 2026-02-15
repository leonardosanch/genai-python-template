# GenAI Python Template

Template de referencia para construir **sistemas de Inteligencia Artificial Generativa en producción** con Python.

No es un demo ni un notebook experimental. Es una base arquitectónica para sistemas mantenibles, escalables y seguros.

---

## Quick Start

### Opción A: Explorar el Template

```bash
# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar y configurar
git clone <repo-url> && cd genai-python-template
uv sync

# Ejecutar tests
uv run pytest tests/unit/ -q

# Ejecutar linters
uv run ruff check src/ tests/
uv run mypy src/

# CLI
uv run genai --help
```

### Opción B: Crear un Proyecto Nuevo

**Usa esto cuando:** Quieres iniciar un proyecto desde cero con la arquitectura del template.

```bash
# 1. Clonar el template (una vez)
git clone <repo-url> ~/templates/genai-python-template

# 2. Crear tu proyecto
cd ~/templates/genai-python-template
./scripts/new-project.sh ~/proyectos/mi-proyecto

# 3. Empezar a trabajar
cd ~/proyectos/mi-proyecto
uv sync
```

### Opción C: Trabajar en un Proyecto Existente

**Usa esto cuando:** Te asignan trabajar en un proyecto de la empresa que ya existe (no creado desde este template).

```bash
# 1. Clonar el template (una vez)
git clone <repo-url> ~/templates/genai-python-template

# 2. Instalar configuración global (una vez por máquina)
cd ~/templates/genai-python-template
./scripts/setup-global.sh

# 3. Clonar y trabajar en el proyecto de la empresa
git clone <company-repo-url> ~/projects/company-project
cd ~/projects/company-project
# Claude Code ahora tiene acceso a las reglas y skills globales
```

**📖 Guía completa:** Ver [SETUP_GUIDE.md](SETUP_GUIDE.md) (English) o [SETUP_GUIDE.es.md](SETUP_GUIDE.es.md) (Español)

---

## Stack Tecnológico

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.12+ |
| Package Manager | uv |
| Web Framework | FastAPI (async-first, Pydantic nativo) |
| LLM Providers | OpenAI, LiteLLM (100+ providers) |
| LLM Frameworks | LangChain, LangGraph, CrewAI, Instructor |
| Bases de Datos | PostgreSQL (pgvector), Redis, MongoDB |
| ORM | SQLAlchemy 2.0 (async) + Alembic |
| Vector Stores | ChromaDB (incluido), Pinecone, Weaviate, Qdrant |
| Cloud Storage | S3 (aiobotocore) |
| Observabilidad | OpenTelemetry, structlog |
| Testing | pytest, DeepEval, RAGAS |
| Data Processing | Pandas, Polars, PySpark |
| ML | scikit-learn, XGBoost, PyTorch, MLflow |
| Task Queues | Celery |
| Message Brokers | Kafka, RabbitMQ |
| CLI | Typer |
| Linting | ruff, mypy |
| CI/CD | GitHub Actions, SonarQube |
| Infraestructura | Docker, Kubernetes, Terraform |

---

## Estructura del Proyecto

```
src/
├── domain/                  # Lógica de negocio pura, sin frameworks
│   ├── entities/            # Dataset, Document
│   ├── ports/               # 12 interfaces abstractas (LLM, storage, MCP...)
│   ├── value_objects/       # Inmutables: schemas, metadata, quality results
│   ├── events.py            # Eventos de dominio
│   └── exceptions.py        # Excepciones de dominio
├── application/             # Casos de uso y orquestación
│   ├── use_cases/           # QueryRAG, RunETL, ValidateDataset, Summarize, Classify
│   ├── pipelines/           # Base pipeline, data cleaning, document ingestion
│   ├── dtos/                # Request/Response DTOs
│   └── guards/              # Output validation (PII, prompt leak)
├── infrastructure/          # Sistemas externos (adapters)
│   ├── llm/                 # OpenAI + LiteLLM adapters
│   ├── cloud/               # S3 storage
│   ├── data/                # Local file source/sink, validators
│   ├── database/            # SQLAlchemy + Alembic migrations
│   ├── cache/               # Redis
│   ├── mcp/                 # MCP client (stdio JSON-RPC)
│   ├── ml/                  # scikit-learn adapter
│   ├── events/              # In-memory event bus
│   ├── observability/       # Logging, tracing, metrics
│   ├── security/            # Guard service (AST + LLM-as-judge)
│   ├── tasks/               # Celery app + workers
│   └── config/              # Settings (pydantic-settings)
├── interfaces/              # Puntos de entrada
│   ├── api/                 # FastAPI app
│   │   ├── routes/          # RAG, pipeline, data, stream, WebSocket
│   │   └── middleware/      # Auth, rate limiting, logging, error handler
│   └── cli/                 # Typer CLI (generate, stream, etl, validate)
├── examples/                # 23 ejemplos de referencia
tests/
├── unit/                    # 122 tests
├── integration/             # 13 tests
└── examples/                # Tests de ejemplos
docs/skills/                 # 15 skills especializados
scripts/                     # setup-global.sh, new-project.sh
deploy/
├── k8s/                     # Kubernetes manifests
├── terraform/               # Terraform scaffold
└── docker/                  # Dockerfile, Dockerfile.worker
```

---

## Documentación

### Guías del Proyecto

| Documento | Descripción |
|-----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Reglas operativas y principios para Claude Code |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Guía de configuración del template |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Cómo contribuir al proyecto |
| [CHANGELOG.md](CHANGELOG.md) | Historial de cambios |

### Arquitectura y Diseño

| Documento | Descripción |
|-----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Clean Architecture, capas, flujo de datos |
| [DECISIONS.md](DECISIONS.md) | Architecture Decision Records (ADRs) |
| [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) | Lakehouse, Data Mesh, analytics |

### Dominio GenAI

| Documento | Descripción |
|-----------|-------------|
| [AGENTS.md](AGENTS.md) | Patrones de agentes, multi-agent, A2A |
| [RAG.md](RAG.md) | Retrieval-Augmented Generation |
| [PROMPTS.md](PROMPTS.md) | Prompt engineering y versionado |
| [MCP.md](MCP.md) | Model Context Protocol |
| [EVALUATION.md](EVALUATION.md) | Evaluación de LLMs y RAG |
| [STREAMING.md](STREAMING.md) | SSE, WebSockets, streaming |

### Infraestructura y Operaciones

| Documento | Descripción |
|-----------|-------------|
| [API.md](API.md) | FastAPI, REST, microservicios |
| [DATABASES.md](DATABASES.md) | SQL, NoSQL, ORMs, patrones de acceso |
| [DATA_ENGINEERING.md](DATA_ENGINEERING.md) | ETL, Pandas, Polars, Spark, Airflow |
| [MACHINE_LEARNING.md](MACHINE_LEARNING.md) | scikit-learn, PyTorch, MLflow |
| [EVENT_DRIVEN.md](EVENT_DRIVEN.md) | Celery, RabbitMQ, Kafka, gRPC |
| [AUTOMATION.md](AUTOMATION.md) | CLI tools, scraping, bots |
| [ANALYTICS.md](ANALYTICS.md) | Dashboards, reporting, BI |

### Calidad y Seguridad

| Documento | Descripción |
|-----------|-------------|
| [SECURITY.md](SECURITY.md) | OWASP LLM Top 10, seguridad |
| [GOVERNANCE.md](GOVERNANCE.md) | AI governance, compliance |
| [TESTING.md](TESTING.md) | Estrategia de testing para GenAI |
| [OBSERVABILITY.md](OBSERVABILITY.md) | OpenTelemetry, logging, métricas |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, Kubernetes, CI/CD |
| [TOOLS.md](TOOLS.md) | uv, ruff, mypy, pytest, SonarQube |

### Skills Especializados (19)

Los **Skills** son documentos que guían a Claude/Gemini para generar código production-ready siguiendo best practices verificadas.

| Skill | Archivo |
|-------|---------|
| Software Architecture | [docs/skills/software_architecture.md](docs/skills/software_architecture.md) |
| Security Engineering | [docs/skills/security.md](docs/skills/security.md) |
| GenAI & RAG | [docs/skills/genai_rag.md](docs/skills/genai_rag.md) |
| Multi-Agent Systems | [docs/skills/multi_agent_systems.md](docs/skills/multi_agent_systems.md) |
| Data & ML Engineering | [docs/skills/data_ml_engineering.md](docs/skills/data_ml_engineering.md) |
| API & Streaming | [docs/skills/api_streaming.md](docs/skills/api_streaming.md) |
| Cloud & Infrastructure | [docs/skills/cloud_infrastructure.md](docs/skills/cloud_infrastructure.md) |
| Testing & Quality | [docs/skills/testing_quality.md](docs/skills/testing_quality.md) |
| Observability | [docs/skills/observability_monitoring.md](docs/skills/observability_monitoring.md) |
| Context Engineering | [docs/skills/context_engineering.md](docs/skills/context_engineering.md) |
| Prompt Engineering | [docs/skills/prompt_engineering.md](docs/skills/prompt_engineering.md) |
| Multi-Tenancy | [docs/skills/multi_tenancy.md](docs/skills/multi_tenancy.md) |
| Hallucination Detection | [docs/skills/hallucination_detection.md](docs/skills/hallucination_detection.md) |
| Automation | [docs/skills/automation.md](docs/skills/automation.md) |
| Analytics | [docs/skills/analytics.md](docs/skills/analytics.md) |
| Databases | [docs/skills/databases.md](docs/skills/databases.md) |
| Event-Driven Systems | [docs/skills/event_driven_systems.md](docs/skills/event_driven_systems.md) |
| Model Context Protocol | [docs/skills/mcp.md](docs/skills/mcp.md) |
| AI Governance | [docs/skills/governance.md](docs/skills/governance.md) |

**Cada skill incluye:**
- ✅ Versiones mínimas de dependencias (previene APIs deprecadas)
- ✅ Advertencias sobre librerías inestables
- ✅ Ejemplos de código verificados
- ✅ Enlaces a documentación oficial

**Validación de Skills:** Ver [SKILL_VALIDATION_PROMPT.md](docs/SKILL_VALIDATION_PROMPT.md) para el proceso de validación de calidad que asegura código correcto desde el inicio.

---

## Claude Code Integration

### Slash Commands (7)

| Comando | Descripción |
|---------|-------------|
| `/review-arch` | Review de arquitectura (Clean Architecture + SOLID) |
| `/review-security` | Review de seguridad (OWASP Top 10) |
| `/review-tests` | Review de cobertura y calidad de tests |
| `/generate-tests` | Scaffold de test file (layer-aware, async mocks, fixtures) |
| `/create-endpoint` | Scaffold de endpoint FastAPI (route + DTO + use case) |
| `/create-agent` | Scaffold de agente LangGraph (state, nodes, graph) |
| `/validate-architecture` | Validar dependency rules de Clean Architecture en `src/` |

### Hooks

| Hook | Descripción |
|------|-------------|
| `pre-commit` | Secrets check + ruff + mypy |
| `pre-push` | Test suite completa + coverage >= 80% |

### MCP Servers

Copiar `.claude/settings.json.example` a `.claude/settings.local.json` y configurar los servers necesarios.

---

## Principios

- **Clean Architecture**: El dominio nunca depende de infraestructura
- **Production-first**: Todo código está pensado para producción
- **Security by default**: Seguridad como requisito, no como extra
- **Observable**: Logging, tracing y métricas desde el día uno
- **Testable**: Domain logic unitaria, LLM interactions con contratos
- **Provider-agnostic**: Sin vendor lock-in en la lógica core

---

## License

MIT
