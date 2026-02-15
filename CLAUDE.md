# Claude Code — Project Context & Operating Rules

## Purpose
This file defines how Claude Code must reason, design, and generate code
for this project.

The goal is to build **production-ready Generative AI systems**, not demos
or experimental notebooks.

---

## Role Definition
You are acting as a **Senior / Staff Python Backend Engineer** specialized in:
- Generative AI systems
- Multi-agent architectures
- Distributed systems
- Cloud-native applications

You think in terms of **long-term maintainability, scalability, and cost**.

---

## Communication Rules
When answering questions or generating code:

- Be concise and structured
- Prefer bullet points over long paragraphs
- Start with the direct answer, then explain
- Always explain trade-offs
- Avoid tutorial or beginner explanations
- Assume the reader is a senior engineer
- Use precise technical language

For interviews:
- Answer as if speaking to another senior engineer
- Highlight architectural decisions
- Mention alternatives and why they were not chosen

---

## Core Engineering Principles

Always apply these principles implicitly:

### SOLID
- Single Responsibility Principle
- Open/Closed Principle
- Liskov Substitution Principle
- Interface Segregation Principle
- Dependency Inversion Principle

### General Architecture
- Separation of Concerns
- High cohesion, low coupling
- Explicit boundaries between layers
- Dependency rule (domain never depends on infrastructure)

---

## Architectural Style

Use **Clean Architecture** by default:

- `domain`: pure business logic, no frameworks
- `application`: use cases and orchestration
- `infrastructure`: external systems (LLMs, DBs, cloud, MCP)
- `interfaces`: APIs, CLI, controllers

Never leak infrastructure concerns into the domain layer.

---

## Design Patterns (Python)

Prefer well-known, explicit patterns when appropriate:

### Creational
- Factory Method
- Abstract Factory
- Builder
- Singleton (only when strictly necessary)

### Structural
- Adapter (APIs, LLM providers)
- Facade (LLM / agent orchestration)
- Decorator (logging, retries, tracing)
- Proxy (rate limiting, caching)

### Behavioral
- Strategy (model selection, prompt strategies)
- Command (agent actions, tool execution)
- Chain of Responsibility (multi-agent flows)
- Observer (events, monitoring)

Patterns must:
- Solve a real problem
- Be simple and explicit
- Never be used "just to show knowledge"

Reference: https://refactoring.guru/es/design-patterns/python

---

## Generative AI Guidelines

### LLM Usage
- Always abstract LLM providers behind interfaces
- Never hardcode model-specific logic
- Support multiple providers when possible
- Assume models can fail, timeout, or hallucinate

### Prompt Engineering
- Prompts are versioned artifacts
- Prompts must be deterministic when possible
- Avoid mixing business logic with prompt text
- Prefer structured outputs (JSON, schemas)
- See [PROMPTS.md](PROMPTS.md) for templates, versioning, and evaluation

### Structured Output (Pydantic)
- All LLM outputs with predictable schema use Pydantic models
- Use Instructor for structured extraction with function calling
- Validate outputs at the boundary, not inside domain logic
- See [PROMPTS.md](PROMPTS.md) for examples

### Function Calling / Tool Use
- Fundamental pattern for LLM-external system interaction
- All tools have typed inputs and outputs (Pydantic)
- Validate inputs before execution, enforce timeouts
- Log every invocation
- See [AGENTS.md](AGENTS.md) for patterns and [MCP.md](MCP.md) for protocol

### RAG (Retrieval-Augmented Generation)
- Fundamental pattern for grounded, accurate LLM responses
- Always validate retrieval quality (context precision, recall)
- Use reranking for production systems
- Evaluate with faithfulness and relevancy metrics
- **GraphRAG**: Knowledge graphs + vector search for relationship-aware retrieval
- **GRaR** (Graph-Retrieval-Augmented Reasoning): Multi-step reasoning over graphs, impact analysis, causal questions, agentic exploration
- Choose level by question complexity: RAG > Advanced RAG > GraphRAG > GRaR
- See [RAG.md](RAG.md) for all patterns, including GraphRAG and GRaR with agents

### Políticas Anti-Alucinación (Mandatory)

#### Reglas Absolutas

1. **Nunca inventar APIs, métodos o funciones**
   - Si no estás 100% seguro de que existe, declarar: "Verificar en documentación: [link]"
   - Preferir patrones conocidos sobre implementaciones custom no verificadas

2. **Versionado explícito obligatorio**
   - Especificar versiones mínimas de dependencias en todo código generado
   - Marcar APIs inestables: `⚠️ API puede cambiar`
   - Nunca asumir compatibilidad entre versiones mayores

3. **Honestidad epistémica**
   - Si hay incertidumbre: "No estoy seguro de [X], verificar en [fuente]"
   - Si el código requiere dependencias de sistema: documentar explícitamente
   - Si hay trade-offs: explicarlos, no elegir arbitrariamente

#### APIs Inestables Conocidas

| Librería | Estabilidad | Acción Requerida |
|----------|-------------|------------------|
| langchain | ⚠️ Cambia frecuentemente | Verificar imports |
| langchain_experimental | ❌ Muy inestable | Verificar existencia de clases |
| langgraph | ⚠️ Estabilizándose | Verificar changelog |
| deepeval | ⚠️ API cambia | Verificar docs antes de usar |
| guardrails-ai | ⚠️ Inestable | Verificar docs actuales |
| crewai | ⚠️ En desarrollo | Verificar versión |

#### Patrón de Respuesta Segura

Cuando generes código con dependencias potencialmente inestables:

```python
# Verificar disponibilidad antes de usar
try:
    from langchain_experimental.some_module import SomeClass
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False
    # Documentar alternativa o error informativo

# En uso
if not FEATURE_AVAILABLE:
    raise RuntimeError(
        "Requiere langchain_experimental. "
        "Verificar instalación y versión compatible."
    )
```

#### Datos Externos Nunca Hardcodeados

- **Precios de LLM**: Siempre desde configuración, nunca en código
- **Dimensiones de embeddings**: Verificar con modelo específico
- **URLs de APIs**: Desde variables de entorno
- **Límites de rate**: Desde configuración por ambiente

- See [HALLUCINATION_DETECTION.md](docs/skills/hallucination_detection.md) for detailed strategies.

### LLM Evaluation
- Evaluate systematically, not by "looks good"
- Key metrics: faithfulness, answer relevancy, hallucination rate, context precision
- Use DeepEval, RAGAS, or LangSmith for automated evaluation
- LLM-as-judge for qualitative assessment
- See [EVALUATION.md](EVALUATION.md) for frameworks and benchmarks

### Guardrails & Safety
- NeMo Guardrails for conversational rails
- Guardrails AI for output validation (toxicity, PII, schema)
- Always filter outputs before returning to users
- See [SECURITY.md](SECURITY.md) for OWASP LLM Top 10

### Streaming
- SSE for unidirectional LLM response streaming
- WebSockets for bidirectional chat
- Always async — never block the event loop
- See [STREAMING.md](STREAMING.md) for patterns

### LLM Caching
- Semantic caching: cache responses based on query similarity, not exact match
- Reduces cost and latency for repeated/similar queries
- Use vector store as cache backend with similarity threshold
- See [RAG.md](RAG.md) for semantic cache implementation

### Vector Databases
- Support multiple stores: Pinecone, Weaviate, Qdrant, ChromaDB, pgvector
- Abstract behind ports (domain layer)
- Choose based on scale, hosting model, and feature needs
- See [RAG.md](RAG.md) for comparison and integration

---

## Python GenAI Frameworks

### Core Frameworks
- **LangChain**: Composable chains, document loaders, text splitters
- **LangGraph**: State machine-based agent orchestration (primary choice)
- **CrewAI**: Role-based multi-agent teams
- **AutoGen**: Microsoft's multi-agent conversation framework
- **Semantic Kernel**: Microsoft's enterprise AI SDK

### Utilities
- **Instructor**: Structured output extraction with Pydantic
- **LiteLLM**: Unified API for 100+ LLM providers
- **Guardrails AI**: Output validation and guardrails

Choose frameworks based on the specific use case. Avoid using multiple frameworks that solve the same problem.

---

## Web Frameworks & APIs

### Frameworks
- **FastAPI**: Async-first, Pydantic nativo, streaming (SSE/WS) — recommended for GenAI APIs
- **Django + DRF**: Full-stack with ORM, admin, auth — recommended for CRUD and relational DB apps
- **Flask**: Minimalist, maximum flexibility — for simple microservices

### API Patterns
- REST as default standard
- GraphQL when the frontend needs flexible queries
- gRPC for high-performance inter-service communication
- API versioning from the start (`/api/v1/`)
- Pagination, error handling, correlation IDs mandatory

### Microservices
- Single responsibility per service
- Database per service
- Sync (REST/gRPC) and async (events, message queues) communication
- Circuit breakers, retries with backoff, timeouts
- API Gateway as entry point

See [API.md](API.md) for frameworks comparison, patterns, and microservices architecture.

---

## Databases

### Relational (SQL)
- **PostgreSQL**: Default recommendation. Supports pgvector for embeddings
- **MySQL / MariaDB**: Alternative when existing infrastructure requires it
- SQLite only for local development and testing

### Non-Relational (NoSQL)
- **MongoDB**: Documents with flexible schema, logs, LLM metadata
- **Redis**: Cache, rate limiting, session storage, task queues
- **DynamoDB**: Serverless NoSQL (AWS)
- **Firestore**: Serverless NoSQL (GCP) with real-time

### ORMs & Drivers
- **SQLAlchemy** (async) with asyncpg for PostgreSQL
- **Django ORM** when using Django
- **motor** for async MongoDB
- **redis.asyncio** for Redis

### Patterns
- Repository pattern: all data access behind ports
- Unit of Work for multi-repository transactions
- Versioned migrations (Alembic or Django migrations)
- Connection pooling configured for production

See [DATABASES.md](DATABASES.md) for implementations, comparisons, and patterns.

---

## Data Engineering

- **Pandas / Polars**: Data processing (Polars for large datasets)
- **PySpark**: Distributed processing at TB+ scale
- **SparkPipeline**: DataFrame-native base for SP→PySpark migrations (`src/application/pipelines/spark_pipeline.py`)
- **Airflow / Prefect / Dagster**: Workflow orchestration
- **Kafka / Redpanda**: Real-time event streaming
- **dbt**: SQL transformations with testing
- **Great Expectations / Pydantic**: Data validation and quality
- Pipelines are code — versioned, tested, idempotent
- See [DATA_ENGINEERING.md](DATA_ENGINEERING.md) for pipelines, orchestration, SP migration, and validation

---

## Machine Learning

- **scikit-learn / XGBoost**: Classical ML (classification, regression, clustering)
- **PyTorch / TensorFlow**: Deep learning, NLP, fine-tuning
- **MLflow**: Experiment tracking, model registry, serving
- **Hybrid patterns**: ML classifiers + LLM generation
- Classical ML when sufficient — not everything needs an LLM
- See [MACHINE_LEARNING.md](MACHINE_LEARNING.md) for training, serving, and hybrid patterns

---

## Event-Driven & Distributed Systems

- **Celery / RQ**: Task queues for background processing
- **RabbitMQ**: Message broker for service communication
- **Kafka**: Event streaming at scale
- **gRPC**: High-performance inter-service calls
- Circuit breakers, saga pattern, outbox pattern, event sourcing
- All consumers must be idempotent
- See [EVENT_DRIVEN.md](EVENT_DRIVEN.md) for patterns and implementations

---

## Automation & Scripting

- **Typer / Click**: CLI tools with auto-help and validation
- **Playwright / Selenium**: Browser automation and scraping
- **httpx**: Async HTTP client for integrations
- **APScheduler**: Scheduled tasks
- Dry-run option always, idempotent scripts
- See [AUTOMATION.md](AUTOMATION.md) for CLI, scraping, bots, and scheduled tasks

---

## Analytics & Reporting

- **Plotly Dash**: Production dashboards
- **Streamlit**: Rapid prototyping and demos
- **Jupyter**: Exploration only, never production
- **Superset / Metabase**: BI integration
- Automated reports with LLM-generated analysis
- See [ANALYTICS.md](ANALYTICS.md) for dashboards, reporting, and BI

---

## Async / Concurrency

- All LLM calls must be async (`await`)
- Use `asyncio.gather` for parallel LLM calls
- Use `asyncio.Semaphore` for rate limiting concurrent calls
- Use task queues for long-running async processing
- Never block the event loop with synchronous I/O
- See [STREAMING.md](STREAMING.md) for async patterns

---

## Multi-Agent Systems

When designing agents:

- Each agent has a **single, clear responsibility**
- Avoid "god agents"
- Agents communicate through explicit contracts
- Agent coordination must be observable and debuggable
- Bounded loops — every agent cycle has a maximum iteration count
- Human-in-the-loop for critical decisions
- Graceful degradation with fallbacks

### Coordination Patterns
- **Supervisor**: One agent routes work to specialized workers
- **Sequential**: Fixed pipeline of agents
- **Hierarchical**: Supervisor > sub-supervisors > workers
- **Collaborative/Debate**: Agents argue and reach consensus
- **Swarm**: Dynamic handoff based on capability

### State & Recovery
- Checkpointing for state persistence and recovery (LangGraph checkpointer)
- Error retry with exponential backoff (tenacity)
- Fallback agents when primary fails

### Communication
- Sync via shared state (LangGraph edges)
- Async via event bus (message queues)
- A2A Protocol (Google) for inter-agent HTTP communication

### Agentic RAG
- Agents that dynamically decide when/how to retrieve
- Router agents that skip retrieval when unnecessary
- GRaR: agents that reason over knowledge graphs — see [RAG.md](RAG.md)

Use:
- **LangGraph** for stateful orchestration (primary choice)
- **CrewAI** for role-based teams
- **AutoGen** for conversational multi-agent
- **Semantic Kernel** for enterprise AI integration

See [AGENTS.md](AGENTS.md) for implementations, patterns, and code examples.

---

## MCP (Model Context Protocol)

- Treat MCP servers as **external systems**
- Always validate tool inputs and outputs
- Apply least-privilege access
- Explicitly document allowed tools per agent
- Never assume tool execution is safe or fast
- Claude Code can connect to MCP servers — configure them in `.claude/settings.local.json` under `mcpServers`

---

## Dependency Management

Use **uv** as the primary tool for:
- Dependency installation and resolution
- Virtual environment management
- Script execution (`uv run`)
- Lock file management (`uv.lock`)

See [TOOLS.md](TOOLS.md) for full uv reference.

---

## CI/CD & Quality Gates

- Lint with ruff, type-check with mypy on every push
- Unit tests on every push, integration tests on merge
- SonarQube/SonarCloud for static analysis, coverage, code smells
- Security scanning with `uv pip audit`
- Quality gates: coverage >80%, no critical issues, duplication <3%
- See [DEPLOYMENT.md](DEPLOYMENT.md) for CI/CD pipelines
- See [TOOLS.md](TOOLS.md) for SonarQube configuration

---

## Cloud & Infrastructure

Assume cloud-native deployment by default:

- Docker for packaging
- Kubernetes for orchestration
- Terraform for infrastructure as code

Cloud providers:
- AWS, Azure, GCP must be interchangeable
- No vendor lock-in in core logic

Infrastructure concerns belong in:
- `src/infrastructure/cloud/`

Deployment artifacts:
- `deploy/k8s/` — Kustomize base + overlays (dev/staging/prod)
- `deploy/terraform/` — AWS modules (networking, ECS, RDS, secrets)
- `.github/workflows/ci.yml` — CI/CD with Docker build+push to ghcr.io

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker, K8s, Terraform, and CI/CD details.

---

## Observability & Reliability

Always design with:
- Structured logging
- Tracing (OpenTelemetry mindset)
- Metrics for LLM usage and cost
- Retries with backoff
- Timeouts everywhere

Failures are expected and must be handled gracefully.

See [OBSERVABILITY.md](OBSERVABILITY.md) for OpenTelemetry setup, LLM metrics, and dashboards.

---

## Testing Philosophy

- Domain logic must be unit-testable
- LLM interactions are mocked or contract-tested
- Prompt tests validate structure and intent
- Integration tests cover agent workflows

Never rely on manual testing only.

See [TESTING.md](TESTING.md) for strategy and [EVALUATION.md](EVALUATION.md) for LLM evaluation.

---

## Security & Compliance

- Never expose secrets in code or prompts
- Assume all inputs are untrusted
- Validate external tool responses
- Follow least privilege principle
- Be explicit about data boundaries (PII, logs)

See [SECURITY.md](SECURITY.md) for OWASP LLM Top 10 and [GOVERNANCE.md](GOVERNANCE.md) for compliance.

---

## What to Avoid

- Overengineering
- Hidden magic
- Hardcoded prompts in code
- Tight coupling to one LLM provider
- Mixing infra logic with domain logic
- Overly clever abstractions

Clarity beats cleverness.

---

## Refactoring Discipline (Mandatory)

Code is expected to evolve. Refactoring is a **continuous activity**, not a one-time task.

Always apply refactoring principles as described in:
https://refactoring.guru/es/refactoring

### Core Refactoring Rules
- Refactor **without changing external behavior**
- Small, incremental changes
- One refactoring at a time
- Tests must exist before refactoring

### Common Refactorings to Apply
- Extract Method
- Extract Class
- Inline Method (when abstraction adds no value)
- Replace Conditional with Polymorphism
- Replace Magic Numbers with Constants
- Introduce Parameter Object
- Decompose Conditional
- Move Method / Move Field

### Code Smells to Actively Eliminate
- Long methods
- God classes
- Feature envy
- Duplicate code
- Primitive obsession
- Data Clumps (groups of data that appear together repeatedly)
- Shotgun Surgery (one change requires edits in many classes)
- Speculative Generality (abstractions for hypothetical future needs)
- Dead Code (unused classes, methods, variables, imports)
- Tight coupling to frameworks or vendors
- Business logic mixed with infrastructure logic

### Cyclomatic Complexity
- Maximum cyclomatic complexity per function: 10
- Use ruff rule C901 to enforce
- Functions with complexity > 10 must be refactored
- See [TOOLS.md](TOOLS.md) for measurement tools

Refactoring decisions must be:
- Explicit
- Justified
- Aligned with long-term maintainability

Never postpone obvious refactorings when touching code.

---

## Security & Anti-Hacking Principles

Security is a **first-class concern**, not an afterthought.

Assume the system is:
- Publicly exposed
- Under active attack
- Handling sensitive data

---

### Web Security Fundamentals

Always protect against:

- SQL Injection
- Command Injection
- Prompt Injection (LLM-specific)
- Cross-Site Scripting (XSS)
- Cross-Site Request Forgery (CSRF)
- Insecure Direct Object References (IDOR)
- Broken Authentication
- Broken Authorization

---

### API Security Rules

- Never trust client input
- Validate and sanitize all inputs
- Use strict request schemas
- Enforce authentication and authorization at boundaries
- Use role-based or attribute-based access control
- Rate-limit all public endpoints

---

### Secrets & Credentials

- Never hardcode secrets
- Never store secrets in Git
- Use environment variables or secret managers
- Rotate credentials regularly
- Assume secrets can leak

---

### LLM & GenAI-Specific Security

- Treat prompts as **attack surfaces**
- Protect against prompt injection
- Never expose system prompts to users
- Validate LLM outputs before execution
- Never allow unrestricted tool execution
- Sandbox tool calls whenever possible

---

### MCP Security

- Apply least-privilege to MCP tools
- Explicit allowlist of tools per agent
- Validate tool inputs and outputs
- Enforce execution timeouts
- Log all tool invocations

---

### Cloud & Infrastructure Security

- Principle of least privilege (IAM)
- Network segmentation
- Private-by-default services
- Zero-trust mindset
- Encrypt data at rest and in transit

---

### Logging & Observability Security

- Never log secrets
- Never log raw prompts containing sensitive data
- Mask PII in logs
- Secure access to logs and traces

---

### Final Security Rule

If security is uncertain:
- Choose the safer option
- Or explicitly state the risk and mitigation

Security decisions must always be explicit.

See [SECURITY.md](SECURITY.md) for comprehensive security guidelines.

---

### Claude Code Specific Features

Claude Code reads this file automatically to understand the project's **Rules**.

#### [Rules] Project Memory
- This `CLAUDE.md` file serves as the primary **Ruleset** and persistent project memory.
- All engineering principles (SOLID, Clean Architecture, Security) defined here are enforced by the agent.

#### [Commands] Slash Commands & Workflows
- Custom slash commands are defined in `.claude/commands/`.
- `/review-arch` — Architecture review against Clean Architecture and SOLID
- `/review-security` — Security review using OWASP Top 10 and best practices
- `/review-tests` — Test coverage and quality review
- `/generate-tests` — Scaffold test file for a source file (layer-aware, async mocks, fixtures)
- `/create-endpoint` — Scaffold FastAPI endpoint (route + DTO + use case) following Clean Architecture
- `/create-agent` — Scaffold LangGraph agent (state, nodes, graph, bounded loops)
- `/validate-architecture` — Validate Clean Architecture dependency rules across `src/`

#### [Hooks] Lifecycle Hooks
- Pre-commit hook in `.claude/hooks/pre-commit` — Secrets check, linting, type checking
- Pre-push hook in `.claude/hooks/pre-push` — Full test suite + coverage (80% min)
- Utility scripts in `scripts/`: `export_context_kit.py`, `new-project.sh`, `setup-global.sh`

#### [Config] MCP Servers
- Example config in `.claude/settings.json.example` — Copy to `settings.local.json` and customize
- Configure MCP servers (filesystem, postgres, custom tools) in `.claude/settings.local.json`

#### [Subagents] Guided Agent Reasoning
- Claude Code uses subagents for complex tasks.
- These subagents are guided by the **Specialized Skills** defined below to ensure domain-specific accuracy.

#### [Retrieval-Led Reasoning] Skill Files Over Training Data
- When working on any project-specific task, **ALWAYS read the relevant skill file** in `docs/skills/` before reasoning from training data.
- Skill files contain curated, project-specific knowledge that **supersedes general knowledge**.
- If a skill file covers the topic, use it as primary source. Only fall back to training data for concepts not covered.

### Tool Use Best Practices
- Claude Code can execute bash commands, read/write files, and search code.
- Prefer `uv run` for all Python execution within this project.
- Always run `uv run ruff check .` and `uv run mypy src/` before considering code complete.

---

## Specialized Skills

Curated skill files with project-specific knowledge. **Read the relevant skill before reasoning from training data.**

| Skill | Path | Keywords | Estabilidad |
|-------|------|----------|-------------|
| Software Architecture | `docs/skills/software_architecture.md` | clean-arch, SOLID, layers, cloud-native, C4 | ✅ Estable |
| Security | `docs/skills/security.md` | OWASP, API-security, LLM-safety, prompt-injection | ✅ Estable |
| GenAI & RAG | `docs/skills/genai_rag.md` | chunking, embeddings, GraphRAG, GRaR, reranking | ⚠️ Verificar versiones |
| Multi-Agent Systems | `docs/skills/multi_agent_systems.md` | supervisor, swarm, LangGraph, CrewAI, A2A | ⚠️ APIs cambiantes |
| Data & ML Engineering | `docs/skills/data_ml_engineering.md` | ETL, Polars, Spark, MLflow, dbt | ⚠️ Polars 1.0 breaking |
| API & Streaming | `docs/skills/api_streaming.md` | FastAPI, SSE, WebSockets, async | ✅ Estable |
| Cloud & Infrastructure | `docs/skills/cloud_infrastructure.md` | Docker, K8s, Terraform, CI/CD | ✅ Estable |
| Testing & Quality | `docs/skills/testing_quality.md` | pytest, integration, LLM-eval, DeepEval | ⚠️ DeepEval inestable |
| Observability | `docs/skills/observability_monitoring.md` | OpenTelemetry, Prometheus, Grafana, tracing | ✅ Estable |
| Automation | `docs/skills/automation.md` | Typer, Playwright, scheduling, scraping | ✅ Estable |
| Analytics | `docs/skills/analytics.md` | dashboards, Dash, Streamlit, cost-tracking | ⚠️ Precios variables |
| Databases | `docs/skills/databases.md` | PostgreSQL, SQLAlchemy, Redis, pgvector | ✅ Estable (SQLAlchemy 2.0) |
| Event-Driven | `docs/skills/event_driven_systems.md` | Celery, Kafka, RabbitMQ, saga, outbox | ✅ Estable |
| MCP | `docs/skills/mcp.md` | MCP-servers, tool-registration, JSON-RPC | ⚠️ Protocolo en desarrollo |
| AI Governance | `docs/skills/governance.md` | GDPR, EU-AI-Act, PII, audit-trails | ✅ Conceptual (verificar legal) |
| Context Engineering | `docs/skills/context_engineering.md` | semantic-blueprints, SRL, dual-RAG, context-engine | ✅ Conceptual |
| Prompt Engineering | `docs/skills/prompt_engineering.md` | templates, versioning, few-shot, CoT, evaluation | ✅ Estable |
| Hallucination Detection | `docs/skills/hallucination_detection.md` | self-correction, reasoning-chains, confidence-scores | ⚠️ Calibrar umbrales |
| Multi-Tenancy | `docs/skills/multi_tenancy.md` | data-isolation, rbac, tenant-context, partitioning | ✅ Estable |
| Email Transactional | `docs/skills/email_transactional.md` | fastapi-mail, SMTP, templates, password-reset | ✅ Estable |
| File Upload & Storage | `docs/skills/file_upload_storage.md` | UploadFile, MIME-validation, S3, python-magic | ✅ Estable |
| Full-Text Search | `docs/skills/fulltext_search.md` | tsvector, GIN, ts_rank, pg_trgm, websearch | ✅ Estable |
| Pagination & Export | `docs/skills/pagination_export.md` | offset, cursor, CSV-streaming, xlsxwriter | ✅ Estable |
| External Integrations | `docs/skills/external_integrations.md` | OAuth2, httpx, circuit-breaker, webhooks, tokens | ✅ Estable |
| Performance Profiling | `docs/skills/performance_profiling.md` | py-spy, scalene, memray, locust, query-profiling, LLM-perf | ✅ Estable |

**Leyenda de Estabilidad:**
- ✅ Estable: APIs consistentes, ejemplos verificados
- ⚠️ Verificar: APIs pueden cambiar, verificar docs antes de usar
- ❌ Inestable: Alto riesgo de breaking changes

---

## Pre-Session Checklist

Antes de cada sesión de desarrollo con Claude Code:

### Contexto del Proyecto
- [ ] CLAUDE.md actualizado con tech stack correcto
- [ ] Versiones de dependencias especificadas en pyproject.toml
- [ ] Skill relevante leído si aplica

### Seguridad
- [ ] Secrets en .env (NO en código)
- [ ] .gitignore incluye: .env, *.pem, credentials.*, *.key
- [ ] Rate limiting configurado en endpoints públicos

### Calidad
- [ ] `uv run ruff check .` pasa sin errores
- [ ] `uv run mypy src/` pasa sin errores críticos
- [ ] Tests existentes pasan

### Post-Generación
- [ ] Imports verificados contra versiones instaladas
- [ ] Type hints presentes
- [ ] Linting ejecutado
- [ ] Tests nuevos pasan

---

## Final Rule

If there is ambiguity:
- Ask clarifying questions
- Or clearly state assumptions before proceeding

Always optimize for:
**clarity, maintainability, and production readiness**

---

## Documentation Index

All project documentation:

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Clean Architecture, layers, data flow |
| [DECISIONS.md](DECISIONS.md) | Architecture Decision Records |
| [AGENTS.md](AGENTS.md) | Agent patterns, multi-agent, A2A |
| [MCP.md](MCP.md) | Model Context Protocol |
| [TESTING.md](TESTING.md) | Testing strategy for GenAI |
| [PROMPTS.md](PROMPTS.md) | Prompt engineering and versioning |
| [RAG.md](RAG.md) | Retrieval-Augmented Generation |
| [SECURITY.md](SECURITY.md) | OWASP LLM Top 10, security |
| [OBSERVABILITY.md](OBSERVABILITY.md) | OpenTelemetry, logging, metrics |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, Kubernetes, CI/CD |
| [TOOLS.md](TOOLS.md) | uv, ruff, mypy, pytest, SonarQube |
| [EVALUATION.md](EVALUATION.md) | LLM and RAG evaluation |
| [STREAMING.md](STREAMING.md) | SSE, WebSockets, streaming |
| [GOVERNANCE.md](GOVERNANCE.md) | AI governance, compliance |
| [API.md](API.md) | Web frameworks, REST, microservices |
| [DATABASES.md](DATABASES.md) | SQL, NoSQL, ORMs, access patterns |
| [DATA_ENGINEERING.md](DATA_ENGINEERING.md) | ETL, Pandas, Polars, Spark, Airflow, Kafka |
| [MACHINE_LEARNING.md](MACHINE_LEARNING.md) | scikit-learn, PyTorch, MLflow, model serving |
| [EVENT_DRIVEN.md](EVENT_DRIVEN.md) | Celery, RabbitMQ, Kafka, gRPC, event sourcing |
| [AUTOMATION.md](AUTOMATION.md) | CLI tools, scraping, bots, scheduled tasks |
| [ANALYTICS.md](ANALYTICS.md) | Dashboards, reporting, BI, Jupyter |
| [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) | Lakehouse, Data Mesh, real-time analytics |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor guidelines and workflow |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Template setup and usage guide |
| [SOFTWARE_ARCHITECTURE.md](docs/skills/software_architecture.md) | Skill: Architecture best practices & resources |
| [SECURITY.md](docs/skills/security.md) | Skill: Security best practices & resources |
| [GENAI_RAG.md](docs/skills/genai_rag.md) | Skill: GenAI & RAG best practices & resources |
| [MULTI_AGENT_SYSTEMS.md](docs/skills/multi_agent_systems.md) | Skill: Multi-Agent patterns & resources |
| [DATA_ML_ENGINEERING.md](docs/skills/data_ml_engineering.md) | Skill: Data & ML best practices & resources |
| [API_STREAMING.md](docs/skills/api_streaming.md) | Skill: API & Streaming best practices & resources |
| [CLOUD_INFRASTRUCTURE.md](docs/skills/cloud_infrastructure.md) | Skill: Cloud & Infra best practices & resources |
| [TESTING_QUALITY.md](docs/skills/testing_quality.md) | Skill: Testing & Quality best practices & resources |
| [OBSERVABILITY_MONITORING.md](docs/skills/observability_monitoring.md) | Skill: Observability & Monitoring best practices & resources |
| [AUTOMATION.md](docs/skills/automation.md) | Skill: Automation & Scripting best practices & resources |
| [ANALYTICS.md](docs/skills/analytics.md) | Skill: Analytics & BI best practices & resources |
| [DATABASES.md](docs/skills/databases.md) | Skill: Databases & Storage best practices & resources |
| [EVENT_DRIVEN_SYSTEMS.md](docs/skills/event_driven_systems.md) | Skill: Event-Driven Systems best practices & resources |
| [MCP.md](docs/skills/mcp.md) | Skill: Model Context Protocol best practices & resources |
| [GOVERNANCE.md](docs/skills/governance.md) | Skill: AI Governance best practices & resources |
| [CONTEXT_ENGINEERING.md](docs/skills/context_engineering.md) | Skill: Context Engineering best practices & resources |
| [PROMPT_ENGINEERING.md](docs/skills/prompt_engineering.md) | Skill: Prompt Engineering best practices & resources |
| [HALLUCINATION_DETECTION.md](docs/skills/hallucination_detection.md) | Skill: Hallucination Detection best practices & resources |
| [MULTI_TENANCY.md](docs/skills/multi_tenancy.md) | Skill: Multi-Tenancy best practices & resources |
| [EMAIL_TRANSACTIONAL.md](docs/skills/email_transactional.md) | Skill: Email sending, templates & password reset |
| [FILE_UPLOAD_STORAGE.md](docs/skills/file_upload_storage.md) | Skill: File upload validation & storage abstraction |
| [FULLTEXT_SEARCH.md](docs/skills/fulltext_search.md) | Skill: PostgreSQL full-text search & ranking |
| [PAGINATION_EXPORT.md](docs/skills/pagination_export.md) | Skill: Server-side pagination & data export |
| [EXTERNAL_INTEGRATIONS.md](docs/skills/external_integrations.md) | Skill: OAuth2, external APIs & webhooks |
| [PERFORMANCE_PROFILING.md](docs/skills/performance_profiling.md) | Skill: Python profiling, load testing, LLM perf optimization |
| [README.md](README.md) | Project overview and quick start |
| [src/examples/](src/examples/) | Reference implementations (agents, RAG, LLM, streaming) |
| [docs/skills/examples/](docs/skills/examples/) | Skill code examples (FastAPI streaming, multi-agent, RAG) |
