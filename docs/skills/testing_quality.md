# Skill: Testing & Quality

## Description
This skill provides comprehensive testing strategies and quality assurance practices for production GenAI applications. Use this when writing tests, ensuring code quality, or validating LLM outputs.

## Executive Summary

**Critical testing rules:**
- ALWAYS mock LLM calls in unit/integration tests — non-deterministic, expensive, slow (use recorded responses)
- Coverage threshold: 80% minimum — enforce in CI with `pytest --cov --cov-fail-under=80`
- Follow test pyramid: 70% unit, 20% integration, 10% E2E — integration tests are not a substitute for unit tests
- RAG evaluation with RAGAS in CI — measure faithfulness, relevancy, context precision with defined thresholds
- Consult Decision Tree 1 before writing tests — unit vs integration vs evaluation test depends on what you're testing

**Read full skill when:** Setting up test infrastructure, writing tests for RAG pipelines, implementing LLM evaluation metrics, configuring CI quality gates, or debugging flaky tests.

---

## Versiones y Thresholds

| Dependencia | Versión Mínima | Threshold CI |
|-------------|----------------|--------------|
| pytest | >= 7.0.0 | N/A |
| pytest-asyncio | >= 0.21.0 | N/A |
| ruff | >= 0.1.0 | 0 Errores |
| mypy | >= 1.5.0 | 0 Errores |
| coverage | >= 7.3.0 | > 80% |

### Mocking LLM Responses

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_llm_response():
    mock = AsyncMock()
    mock.ainvoke.return_value = "Respuesta mockeada"
    return mock
```

---

## Deep Dive

## Core Concepts

1. **Test Pyramid**: Unit tests (base), integration tests (middle), E2E tests (top).
2. **Contract Testing**: Validate LLM API contracts without calling actual models.
3. **Property-Based Testing**: Generate test cases automatically to find edge cases.
4. **Quality Gates**: Automated checks that prevent low-quality code from reaching production.

---

## External Resources

### 🧪 Testing Frameworks & Tools

#### Python Testing
- **pytest Documentation**: [docs.pytest.org](https://docs.pytest.org/)
    - *Best for*: Unit testing, fixtures, parametrization
- **pytest-asyncio**: [pytest-asyncio.readthedocs.io](https://pytest-asyncio.readthedocs.io/)
    - *Best for*: Testing async code (LLM calls, agents)
- **pytest-mock**: [pytest-mock.readthedocs.io](https://pytest-mock.readthedocs.io/)
    - *Best for*: Mocking external dependencies
- **Hypothesis**: [hypothesis.readthedocs.io](https://hypothesis.readthedocs.io/)
    - *Best for*: Property-based testing, finding edge cases

#### Coverage & Quality
- **Coverage.py**: [coverage.readthedocs.io](https://coverage.readthedocs.io/)
    - *Best for*: Code coverage measurement
- **pytest-cov**: [pytest-cov.readthedocs.io](https://pytest-cov.readthedocs.io/)
    - *Best for*: Coverage integration with pytest
- **mutmut**: [mutmut.readthedocs.io](https://mutmut.readthedocs.io/)
    - *Best for*: Mutation testing (test quality validation)

---

### 🤖 LLM & GenAI Testing

#### LLM Testing Frameworks
- **LangSmith**: [docs.smith.langchain.com](https://docs.smith.langchain.com/)
    - *Best for*: LLM application testing, tracing, evaluation
- **PromptFoo**: [promptfoo.dev](https://www.promptfoo.dev/)
    - *Best for*: Prompt testing, regression testing
- **Giskard**: [docs.giskard.ai](https://docs.giskard.ai/)
    - *Best for*: LLM vulnerability scanning, bias detection

#### Evaluation Frameworks
- **RAGAS**: [docs.ragas.io](https://docs.ragas.io/)
    - *Best for*: RAG evaluation (faithfulness, relevancy)
- **DeepEval**: [docs.confident-ai.com](https://docs.confident-ai.com/)
    - *Best for*: LLM evaluation metrics, hallucination detection
- **TruLens**: [trulens.org](https://www.trulens.org/)
    - *Best for*: RAG observability, feedback functions

---

### 🔍 Code Quality Tools

#### Linting & Formatting
- **Ruff**: [docs.astral.sh/ruff/](https://docs.astral.sh/ruff/)
    - *Best for*: Fast Python linter (replaces Flake8, isort, Black)
- **Black**: [black.readthedocs.io](https://black.readthedocs.io/)
    - *Best for*: Opinionated code formatting
- **isort**: [pycqa.github.io/isort/](https://pycqa.github.io/isort/)
    - *Best for*: Import sorting

#### Type Checking
- **mypy**: [mypy.readthedocs.io](https://mypy.readthedocs.io/)
    - *Best for*: Static type checking
- **pyright**: [github.com/microsoft/pyright](https://github.com/microsoft/pyright)
    - *Best for*: Fast type checker (VS Code integration)
- **Pydantic**: [docs.pydantic.dev](https://docs.pydantic.dev/)
    - *Best for*: Runtime validation with type hints

#### Static Analysis
- **SonarQube**: [sonarqube.org](https://www.sonarqube.org/)
    - *Best for*: Code quality, security vulnerabilities, technical debt
- **Bandit**: [bandit.readthedocs.io](https://bandit.readthedocs.io/)
    - *Best for*: Security issue detection in Python
- **Semgrep**: [semgrep.dev](https://semgrep.dev/)
    - *Best for*: Custom security rules, SAST

---

### 📚 Testing Best Practices

#### Books & Guides
- **Test-Driven Development with Python** (Harry Percival)
    - [obeythetestinggoat.com](https://www.obeythetestinggoat.com/)
    - *Best for*: TDD methodology, Django testing
- **Python Testing with pytest** (Brian Okken)
    - *Best for*: Comprehensive pytest guide
- **Effective Software Testing** (Maurício Aniche)
    - *Best for*: Testing principles, test design

#### Testing Patterns
- **Test Doubles**: [martinfowler.com/bliki/TestDouble.html](https://martinfowler.com/bliki/TestDouble.html)
    - *Best for*: Mocks, stubs, fakes, spies
- **Testing Pyramid**: [martinfowler.com/articles/practical-test-pyramid.html](https://martinfowler.com/articles/practical-test-pyramid.html)
    - *Best for*: Test strategy, test distribution

#### Industry Benchmarks
- **DORA 2024 State of DevOps Report**: [dora.dev/research/2024/](https://dora.dev/research/2024/)
    - *Best for*: Elite performance metrics (Deployment Frequency, Lead Time for Changes)

---

## Decision Trees

### Decision Tree 1: Qué tipo de test escribir

```
¿Qué estás testeando?
├── Lógica de dominio pura (cálculos, validaciones, reglas)
│   └── Unit test (pytest)
│       ├── Sin mocks — input → output
│       └── Rápido, determinista, alta cobertura
├── Interacción con DB, cache, o APIs
│   └── Integration test (pytest + testcontainers)
│       ├── DB real en Docker container
│       └── Más lento, pero valida contratos reales
├── Pipeline RAG completo (retrieve → generate → validate)
│   └── Integration test con mocks de LLM
│       ├── Mock el LLM, usar vector store real o in-memory
│       └── Validar que el pipeline conecta correctamente
├── Respuestas del LLM (calidad, faithfulness)
│   └── Evaluation test (RAGAS, DeepEval)
│       ├── Requiere dataset de evaluación curado
│       └── Correr en CI con thresholds (faithfulness > 0.85)
├── Prompt templates
│   └── Property-based test (Hypothesis)
│       ├── Verificar que nunca crashean con input random
│       └── Verificar que siempre incluyen variables requeridas
├── API endpoints
│   └── API test (TestClient de FastAPI)
│       ├── Test de request/response, status codes, validación
│       └── Mock servicios internos
└── Flujo completo usuario → API → LLM → respuesta
    └── E2E test (solo los críticos)
        └── Costoso, lento — mantener < 10% del total
```

### Decision Tree 2: Cuándo mockear vs usar servicio real

```
¿Qué servicio estás llamando?
├── LLM provider (OpenAI, Anthropic)
│   └── SIEMPRE mock en unit/integration tests
│       ├── No determinista, costoso, lento
│       └── Usar respuestas fijas o recorded responses
│           └── Evaluation tests con LLM real (separados, no en CI rápido)
├── Base de datos
│   └── ¿Es un unit test de dominio?
│       ├── SÍ → Mock del repository (interface)
│       └── NO → DB real en testcontainer (PostgreSQL, Redis)
├── Vector store
│   └── ¿Test de retrieval quality?
│       ├── SÍ → Vector store real (ChromaDB in-memory)
│       └── NO → Mock con respuestas predefinidas
├── API externa
│   └── Mock siempre (httpx mock, respfx)
│       └── Contract tests separados para validar schema
└── Message broker (Kafka, RabbitMQ)
    └── Testcontainer o mock según el nivel del test
```

---

## Code Examples

### Example 1: Testing RAG Pipeline

```python
# tests/test_rag_pipeline.py
import pytest
from unittest.mock import AsyncMock, patch
from src.rag.pipeline import RAGPipeline
from src.rag.models import QueryResponse

@pytest.fixture
def mock_llm():
    """Mock LLM to avoid actual API calls."""
    with patch('src.rag.pipeline.ChatOpenAI') as mock:
        mock_instance = AsyncMock()
        mock_instance.ainvoke.return_value = "Mocked response"
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_vectorstore():
    """Mock vector store."""
    with patch('src.rag.pipeline.Pinecone') as mock:
        mock_instance = AsyncMock()
        mock_instance.asimilarity_search.return_value = [
            {"page_content": "Context 1", "metadata": {"source": "doc1"}},
            {"page_content": "Context 2", "metadata": {"source": "doc2"}},
        ]
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_rag_pipeline_returns_response(mock_llm, mock_vectorstore):
    """Test that RAG pipeline returns a valid response."""
    pipeline = RAGPipeline()
    
    response = await pipeline.query("What is Clean Architecture?")
    
    assert isinstance(response, QueryResponse)
    assert response.answer == "Mocked response"
    assert len(response.sources) == 2
    mock_vectorstore.asimilarity_search.assert_called_once()
    mock_llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_rag_pipeline_handles_empty_context(mock_llm, mock_vectorstore):
    """Test RAG pipeline when no relevant documents are found."""
    mock_vectorstore.asimilarity_search.return_value = []
    pipeline = RAGPipeline()
    
    response = await pipeline.query("Obscure question")
    
    assert response.answer is not None
    assert len(response.sources) == 0
```

### Example 2: Property-Based Testing for Prompt Templates

```python
# tests/test_prompt_templates.py
from hypothesis import given, strategies as st
from src.prompts.templates import format_system_prompt

@given(
    role=st.text(min_size=1, max_size=100),
    context=st.text(min_size=0, max_size=500),
)
def test_prompt_template_never_crashes(role, context):
    """Ensure prompt template handles any input without crashing."""
    try:
        result = format_system_prompt(role=role, context=context)
        assert isinstance(result, str)
        assert len(result) > 0
    except ValueError as e:
        # Expected for invalid inputs
        assert "Invalid" in str(e)

@given(role=st.text(min_size=1, max_size=50))
def test_prompt_includes_role(role):
    """Ensure role is always included in the prompt."""
    result = format_system_prompt(role=role, context="")
    assert role in result
```

### Example 3: conftest.py with Reusable Fixtures

```python
# tests/conftest.py
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


# --- Event loop ---
@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for all async tests in session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --- Mock LLM Client ---
@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Reusable mock LLM client that returns structured responses."""
    client = AsyncMock()
    client.generate.return_value = "Mocked LLM response"
    client.astream.return_value = _async_token_generator(["Hello", " ", "World"])
    client.close = AsyncMock()
    return client


async def _async_token_generator(tokens: list[str]):
    for token in tokens:
        yield token


# --- Mock Vector Store ---
@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Mock vector store with predefined search results."""
    store = AsyncMock()
    store.similarity_search.return_value = [
        {"content": "Test context 1", "score": 0.95, "metadata": {"source": "doc1.pdf"}},
        {"content": "Test context 2", "score": 0.87, "metadata": {"source": "doc2.pdf"}},
    ]
    store.upsert = AsyncMock()
    store.delete = AsyncMock()
    return store


# --- Mock Redis ---
@pytest.fixture
def mock_redis() -> AsyncMock:
    cache: dict[str, Any] = {}
    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=lambda k: cache.get(k))
    redis.set = AsyncMock(side_effect=lambda k, v, **kw: cache.update({k: v}))
    redis.delete = AsyncMock(side_effect=lambda k: cache.pop(k, None))
    redis.ping = AsyncMock(return_value=True)
    return redis


# --- FastAPI Test Client ---
@pytest_asyncio.fixture
async def app() -> FastAPI:
    """Create test application with overridden dependencies."""
    from src.interfaces.api.main import create_app

    app = create_app()
    # Override dependencies for testing
    return app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client for FastAPI."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# --- Test Data Factory ---
class Factory:
    """Factory for generating test data."""

    @staticmethod
    def user_dict(**overrides: Any) -> dict[str, Any]:
        defaults = {
            "id": "user-123",
            "email": "test@example.com",
            "name": "Test User",
            "tier": "pro",
        }
        return {**defaults, **overrides}

    @staticmethod
    def llm_response(**overrides: Any) -> dict[str, Any]:
        defaults = {
            "content": "Generated response",
            "model": "gpt-4",
            "tokens_used": 150,
            "finish_reason": "stop",
        }
        return {**defaults, **overrides}


@pytest.fixture
def factory() -> Factory:
    return Factory()
```

### Example 4: Testing Async Agent Workflows

```python
# tests/unit/application/test_agent_workflow.py
import pytest
from unittest.mock import AsyncMock, patch

from src.application.agents.research_agent import ResearchAgent
from src.domain.models.agent_state import AgentState


@pytest.fixture
def research_agent(mock_llm_client, mock_vector_store):
    return ResearchAgent(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        max_iterations=3,
    )


@pytest.mark.asyncio
async def test_research_agent_completes_within_max_iterations(research_agent):
    """Agent must respect max_iterations bound to prevent infinite loops."""
    state = AgentState(goal="Research quantum computing advances")

    result = await research_agent.execute(state)

    assert result.status == "completed"
    assert result.iterations <= 3


@pytest.mark.asyncio
async def test_research_agent_retrieves_before_generating(
    research_agent, mock_vector_store, mock_llm_client,
):
    """Agent must retrieve context before generating response."""
    state = AgentState(goal="Explain RAG pipelines")

    await research_agent.execute(state)

    # Verify retrieval happened before generation
    mock_vector_store.similarity_search.assert_called_once()
    mock_llm_client.generate.assert_called_once()

    # Verify retrieved context was passed to LLM
    call_args = mock_llm_client.generate.call_args
    assert "Test context 1" in call_args[1].get("context", call_args[0][0])


@pytest.mark.asyncio
async def test_research_agent_handles_empty_retrieval(
    research_agent, mock_vector_store,
):
    """Agent should gracefully handle no retrieval results."""
    mock_vector_store.similarity_search.return_value = []
    state = AgentState(goal="Very obscure topic")

    result = await research_agent.execute(state)

    assert result.status == "completed"
    assert "no relevant" in result.output.lower() or result.output != ""


@pytest.mark.asyncio
async def test_research_agent_retries_on_llm_failure(
    research_agent, mock_llm_client,
):
    """Agent should retry LLM call on transient failure."""
    mock_llm_client.generate.side_effect = [
        ConnectionError("Timeout"),
        "Success after retry",
    ]
    state = AgentState(goal="Test resilience")

    result = await research_agent.execute(state)

    assert result.status == "completed"
    assert mock_llm_client.generate.call_count == 2
```

### Example 5: RAGAS Evaluation in CI

```python
# tests/evaluation/test_rag_quality.py
"""RAG evaluation tests — run separately from unit tests.

These tests measure retrieval and generation quality using RAGAS metrics.
Run with: uv run pytest tests/evaluation/ -m evaluation --timeout=120
"""
import pytest

# ⚠️ RAGAS API may change — verify imports against installed version
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from datasets import Dataset


# Evaluation dataset — curated question/answer/context triples
EVAL_DATASET = [
    {
        "question": "What is Clean Architecture?",
        "answer": "",  # Filled by RAG pipeline
        "contexts": [],  # Filled by retriever
        "ground_truth": (
            "Clean Architecture separates software into layers: "
            "domain, application, infrastructure, and interfaces. "
            "The domain layer contains pure business logic with no framework dependencies."
        ),
    },
    {
        "question": "How does the Strategy pattern work?",
        "answer": "",
        "contexts": [],
        "ground_truth": (
            "The Strategy pattern defines a family of algorithms, "
            "encapsulates each one, and makes them interchangeable. "
            "It lets the algorithm vary independently from clients that use it."
        ),
    },
]

# Thresholds — fail CI if metrics drop below these
THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}


@pytest.fixture(scope="module")
def rag_pipeline():
    """Initialize RAG pipeline for evaluation."""
    from src.application.rag.pipeline import RAGPipeline

    return RAGPipeline()


@pytest.fixture(scope="module")
def evaluated_dataset(rag_pipeline):
    """Run RAG pipeline on evaluation dataset."""
    import asyncio

    async def fill_responses():
        for item in EVAL_DATASET:
            result = await rag_pipeline.query(item["question"])
            item["answer"] = result.answer
            item["contexts"] = [doc.content for doc in result.sources]

    asyncio.run(fill_responses())
    return Dataset.from_list(EVAL_DATASET)


@pytest.mark.evaluation
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_rag_faithfulness(evaluated_dataset):
    """LLM answers must be grounded in retrieved context."""
    result = evaluate(evaluated_dataset, metrics=[faithfulness])
    score = result["faithfulness"]
    assert score >= THRESHOLDS["faithfulness"], (
        f"Faithfulness {score:.2f} below threshold {THRESHOLDS['faithfulness']}"
    )


@pytest.mark.evaluation
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_rag_answer_relevancy(evaluated_dataset):
    """Answers must be relevant to the question asked."""
    result = evaluate(evaluated_dataset, metrics=[answer_relevancy])
    score = result["answer_relevancy"]
    assert score >= THRESHOLDS["answer_relevancy"], (
        f"Answer relevancy {score:.2f} below threshold {THRESHOLDS['answer_relevancy']}"
    )


@pytest.mark.evaluation
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_rag_context_precision(evaluated_dataset):
    """Retrieved contexts must be precise (relevant to question)."""
    result = evaluate(evaluated_dataset, metrics=[context_precision])
    score = result["context_precision"]
    assert score >= THRESHOLDS["context_precision"], (
        f"Context precision {score:.2f} below threshold {THRESHOLDS['context_precision']}"
    )
```

### Example 6: Snapshot Testing for Prompt Templates

```python
# tests/unit/prompts/test_prompt_snapshots.py
"""Snapshot testing ensures prompt templates don't change unexpectedly.

When a prompt changes, the snapshot must be explicitly updated:
  uv run pytest --snapshot-update tests/unit/prompts/
"""
import json
from pathlib import Path

import pytest

from src.application.prompts.templates import (
    build_rag_prompt,
    build_system_prompt,
    build_summary_prompt,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def load_snapshot(name: str) -> str | None:
    path = SNAPSHOT_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text()
    return None


def save_snapshot(name: str, content: str) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / f"{name}.txt").write_text(content)


def assert_snapshot(name: str, actual: str, update: bool = False):
    """Compare against saved snapshot or create new one."""
    expected = load_snapshot(name)
    if expected is None or update:
        save_snapshot(name, actual)
        return
    assert actual == expected, (
        f"Prompt snapshot '{name}' changed. "
        f"Run with --snapshot-update to accept changes.\n"
        f"Expected:\n{expected}\n\nActual:\n{actual}"
    )


def test_system_prompt_snapshot(request):
    prompt = build_system_prompt(
        role="Senior Python Engineer",
        context="Clean Architecture project",
    )
    update = request.config.getoption("--snapshot-update", default=False)
    assert_snapshot("system_prompt", prompt, update=update)


def test_rag_prompt_includes_context():
    prompt = build_rag_prompt(
        question="What is dependency injection?",
        contexts=["DI is a design pattern...", "Constructor injection is..."],
    )
    # Structural assertions (stable across prompt changes)
    assert "What is dependency injection?" in prompt
    assert "DI is a design pattern" in prompt
    assert "Constructor injection" in prompt


def test_summary_prompt_respects_max_length():
    prompt = build_summary_prompt(
        text="A" * 10000,
        max_tokens=500,
    )
    assert "500" in prompt  # Max token instruction included
    assert len(prompt) < 15000  # Prompt itself is bounded
```

### Example 7: Testcontainers for Integration Tests

```python
# tests/integration/conftest.py
"""Integration test fixtures using testcontainers.

Requires: pip install testcontainers[postgres,redis]
"""
import pytest
import pytest_asyncio

try:
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

import asyncpg
import redis.asyncio as aioredis


@pytest.fixture(scope="session")
def postgres_container():
    """Spin up a PostgreSQL container for the test session."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not installed")

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def redis_container():
    """Spin up a Redis container for the test session."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not installed")

    with RedisContainer("redis:7-alpine") as r:
        yield r


@pytest_asyncio.fixture
async def db_pool(postgres_container):
    """Async connection pool to test PostgreSQL."""
    pool = await asyncpg.create_pool(
        host=postgres_container.get_container_host_ip(),
        port=postgres_container.get_exposed_port(5432),
        user=postgres_container.username,
        password=postgres_container.password,
        database=postgres_container.dbname,
    )

    # Run migrations
    async with pool.acquire() as conn:
        migrations = (Path(__file__).parent.parent.parent / "migrations").glob("*.sql")
        for migration in sorted(migrations):
            await conn.execute(migration.read_text())

    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def redis_client(redis_container):
    """Async Redis client connected to test container."""
    client = aioredis.Redis(
        host=redis_container.get_container_host_ip(),
        port=redis_container.get_exposed_port(6379),
    )
    yield client
    await client.flushall()
    await client.close()
```

### Example 8: Testing FastAPI Endpoints with Auth

```python
# tests/integration/api/test_generate_endpoint.py
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock

from src.interfaces.api.main import create_app
from src.interfaces.api.dependencies import get_llm_client


@pytest_asyncio.fixture
async def authenticated_client():
    """Test client with authentication and mocked LLM."""
    app = create_app()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Test response"
    mock_llm.close = AsyncMock()

    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-jwt-token"},
    ) as client:
        yield client, mock_llm

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_returns_200(authenticated_client):
    client, mock_llm = authenticated_client

    response = await client.post(
        "/api/v1/generate",
        json={"prompt": "Hello", "max_tokens": 100},
    )

    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_validates_input(authenticated_client):
    client, _ = authenticated_client

    # Empty prompt
    response = await client.post(
        "/api/v1/generate",
        json={"prompt": "", "max_tokens": 100},
    )
    assert response.status_code == 422  # Pydantic validation

    # Exceeds max_tokens limit
    response = await client.post(
        "/api/v1/generate",
        json={"prompt": "Hello", "max_tokens": 999999},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_generate_without_auth():
    """Unauthenticated requests should be rejected."""
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/generate",
            json={"prompt": "Hello"},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_generate_handles_llm_error(authenticated_client):
    """LLM errors should return 502, not 500."""
    client, mock_llm = authenticated_client
    mock_llm.generate.side_effect = ConnectionError("LLM provider timeout")

    response = await client.post(
        "/api/v1/generate",
        json={"prompt": "Hello"},
    )

    assert response.status_code in (502, 503)
    assert "error" in response.json()
```

### Example 9: Contract Testing for LLM API

```python
# tests/test_llm_contract.py
import pytest
from pydantic import ValidationError
from src.llm.client import LLMClient
from src.llm.models import LLMResponse

@pytest.mark.asyncio
async def test_llm_response_schema():
    """Validate that LLM response matches expected schema."""
    client = LLMClient(provider="openai", model="gpt-4")
    
    response = await client.generate("Hello")
    
    # Pydantic will raise ValidationError if schema doesn't match
    assert isinstance(response, LLMResponse)
    assert hasattr(response, 'content')
    assert hasattr(response, 'tokens_used')
    assert hasattr(response, 'model')
    assert response.tokens_used > 0

@pytest.mark.asyncio
async def test_llm_handles_timeout():
    """Ensure LLM client respects timeout."""
    client = LLMClient(provider="openai", model="gpt-4", timeout=0.001)
    
    with pytest.raises(TimeoutError):
        await client.generate("Long prompt" * 1000)
```

---

## Anti-Patterns to Avoid

### ❌ Testing Implementation Details
**Problem**: Tests break when refactoring internal logic  
**Example**:
```python
# BAD: Testing private methods
def test_internal_parsing():
    parser = DocumentParser()
    result = parser._parse_internal(data)  # Testing private method
    assert result == expected
```
**Solution**: Test public interface only
```python
# GOOD: Testing public API
def test_document_parsing():
    parser = DocumentParser()
    result = parser.parse(document)  # Test public method
    assert result.title == "Expected Title"
```

### ❌ Flaky Tests
**Problem**: Tests pass/fail randomly due to timing, external dependencies  
**Example**:
```python
# BAD: Depends on external API
@pytest.mark.asyncio
async def test_llm_generation():
    response = await openai.ChatCompletion.create(...)  # Real API call
    assert "expected" in response  # May fail due to network/API issues
```
**Solution**: Mock external dependencies
```python
# GOOD: Mocked external dependency
@pytest.mark.asyncio
async def test_llm_generation(mock_openai):
    mock_openai.return_value = "Mocked response"
    response = await generate_text("prompt")
    assert response == "Mocked response"
```

### ❌ No Assertions
**Problem**: Test runs but doesn't validate anything  
**Example**:
```python
# BAD: No assertions
def test_agent_execution():
    agent = Agent()
    agent.run("task")  # No validation
```
**Solution**: Always assert expected behavior
```python
# GOOD: Clear assertions
def test_agent_execution():
    agent = Agent()
    result = agent.run("task")
    assert result.status == "completed"
    assert len(result.steps) > 0
```

### ❌ Testing Everything with Integration Tests
**Problem**: Slow test suite, hard to debug failures
**Solution**: Follow test pyramid (70% unit, 20% integration, 10% E2E)

### ❌ Shared Mutable State Between Tests
**Problem**: Tests pass individually but fail when run together due to shared state.
```python
# BAD: Module-level mutable state
_cache = {}

def test_add_to_cache():
    _cache["key"] = "value"
    assert _cache["key"] == "value"

def test_cache_is_empty():
    assert len(_cache) == 0  # FAILS — polluted by previous test
```
**Solution**: Use pytest fixtures with proper scope. Always clean up in fixtures.

### ❌ Not Mocking Time in Async Tests
**Problem**: Tests with `asyncio.sleep` or timeouts are slow and flaky.
**Solution**: Use `freezegun` or mock `asyncio.sleep` to make time-dependent tests instant and deterministic.

### ❌ Testing LLM Output Content
**Problem**: Asserting exact LLM response text. Tests break on every model update.
```python
# BAD: Brittle assertion on LLM content
assert response == "Clean Architecture is a software design philosophy..."
```
**Solution**: Assert structure, schema, and constraints — not exact content.
```python
# GOOD: Assert structure and constraints
assert isinstance(response, ArchitectureReview)
assert len(response.findings) > 0
assert all(f.severity in ("Critical", "High", "Medium", "Low") for f in response.findings)
```

### ❌ No Test Markers or Categories
**Problem**: All tests run together. CI is slow because evaluation tests (with real LLM calls) run on every push.
**Solution**: Use pytest markers to categorize tests:
```python
# pytest.ini or pyproject.toml
# [tool.pytest.ini_options]
# markers = [
#     "unit: Fast, isolated unit tests",
#     "integration: Tests with external systems",
#     "evaluation: LLM evaluation (slow, costly)",
#     "e2e: End-to-end tests",
# ]

# CI: uv run pytest -m "not evaluation" (fast)
# Nightly: uv run pytest -m evaluation (thorough)
```

---

## CI Quality Gate Configuration

### pytest configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "unit: Fast isolated unit tests (no I/O)",
    "integration: Tests requiring external systems (DB, Redis, APIs)",
    "evaluation: LLM quality evaluation (RAGAS, DeepEval) — slow and costly",
    "e2e: End-to-end tests — minimal count",
]
filterwarnings = [
    "ignore::DeprecationWarning:langchain.*",
]
addopts = [
    "--strict-markers",
    "--tb=short",
    "-q",
]

[tool.coverage.run]
source = ["src"]
omit = ["src/examples/*", "tests/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.",
]
```

### CI Pipeline Test Steps

```yaml
# .github/workflows/ci.yml (test section)
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4

      # Lint + type check (fast fail)
      - run: uv run ruff check .
      - run: uv run mypy src/

      # Unit tests (always)
      - run: uv run pytest tests/unit/ --cov --cov-fail-under=80 -q

      # Integration tests (on merge to main)
      - if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: uv run pytest tests/integration/ -m integration -q

      # Evaluation tests (scheduled nightly)
      # - run: uv run pytest tests/evaluation/ -m evaluation --timeout=300
```

---

## Testing Checklist

### Pre-Commit Checklist
- [ ] All tests pass locally (`pytest`)
- [ ] Code coverage ≥ 80% (`pytest --cov`)
- [ ] No linting errors (`ruff check .`)
- [ ] Type checking passes (`mypy src/`)
- [ ] No security issues (`bandit -r src/`)

### Unit Testing Checklist
- [ ] Each function has at least one test
- [ ] Edge cases are tested (empty input, None, large values)
- [ ] Error handling is tested (exceptions, timeouts)
- [ ] Mocks are used for external dependencies
- [ ] Tests are fast (< 1s per test)

### Integration Testing Checklist
- [ ] RAG pipeline tested end-to-end
- [ ] Agent workflows tested with real state
- [ ] Database interactions tested (with test DB)
- [ ] API endpoints tested (with TestClient)
- [ ] Error scenarios tested (network failures, timeouts)

### LLM Testing Checklist
- [ ] Prompt templates validated
- [ ] LLM responses conform to schema (Pydantic)
- [ ] Evaluation metrics measured (RAGAS, DeepEval)
- [ ] Hallucination detection tested
- [ ] Cost tracking validated

---

## Instructions for the Agent

1. **Unit Testing**:
   - Mock all external dependencies (LLMs, DBs, APIs)
   - Use `pytest.fixture` for reusable test setup
   - Use `pytest.mark.asyncio` for async tests
   - Aim for 80%+ code coverage

2. **Integration Testing**:
   - Use test databases (SQLite, Docker containers)
   - Use `TestClient` for FastAPI endpoints
   - Test complete workflows (RAG query, agent execution)
   - Clean up resources after tests

3. **LLM Testing**:
   - Use LangSmith for tracing and evaluation
   - Validate outputs with Pydantic models
   - Measure evaluation metrics (faithfulness, relevancy)
   - Test prompt injection scenarios

4. **Property-Based Testing**:
   - Use Hypothesis for complex input validation
   - Test invariants (properties that should always hold)
   - Find edge cases automatically

5. **Quality Gates**:
   - Run `ruff check .` before commit
   - Run `mypy src/` for type safety
   - Run `pytest --cov` for coverage
   - Run `bandit -r src/` for security

6. **Test Organization**:
   - Mirror `src/` structure in `tests/`
   - Use descriptive test names (`test_<what>_<when>_<expected>`)
   - Group related tests with classes
   - Use fixtures for common setup

7. **Continuous Testing**:
   - Run tests on every commit (CI/CD)
   - Fail builds on test failures
   - Track test coverage over time
   - Monitor test execution time
