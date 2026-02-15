# Testing

## Estrategia

```
                    ┌─────────────┐
                    │   E2E Tests  │  Pocos, lentos, costosos
                   ┌┴─────────────┴┐
                   │ Integration    │  Agentes, pipelines RAG
                  ┌┴───────────────┴┐
                  │  Contract Tests  │  LLM mocks con contratos
                 ┌┴─────────────────┴┐
                 │    Unit Tests      │  Domain logic, funciones puras
                 └───────────────────┘
```

---

## Unit Tests

Domain logic pura. Sin LLMs, sin I/O, sin side effects.

```python
# tests/unit/test_domain.py
def test_document_chunk_respects_max_length():
    doc = Document(content="a" * 1000)
    chunks = doc.chunk(max_length=200, overlap=50)
    assert all(len(c.content) <= 200 for c in chunks)
    assert len(chunks) == 6
```

---

## Contract Tests (LLM Mocks)

Mockear LLMs con contratos que validan estructura, no contenido exacto.

```python
# tests/contract/test_llm_contract.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMPort)
    llm.generate_structured.return_value = Summary(
        title="Test", points=["Point 1"], confidence=0.9
    )
    return llm

async def test_summarize_returns_valid_structure(mock_llm):
    use_case = SummarizeUseCase(llm=mock_llm)
    result = await use_case.execute("test query")
    assert isinstance(result, Summary)
    assert result.confidence > 0
    mock_llm.generate_structured.assert_called_once()
```

---

## Prompt Tests

Validar que los prompts generados tienen la estructura e intención correcta.

```python
# tests/prompts/test_summary_prompt.py
def test_summary_prompt_includes_context():
    prompt = build_summary_prompt(
        query="AI trends",
        context=[Document(content="LLMs are evolving...")],
    )
    assert "AI trends" in prompt
    assert "LLMs are evolving" in prompt
    assert "summary" in prompt.lower() or "summarize" in prompt.lower()

def test_summary_prompt_enforces_output_format():
    prompt = build_summary_prompt(query="test", context=[])
    assert "JSON" in prompt or "json" in prompt
```

---

## Integration Tests

Testear flujos completos con servicios reales (o containers).

```python
# tests/integration/test_rag_pipeline.py
@pytest.mark.integration
async def test_rag_pipeline_returns_grounded_answer():
    pipeline = create_rag_pipeline()
    result = await pipeline.query("What is Python?")
    assert result.answer is not None
    assert len(result.sources) > 0
    assert result.confidence > 0.5
```

---

## Evaluación de LLMs

Tests especializados para medir calidad de outputs de LLM. Ver [EVALUATION.md](EVALUATION.md) para detalle completo.

```python
# tests/evaluation/test_faithfulness.py
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_answer_is_faithful_to_context():
    test_case = LLMTestCase(
        input="What is RAG?",
        actual_output="RAG is Retrieval-Augmented Generation...",
        retrieval_context=["RAG combines retrieval with generation..."],
    )
    metric = FaithfulnessMetric(threshold=0.7)
    metric.measure(test_case)
    assert metric.score >= 0.7
```

---

## Fixtures y Configuración

```python
# conftest.py
import pytest

@pytest.fixture
def sample_documents():
    return [
        Document(content="Python is a programming language", metadata={"source": "wiki"}),
        Document(content="FastAPI is a web framework", metadata={"source": "docs"}),
    ]

@pytest.fixture
def mock_vector_store(sample_documents):
    store = AsyncMock(spec=VectorStorePort)
    store.search.return_value = sample_documents
    return store
```

---

## Estructura de Tests

```
tests/
├── conftest.py              # Fixtures compartidas
├── unit/                    # Tests unitarios (rápidos, sin I/O)
│   ├── domain/
│   └── application/
├── contract/                # Contract tests con mocks
├── integration/             # Tests con servicios reales
│   ├── conftest.py
│   └── test_rag_pipeline.py
├── prompts/                 # Tests de prompts
├── evaluation/              # Evaluación de LLMs
└── e2e/                     # Tests end-to-end
```

---

## Comandos

```bash
# Todos los tests
uv run pytest

# Solo unitarios (rápidos)
uv run pytest tests/unit/ -v

# Con coverage
uv run pytest --cov=src --cov-report=html

# Solo marcados como integration
uv run pytest -m integration

# Evaluación de LLMs (requiere API keys)
uv run pytest tests/evaluation/ -m evaluation
```

---

## Reglas

1. **Domain logic siempre tiene unit tests**
2. **LLM interactions se mockean** en tests unitarios y de contrato
3. **Prompts se validan** por estructura e intención, no por texto exacto
4. **Integration tests son opcionales** en CI rápido, obligatorios en merge
5. **Evaluation tests se ejecutan periódicamente**, no en cada commit
6. **No depender de output exacto del LLM** — validar estructura y constraints

Ver también: [EVALUATION.md](EVALUATION.md), [TOOLS.md](TOOLS.md)
