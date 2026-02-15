# Skill: Testing & Quality

## Description
This skill provides comprehensive testing strategies and quality assurance practices for production GenAI applications. Use this when writing tests, ensuring code quality, or validating LLM outputs.

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

### Example 3: Contract Testing for LLM API

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
