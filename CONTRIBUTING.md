# Contributing

## Setup

```bash
git clone <repo-url>
cd genai-python-template
uv sync --all-extras --dev
uv run pre-commit install
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/short-description`
2. Make changes following Clean Architecture conventions
3. Run quality checks before committing:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv run pytest tests/unit/ -q
```

4. Commit with a descriptive message
5. Open a pull request against `main`

## Code Standards

- Follow SOLID principles and Clean Architecture layers
- All domain logic must be framework-free
- Infrastructure adapters implement domain ports
- Type annotations on all public functions
- Tests required for new features and bug fixes
- Coverage must remain above 80%

## Project Structure

```
src/
  domain/         # Pure business logic, no imports from infrastructure
  application/    # Use cases, DTOs, pipelines
  infrastructure/ # Adapters for external systems (LLM, DB, cloud)
  interfaces/     # API routes, CLI commands, middleware
tests/
  unit/           # Fast, isolated tests
  integration/    # Tests with external dependencies (mocked or real)
```

## Pull Request Checklist

- [ ] Ruff check passes (`uv run ruff check src/ tests/`)
- [ ] Type check passes (`uv run mypy src/`)
- [ ] All tests pass (`uv run pytest tests/ -q`)
- [ ] New code has tests
- [ ] No secrets or credentials in code
