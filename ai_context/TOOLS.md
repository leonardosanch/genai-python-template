# Herramientas

## uv — Package Manager

Herramienta principal para gestión de dependencias y entornos virtuales.

```bash
# Instalar
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear proyecto
uv init

# Instalar dependencias
uv sync

# Agregar dependencia
uv add langchain openai

# Agregar dependencia de desarrollo
uv add --dev pytest ruff mypy

# Ejecutar comando en el entorno virtual
uv run pytest
uv run python -m src.interfaces.cli.main

# Lock dependencies
uv lock

# Actualizar dependencias
uv lock --upgrade
```

**Por qué uv:**
- 10-100x más rápido que pip
- Lock file nativo (`uv.lock`)
- Gestión de virtualenv integrada
- Compatible con `pyproject.toml` estándar
- Reemplaza pip, pip-tools, venv, virtualenv

---

## ruff — Linter y Formatter

Reemplaza flake8, black, isort, pyflakes, pycodestyle en una sola herramienta.

```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "S",    # flake8-bandit (security)
    "A",    # flake8-builtins
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "RUF",  # ruff-specific rules
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]  # Allow assert in tests
```

```bash
# Lint
uv run ruff check .

# Lint con autofix
uv run ruff check --fix .

# Format
uv run ruff format .

# Check format without modifying
uv run ruff format --check .
```

---

## mypy — Type Checking

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

```bash
uv run mypy src/
```

---

## pytest — Testing

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: integration tests (require external services)",
    "evaluation: LLM evaluation tests (require API keys)",
]
addopts = "-v --tb=short"
```

```bash
# Todos los tests
uv run pytest

# Con coverage
uv run pytest --cov=src --cov-report=html --cov-report=xml

# Solo unitarios
uv run pytest tests/unit/

# Excluir integration
uv run pytest -m "not integration"
```

---

## pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-added-large-files
      - id: check-yaml
      - id: detect-private-key
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

```bash
# Instalar hooks
uv run pre-commit install

# Ejecutar manualmente
uv run pre-commit run --all-files
```

---

## SonarQube / SonarCloud

Análisis estático de código: bugs, vulnerabilities, code smells, coverage, duplicación.

```properties
# sonar-project.properties
sonar.projectKey=genai-python-template
sonar.sources=src
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.version=3.12
```

**Quality Gates recomendados:**
- Coverage > 80%
- No critical/blocker issues
- Duplicación < 3%
- Complejidad ciclomática por función < 10
- Maintainability rating A

---

## Complejidad Ciclomática

Medir y limitar la complejidad del código.

| Complejidad | Clasificación | Acción |
|-------------|--------------|--------|
| 1-5 | Simple | OK |
| 6-10 | Moderada | Revisar si se puede simplificar |
| 11-20 | Compleja | Refactorizar obligatorio |
| 21+ | Muy compleja | Dividir inmediatamente |

```bash
# Medir con radon
uv run radon cc src/ -s -a

# ruff también detecta complejidad
# En pyproject.toml:
# [tool.ruff.lint]
# select = ["C901"]  # McCabe complexity
# [tool.ruff.lint.mccabe]
# max-complexity = 10
```

---

## Makefile

```makefile
.PHONY: install lint test format check

install:
	uv sync

lint:
	uv run ruff check .
	uv run mypy src/

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest tests/unit/ -v

test-all:
	uv run pytest --cov=src --cov-report=html

check: lint test

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache htmlcov .ruff_cache
```

Ver también: [DEPLOYMENT.md](DEPLOYMENT.md), [TESTING.md](TESTING.md)
