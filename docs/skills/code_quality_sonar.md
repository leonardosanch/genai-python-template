# Skill: Code Quality, Code Smells & SonarQube

## Description
This skill covers static analysis, code smell detection, SonarQube/SonarCloud configuration, quality metrics, and refactoring strategies for Python backend and GenAI systems. Use this when configuring quality gates, analyzing technical debt, eliminating code smells, or integrating SonarQube into CI/CD.

## Executive Summary

**Critical quality rules:**
- **Cyclomatic complexity per function: max 10** — Enforce with ruff C901 and SonarQube. Functions >10 must be refactored
- **Cognitive complexity per function: max 15** — SonarQube's metric; better than cyclomatic for readability assessment
- **Duplication < 3%** — Measured by SonarQube; extract common logic into shared modules
- **Zero critical/blocker issues** — Quality gate must block merge on any critical finding
- **Coverage > 80%** — Lines and branches; exclude generated code and migrations
- **Technical debt ratio < 5%** — Ratio of remediation cost to development cost

**Read full skill when:** Configuring SonarQube/SonarCloud, setting up quality gates, eliminating code smells, measuring technical debt, integrating static analysis in CI/CD, or reviewing code quality metrics.

---

## Versiones y Herramientas

| Herramienta | Versión Mínima | Tipo | Notas |
|-------------|----------------|------|-------|
| SonarQube CE | >= 10.0 | Static analysis platform | Self-hosted, free community edition |
| SonarCloud | SaaS | Static analysis platform | Free for open source |
| sonar-scanner | >= 5.0 | CLI scanner | Ejecuta análisis local o en CI |
| ruff | >= 0.1.0 | Linter + formatter | Reemplaza flake8, isort, pycodestyle |
| radon | >= 6.0.1 | Complexity analyzer | Cyclomatic complexity, Halstead metrics |
| vulture | >= 2.10 | Dead code finder | Detecta código no utilizado |
| bandit | >= 1.7.5 | Security linter | SAST para Python |
| mypy | >= 1.5.0 | Type checker | Static type analysis |
| xenon | >= 0.9.1 | Complexity enforcer | Wrapper sobre radon para CI |

---

## Core Concepts

1. **Code Smell**: Indicador superficial de un problema de diseño más profundo. No es un bug, pero incrementa el costo de mantenimiento.
2. **Technical Debt**: Costo futuro de elegir una solución rápida hoy. SonarQube lo mide en tiempo de remediación.
3. **Cognitive Complexity**: Métrica de SonarQube que mide cuán difícil es *entender* una función (mejor que cyclomatic para legibilidad).
4. **Quality Gate**: Conjunto de condiciones que el código debe cumplir para ser promovido (merge/deploy).
5. **Quality Profile**: Conjunto de reglas activas en SonarQube para un lenguaje específico.

---

## Decision Trees

### Decision Tree 1: Qué herramienta usar

```
¿Qué necesitas analizar?
├── Análisis integral (smells + bugs + security + coverage + deuda)
│   ├── ¿Open source o equipo pequeño? → SonarCloud (gratis)
│   └── ¿Self-hosted o enterprise? → SonarQube
├── Solo linting y formato
│   └── ruff (reemplaza flake8 + isort + pycodestyle + pydocstyle)
├── Solo complejidad ciclomática
│   └── radon (análisis) + xenon (enforcement en CI)
├── Solo código muerto
│   └── vulture
├── Solo seguridad (SAST)
│   └── bandit + ruff (reglas S)
└── Solo tipos
    └── mypy --strict
```

### Decision Tree 2: Qué hacer con un code smell

```
¿Qué smell detectaste?
├── Función/método largo (>20 líneas)
│   └── Extract Method → dividir en funciones con nombre descriptivo
├── Clase dios (>200 líneas o >5 responsabilidades)
│   └── Extract Class → separar por responsabilidad (SRP)
├── Feature Envy (método usa más datos de otra clase)
│   └── Move Method → mover a la clase que posee los datos
├── Código duplicado
│   ├── ¿Mismo módulo? → Extract Method
│   ├── ¿Misma jerarquía? → Pull Up Method / Extract Superclass
│   └── ¿Módulos distintos? → Extract a shared utility
├── Primitive Obsession (strings/ints como conceptos de dominio)
│   └── Introduce Value Object (dataclass o Pydantic model)
├── Data Clumps (grupos de datos que viajan juntos)
│   └── Introduce Parameter Object (dataclass)
├── Shotgun Surgery (un cambio toca muchas clases)
│   └── Move Method + Inline Class → consolidar responsabilidad
├── Switch/if-else chains (>3 ramas por tipo)
│   └── Replace Conditional with Polymorphism (Strategy pattern)
├── Speculative Generality (abstracción sin uso real)
│   └── Inline Class / Collapse Hierarchy → eliminar
└── Dead Code (imports, funciones, variables sin uso)
    └── Delete → vulture + ruff F841/F401 detectan automáticamente
```

---

## SonarQube / SonarCloud Configuration

### Project Properties

```properties
# sonar-project.properties
sonar.projectKey=my-project
sonar.projectName=My Project
sonar.projectVersion=1.0.0

# Sources
sonar.sources=src
sonar.tests=tests
sonar.python.version=3.12

# Coverage
sonar.python.coverage.reportPaths=coverage.xml

# Exclusions
sonar.exclusions=\
  **/migrations/**,\
  **/alembic/**,\
  **/__pycache__/**,\
  **/node_modules/**,\
  **/static/**,\
  **/*.generated.py

# Test exclusions (no medir coverage de tests)
sonar.coverage.exclusions=\
  tests/**,\
  **/conftest.py,\
  **/fixtures/**

# Duplication exclusions
sonar.cpd.exclusions=\
  **/migrations/**,\
  **/*_pb2.py

# Encoding
sonar.sourceEncoding=UTF-8
```

### Quality Gate — Production Ready

| Métrica | Condición | Valor |
|---------|-----------|-------|
| Coverage on new code | >= | 80% |
| Duplicated lines on new code | <= | 3% |
| Maintainability rating | = | A |
| Reliability rating | = | A |
| Security rating | = | A |
| Security hotspots reviewed | = | 100% |
| Blocker issues | = | 0 |
| Critical issues | = | 0 |

### Quality Profile — Python Recomendado

Reglas clave a activar (sobre el perfil default "Sonar way"):

```
# Complexity
python:S3776    Cognitive complexity max 15
python:S1541    Cyclomatic complexity max 10

# Code Smells
python:S107     Too many parameters (max 5)
python:S1192    String literals duplicated (max 3)
python:S1871    Identical branches in if/else
python:S1481    Unused local variable
python:S1135    Track TODO/FIXME comments
python:S125     Remove commented-out code

# Bugs
python:S5727    Comparison to None should use 'is'
python:S5719    Raising non-exception instances
python:S930     Function with incorrect number of arguments

# Security
python:S4423    Weak SSL/TLS protocols
python:S2077    SQL injection
python:S4790    Weak hashing algorithms
python:S5131    XSS vulnerabilities

# Naming
python:S117     Local variable naming convention
python:S100     Function naming convention
```

---

## SonarQube Metrics Deep Dive

### Reliability (Bugs)

| Rating | Definición |
|--------|-----------|
| A | 0 bugs |
| B | >= 1 minor bug |
| C | >= 1 major bug |
| D | >= 1 critical bug |
| E | >= 1 blocker bug |

### Security

| Rating | Definición |
|--------|-----------|
| A | 0 vulnerabilities |
| B | >= 1 minor vulnerability |
| C | >= 1 major vulnerability |
| D | >= 1 critical vulnerability |
| E | >= 1 blocker vulnerability |

### Maintainability (Technical Debt)

| Rating | Technical Debt Ratio |
|--------|---------------------|
| A | <= 5% |
| B | 6-10% |
| C | 11-20% |
| D | 21-50% |
| E | > 50% |

**Technical Debt Ratio** = Remediation Cost / Development Cost

### Cognitive Complexity vs Cyclomatic Complexity

| Aspecto | Cyclomatic | Cognitive |
|---------|-----------|-----------|
| Mide | Caminos de ejecución | Dificultad de comprensión |
| Penaliza nesting | No | Sí (incrementalmente) |
| switch/match | +1 por case | +1 total (no por case) |
| Recursión | No cuenta | +1 (más difícil de entender) |
| break/continue | No cuenta | +1 |
| Uso principal | Testing (paths) | Readability |
| **Threshold** | **max 10** | **max 15** |

```python
# Cognitive complexity: 1 (simple)
def is_eligible(age: int) -> bool:
    if age >= 18:  # +1
        return True
    return False

# Cognitive complexity: 8 (nesting penalty)
def process(user: User) -> str:
    if user.is_active:          # +1
        if user.has_subscription:  # +2 (nesting=1)
            if user.plan == "pro":   # +3 (nesting=2)
                return "full_access"
            else:                    # +1
                return "limited"
        else:                      # +1
            return "free"
    return "inactive"

# Refactored: cognitive complexity 3
def process(user: User) -> str:
    if not user.is_active:     # +1
        return "inactive"
    if not user.has_subscription:  # +1
        return "free"
    if user.plan == "pro":     # +1
        return "full_access"
    return "limited"
```

---

## Code Smells Catalog — Python Específicos

### 1. God Class / Blob

```python
# ❌ SMELL: Clase con demasiadas responsabilidades
class OrderService:
    def create_order(self, data): ...
    def send_confirmation_email(self, order): ...
    def charge_payment(self, order): ...
    def update_inventory(self, order): ...
    def generate_invoice_pdf(self, order): ...
    def notify_warehouse(self, order): ...
    def calculate_shipping(self, order): ...
    def apply_discount(self, order, code): ...

# ✅ REFACTORED: Single Responsibility
class OrderService:
    def __init__(
        self,
        payment: PaymentPort,
        inventory: InventoryPort,
        notifications: NotificationPort,
    ):
        self._payment = payment
        self._inventory = inventory
        self._notifications = notifications

    async def create_order(self, data: CreateOrderDTO) -> Order:
        order = Order.from_dto(data)
        await self._payment.charge(order)
        await self._inventory.reserve(order.items)
        await self._notifications.order_created(order)
        return order
```

### 2. Primitive Obsession

```python
# ❌ SMELL: Strings como conceptos de dominio
def create_user(email: str, role: str, status: str) -> dict:
    if role not in ("admin", "user", "viewer"):
        raise ValueError("Invalid role")
    ...

# ✅ REFACTORED: Value Objects
from enum import StrEnum
from pydantic import EmailStr

class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class UserStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"

def create_user(email: EmailStr, role: UserRole, status: UserStatus) -> User:
    ...
```

### 3. Long Parameter List

```python
# ❌ SMELL: Demasiados parámetros
def search_products(
    query: str, category: str, min_price: float, max_price: float,
    brand: str, color: str, size: str, sort_by: str, page: int, limit: int,
) -> list[Product]:
    ...

# ✅ REFACTORED: Parameter Object
from dataclasses import dataclass

@dataclass(frozen=True)
class ProductSearchCriteria:
    query: str
    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    brand: str | None = None
    color: str | None = None
    size: str | None = None
    sort_by: str = "relevance"
    page: int = 1
    limit: int = 20

def search_products(criteria: ProductSearchCriteria) -> list[Product]:
    ...
```

### 4. Feature Envy

```python
# ❌ SMELL: Método que usa más datos de otra clase
class InvoiceService:
    def calculate_total(self, order: Order) -> Decimal:
        subtotal = sum(item.price * item.quantity for item in order.items)
        tax = subtotal * order.tax_rate
        discount = subtotal * order.discount_rate
        return subtotal + tax - discount

# ✅ REFACTORED: Mover lógica al dueño de los datos
class Order:
    def calculate_total(self) -> Decimal:
        subtotal = sum(item.price * item.quantity for item in self.items)
        tax = subtotal * self.tax_rate
        discount = subtotal * self.discount_rate
        return subtotal + tax - discount
```

### 5. Shotgun Surgery (GenAI specific)

```python
# ❌ SMELL: Cambiar el modelo LLM requiere editar 10 archivos
# file1.py
response = openai.chat.completions.create(model="gpt-4", ...)
# file2.py
response = openai.chat.completions.create(model="gpt-4", ...)
# file3.py
response = openai.chat.completions.create(model="gpt-4", ...)

# ✅ REFACTORED: Adapter pattern — un solo lugar para cambiar
class LLMPort(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...

class OpenAIAdapter:
    def __init__(self, model: str = "gpt-4"):
        self._model = model
        self._client = AsyncOpenAI()

    async def complete(self, messages: list[Message]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
        )
        return response.choices[0].message.content
```

### 6. Dead Code

```python
# ❌ SMELL: Código comentado, imports sin usar, funciones huérfanas
import os  # never used
import json  # never used

# def old_implementation():
#     """This was the old way"""
#     pass

class UserService:
    def _legacy_method(self):  # never called
        ...

# ✅ DETECTION: Herramientas automáticas
# ruff F401 — unused imports
# ruff F841 — unused variables
# vulture — unused functions, classes, variables
```

---

## CI/CD Integration

### GitHub Actions — SonarCloud

```yaml
# .github/workflows/sonar.yml
name: SonarCloud Analysis

on:
  push:
    branches: [main, develop]
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  sonarcloud:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for blame

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run tests with coverage
        run: |
          uv run pytest --cov=src --cov-report=xml:coverage.xml \
            --junitxml=test-results.xml

      - name: Run ruff
        run: uv run ruff check src/ --output-format=json > ruff-report.json || true

      - name: SonarCloud Scan
        uses: SonarSource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        with:
          args: >
            -Dsonar.python.coverage.reportPaths=coverage.xml
            -Dsonar.python.xunit.reportPath=test-results.xml
```

### GitHub Actions — Standalone Quality (sin SonarQube)

```yaml
# .github/workflows/quality.yml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install
        run: |
          pip install uv
          uv sync

      - name: Lint (ruff)
        run: uv run ruff check src/ tests/

      - name: Type check (mypy)
        run: uv run mypy src/

      - name: Complexity check (xenon)
        run: uv run xenon src/ --max-absolute B --max-modules A --max-average A

      - name: Dead code (vulture)
        run: uv run vulture src/ --min-confidence 80

      - name: Security (bandit)
        run: uv run bandit -r src/ -c pyproject.toml

      - name: Tests + coverage
        run: |
          uv run pytest --cov=src --cov-fail-under=80 --cov-report=term-missing
```

---

## Measuring Complexity — Local Commands

```bash
# Cyclomatic complexity — por función
uv run radon cc src/ -s -a -nc

# Maintainability index (A-F rating)
uv run radon mi src/ -s

# Halstead metrics (effort, difficulty, volume)
uv run radon hal src/

# Enforce complexity in CI (fail if any function > B)
uv run xenon src/ --max-absolute B --max-modules A --max-average A

# Dead code detection
uv run vulture src/ --min-confidence 80

# ruff complexity rule
# pyproject.toml → [tool.ruff.lint.mccabe] max-complexity = 10
```

### Radon Complexity Grades

| Grade | CC Range | Risk |
|-------|----------|------|
| A | 1-5 | Low — simple, well-tested |
| B | 6-10 | Low — moderate, testable |
| C | 11-15 | Moderate — review recommended |
| D | 16-25 | High — refactor required |
| E | 26-50 | Very High — error prone |
| F | 51+ | Critical — untestable |

### Radon Maintainability Index

| Grade | MI Range | Interpretation |
|-------|----------|---------------|
| A | 20-100 | Highly maintainable |
| B | 10-19 | Moderately maintainable |
| C | 0-9 | Difficult to maintain |

---

## pyproject.toml Configuration

```toml
[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes (unused imports, variables)
    "C901", # McCabe cyclomatic complexity
    "S",    # bandit security rules
    "B",    # bugbear (common pitfalls)
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "I",    # isort
    "SIM",  # simplify (code simplification)
    "RUF",  # ruff-specific rules
]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]  # Allow assert in tests
"**/migrations/**" = ["E501"]  # Long lines OK in migrations

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # Skip assert warnings

[tool.vulture]
min_confidence = 80
paths = ["src"]
exclude = ["migrations", "__pycache__"]
```

---

## SonarLint — IDE Integration

Configurar SonarLint para detección temprana (antes de push):

| IDE | Plugin | Connected Mode |
|-----|--------|---------------|
| VS Code | SonarLint extension | Sí — sincroniza reglas con SonarCloud/SonarQube |
| PyCharm | SonarLint plugin | Sí — sincroniza quality profile |
| Vim/Neovim | sonarlint.nvim (LSP) | Limitado |

**Connected Mode**: Sincroniza las reglas del servidor (quality profile) con el IDE. Garantiza que el desarrollador vea los mismos issues que CI.

```json
// .vscode/settings.json
{
  "sonarlint.connectedMode.project": {
    "connectionId": "my-sonarcloud",
    "projectKey": "my-project"
  }
}
```

---

## SonarQube vs SonarCloud

| Aspecto | SonarQube (CE) | SonarCloud |
|---------|---------------|------------|
| Hosting | Self-hosted | SaaS |
| Precio | Free (CE), paid (DE/EE) | Free open source, paid private |
| CI Integration | Cualquier CI | GitHub, GitLab, Azure DevOps, Bitbucket |
| Branch analysis | Solo paid editions | Incluido |
| PR decoration | Solo paid editions | Incluido |
| Custom rules | Sí | No |
| Plugins | Sí | No |
| **Recomendación** | Enterprise / on-prem | **Default para proyectos nuevos** |

---

## Anti-Patterns to Avoid

### 1. Suppressing Without Justification
```python
# ❌ Suppress sin razón
# noqa: C901
def mega_function(): ...

# ✅ Suppress con justificación documentada
# noqa: C901 — State machine requires sequential evaluation, refactoring would reduce readability
def state_machine_handler(): ...
```

### 2. Gaming Coverage
```python
# ❌ Tests que tocan código sin verificar comportamiento
def test_create_user():
    create_user("test@test.com", "admin", "active")
    # no assertions — coverage sube pero no valida nada

# ✅ Tests que verifican comportamiento
def test_create_user_assigns_role():
    user = create_user("test@test.com", UserRole.ADMIN, UserStatus.ACTIVE)
    assert user.role == UserRole.ADMIN
    assert user.is_active is True
```

### 3. Ignoring Security Hotspots
```python
# ❌ Marcar "Safe" en SonarQube sin revisar
# Security Hotspot: "Make sure using unverified input is safe"
query = f"SELECT * FROM users WHERE name = '{user_input}'"  # SQL injection

# ✅ Revisar y remediar
query = "SELECT * FROM users WHERE name = :name"
result = await db.execute(text(query), {"name": user_input})
```

---

## External Resources

### SonarQube / SonarCloud
- **SonarQube Docs**: [docs.sonarsource.com/sonarqube](https://docs.sonarsource.com/sonarqube/latest/)
    - *Best for*: Server setup, quality profiles, quality gates, administration
- **SonarCloud Docs**: [docs.sonarsource.com/sonarcloud](https://docs.sonarsource.com/sonarcloud/)
    - *Best for*: SaaS setup, GitHub integration, PR decoration
- **Python Rules**: [rules.sonarsource.com/python](https://rules.sonarsource.com/python/)
    - *Best for*: Complete catalog of Python rules with examples
- **Cognitive Complexity Whitepaper**: [sonarsource.com/resources/cognitive-complexity](https://www.sonarsource.com/resources/cognitive-complexity/)
    - *Best for*: Understanding cognitive vs cyclomatic complexity

### Code Smells & Refactoring
- **Refactoring Guru — Code Smells**: [refactoring.guru/refactoring/smells](https://refactoring.guru/refactoring/smells)
    - *Best for*: Complete catalog with examples and solutions
- **Refactoring Guru — Refactoring Catalog**: [refactoring.guru/refactoring/catalog](https://refactoring.guru/refactoring/catalog)
    - *Best for*: Specific refactoring techniques
- **Martin Fowler — Refactoring**: [refactoring.com](https://refactoring.com/)
    - *Best for*: Original refactoring patterns

### Python Quality Tools
- **Ruff**: [docs.astral.sh/ruff](https://docs.astral.sh/ruff/)
    - *Best for*: Fast Python linting (replaces flake8, isort, pycodestyle)
- **Radon**: [radon.readthedocs.io](https://radon.readthedocs.io/)
    - *Best for*: Complexity metrics (cyclomatic, Halstead, maintainability index)
- **Vulture**: [github.com/jendrikseipp/vulture](https://github.com/jendrikseipp/vulture)
    - *Best for*: Dead code detection
- **Xenon**: [xenon.readthedocs.io](https://xenon.readthedocs.io/)
    - *Best for*: Complexity enforcement in CI (wrapper over radon)
- **Bandit**: [bandit.readthedocs.io](https://bandit.readthedocs.io/)
    - *Best for*: Security-focused static analysis for Python

---

## Checklists

### Pre-Merge Quality Checklist
- [ ] `uv run ruff check src/` — 0 errors
- [ ] `uv run mypy src/` — 0 errors
- [ ] `uv run pytest --cov=src --cov-fail-under=80` — coverage met
- [ ] `uv run radon cc src/ -nc` — no function above C (>15)
- [ ] `uv run vulture src/ --min-confidence 80` — no dead code
- [ ] SonarQube quality gate — passed (if configured)

### Code Review Quality Checklist
- [ ] No god classes (>200 lines or >5 responsibilities)
- [ ] No functions with >5 parameters (use Parameter Object)
- [ ] No duplicated logic (>3 occurrences)
- [ ] No primitive obsession (domain concepts as strings/ints)
- [ ] No feature envy (method using other class's data)
- [ ] No commented-out code
- [ ] No unused imports/variables
- [ ] All suppressions (`noqa`, `type: ignore`) have justification
