# Skill: Software Architecture

## Description
This skill provides guidelines and external resources for making high-quality architectural decisions. Use this when designing new components, refactoring legacy code, or evaluating system trade-offs.

## Core Principles

1.  **Clean Architecture**: Always separate concerns. Domain logic must never depend on infrastructure.
2.  **SOLID**: Apply SRP, OCP, LSP, ISP, and DIP rigorously.
3.  **12-Factor App**: Build cloud-native applications suitable for deployment on modern platforms.
4.  **C4 Model**: When visualizing architecture, use the C4 model (Context, Containers, Components, Code).

---

## External Resources

### 📚 Essential Books & PDFs

#### Architecture Fundamentals
- **Clean Architecture** (Robert C. Martin)
    - [Book Overview](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
    - *Best for*: Separation of concerns, dependency rule, domain-centric design
- **Domain-Driven Design** (Eric Evans)
    - [DDD Reference PDF](https://www.domainlanguage.com/ddd/reference/)
    - *Best for*: Bounded contexts, aggregates, ubiquitous language
- **Patterns of Enterprise Application Architecture** (Martin Fowler)
    - [Catalog](https://martinfowler.com/eaaCatalog/)
    - *Best for*: Repository, Unit of Work, Service Layer patterns

#### Distributed Systems
- **Designing Data-Intensive Applications** (Martin Kleppmann)
    - [Book Site](https://dataintensive.net/)
    - *Best for*: Replication, partitioning, consistency, consensus algorithms
- **Building Microservices** (Sam Newman)
    - [2nd Edition Resources](https://samnewman.io/books/building_microservices_2nd_edition/)
    - *Best for*: Microservices decomposition, communication patterns, deployment

#### System Design
- **System Design Primer** (GitHub)
    - [github.com/donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer)
    - *Best for*: Scalability, load balancing, caching, databases
- **The Architecture of Open Source Applications**
    - [aosabook.org](https://aosabook.org/en/index.html)
    - *Best for*: Real-world architecture examples (LLVM, nginx, SQLAlchemy)

---

### 🌐 Authoritative Websites

#### General Architecture
- **Martin Fowler's Architecture Guide**: [martinfowler.com/architecture/](https://martinfowler.com/architecture/)
    - *Best for*: Microservices, event-driven architecture, serverless
- **Refactoring Guru**: [refactoring.guru](https://refactoring.guru/)
    - *Best for*: Design patterns (Gang of Four), code smells, refactoring catalog
- **The Twelve-Factor App**: [12factor.net](https://12factor.net/)
    - *Best for*: Cloud-native best practices (config, backing services, disposability)

#### Microservices & Distributed Systems
- **Microservices Patterns** (Chris Richardson): [microservices.io](https://microservices.io/)
    - *Best for*: Saga pattern, API gateway, circuit breaker, event sourcing
- **Microsoft Azure Architecture Center**: [learn.microsoft.com/en-us/azure/architecture/](https://learn.microsoft.com/en-us/azure/architecture/)
    - *Best for*: Cloud design patterns (retry, throttling, sidecar, ambassador)
- **AWS Architecture Center**: [aws.amazon.com/architecture/](https://aws.amazon.com/architecture/)
    - *Best for*: Well-Architected Framework, reference architectures
- **Google Cloud Architecture Framework**: [cloud.google.com/architecture/framework](https://cloud.google.com/architecture/framework)
    - *Best for*: Operational excellence, security, reliability, cost optimization

#### Python Specific
- **Cosmic Python** (Architecture Patterns with Python): [cosmicpython.com](https://www.cosmicpython.com/)
    - *Best for*: DDD in Python, repository pattern, unit of work, CQRS
- **Real Python - Design Patterns**: [realpython.com/tutorials/patterns/](https://realpython.com/tutorials/patterns/)
    - *Best for*: Python-specific pattern implementations
- **Python 3.13 New Features** (2024): [docs.python.org/3.13/whatsnew/3.13.html](https://docs.python.org/3.13/whatsnew/3.13.html)
    - *Best for*: Architectural impact of **Free-threaded mode** (GIL removal experiment) and JIT compiler. Crucial for CPU-bound architectural decisions.

---

### 📐 Architecture Decision Records (ADRs)

- **ADR GitHub Organization**: [adr.github.io](https://adr.github.io/)
    - *Best for*: Templates, tools, examples of architecture decisions
- **Markdown ADR Template**: [github.com/joelparkerhenderson/architecture-decision-record](https://github.com/joelparkerhenderson/architecture-decision-record)
    - *Best for*: Documenting architectural choices with context and consequences
- **ADR Tools**: [github.com/npryce/adr-tools](https://github.com/npryce/adr-tools)
    - *Best for*: CLI tools to create and manage ADRs

---

### 🛠️ Tools & Diagrams

#### Visualization
- **C4 Model**: [c4model.com](https://c4model.com/)
    - *Best for*: Context, container, component, code diagrams
- **PlantUML**: [plantuml.com](https://plantuml.com/)
    - *Best for*: Diagram-as-code (sequence, class, component diagrams)
- **Mermaid**: [mermaid.js.org](https://mermaid.js.org/)
    - *Best for*: Markdown-native diagrams (flowcharts, sequence, ER diagrams)
- **Structurizr**: [structurizr.com](https://structurizr.com/)
    - *Best for*: C4 model diagrams with versioning

#### Analysis
- **SonarQube**: [sonarqube.org](https://www.sonarqube.org/)
    - *Best for*: Code quality, technical debt, architecture violations
- **ArchUnit** (Python): [github.com/TNG/archunit](https://github.com/TNG/archunit)
    - *Best for*: Testing architecture rules in code

---

### 📖 Additional Reading

#### API Design
- **REST API Design Rulebook** (Mark Masse)
    - *Best for*: RESTful API best practices
- **API Design Patterns** (JJ Geewax)
    - *Best for*: Naming, versioning, pagination, error handling

#### Event-Driven Architecture
- **Enterprise Integration Patterns**: [enterpriseintegrationpatterns.com](https://www.enterpriseintegrationpatterns.com/)
    - *Best for*: Message routing, transformation, event-driven patterns
- **Event Sourcing**: [martinfowler.com/eaaDev/EventSourcing.html](https://martinfowler.com/eaaDev/EventSourcing.html)
    - *Best for*: Event store, projections, CQRS

---

## Instructions for the Agent

1.  **Before proposing a major change**: 
    - Consult relevant books (Clean Architecture, DDD) to justify design choices
    - Reference specific patterns from Refactoring Guru or Martin Fowler's catalog
    - Create an ADR to document the decision

2.  **When refactoring**: 
    - Identify specific code smells from Refactoring Guru
    - Propose the corresponding pattern with a link to the pattern description
    - Ensure SOLID principles are maintained

3.  **When designing APIs**: 
    - Follow 12-factor principles (stateless, backing services as attached resources)
    - Reference REST API Design Rulebook for naming and versioning
    - Use OpenAPI/Swagger for documentation

4.  **When designing distributed systems**:
    - Consult "Designing Data-Intensive Applications" for data consistency patterns
    - Reference Microservices.io for service decomposition and communication
    - Apply cloud design patterns from Azure/AWS/GCP architecture centers

5.  **When creating diagrams**:
    - Use C4 Model for architecture visualization
    - Use PlantUML or Mermaid for diagram-as-code
    - Keep diagrams in version control alongside code

6.  **When making architectural decisions**:
    - Create an ADR using the template from adr.github.io
    - Document context, decision, and consequences
    - Store ADRs in `docs/architecture/decisions/`

---

## Code Examples

### Example 1: Clean Architecture - Repository Pattern

```python
# domain/repositories/user_repository.py (Port - Interface)
from abc import ABC, abstractmethod
from typing import Optional
from domain.entities.user import User

class UserRepository(ABC):
    """Domain layer interface - no infrastructure dependencies."""
    
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        pass
    
    @abstractmethod
    async def save(self, user: User) -> None:
        pass

# infrastructure/repositories/postgres_user_repository.py (Adapter)
from sqlalchemy.ext.asyncio import AsyncSession
from domain.repositories.user_repository import UserRepository
from domain.entities.user import User

class PostgresUserRepository(UserRepository):
    """Infrastructure implementation - depends on domain, not vice versa."""
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user_model = result.scalar_one_or_none()
        return User.from_orm(user_model) if user_model else None
    
    async def save(self, user: User) -> None:
        user_model = UserModel(**user.dict())
        self._session.add(user_model)
        await self._session.commit()
```

### Example 2: Dependency Injection with FastAPI

```python
# application/dependencies.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from infrastructure.database import get_db_session
from infrastructure.repositories.postgres_user_repository import PostgresUserRepository
from domain.repositories.user_repository import UserRepository

async def get_user_repository(
    session: AsyncSession = Depends(get_db_session)
) -> UserRepository:
    """Dependency injection - returns interface, not concrete class."""
    return PostgresUserRepository(session)

# interfaces/api/routes/users.py
from fastapi import APIRouter, Depends
from domain.repositories.user_repository import UserRepository
from application.dependencies import get_user_repository

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    user_repo: UserRepository = Depends(get_user_repository)  # Depends on interface
):
    """Controller depends on domain interface, not infrastructure."""
    user = await user_repo.get_by_id(user_id)
    return user
```

### Example 3: Strategy Pattern for LLM Provider Selection

```python
# domain/services/llm_service.py (Strategy Interface)
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

# infrastructure/llm/openai_provider.py
class OpenAIProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        response = await openai.ChatCompletion.acreate(...)
        return response.choices[0].message.content

# infrastructure/llm/anthropic_provider.py
class AnthropicProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        response = await anthropic.messages.create(...)
        return response.content[0].text

# application/use_cases/generate_text.py
class GenerateTextUseCase:
    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider  # Depends on interface
    
    async def execute(self, prompt: str) -> str:
        return await self._llm.generate(prompt)
```

---

## Anti-Patterns to Avoid

### ❌ God Class
**Problem**: Single class with too many responsibilities  
**Example**:
```python
# BAD: AgentManager does everything
class AgentManager:
    def load_prompts(self): ...
    def call_llm(self): ...
    def save_to_db(self): ...
    def send_email(self): ...
    def log_metrics(self): ...
```
**Solution**: Apply Single Responsibility Principle
```python
# GOOD: Separate classes for each responsibility
class PromptLoader: ...
class LLMClient: ...
class UserRepository: ...
class EmailService: ...
class MetricsLogger: ...
```

### ❌ Anemic Domain Model
**Problem**: Domain entities with no behavior, just getters/setters  
**Example**:
```python
# BAD: No business logic in domain
class User:
    def __init__(self, email: str, role: str):
        self.email = email
        self.role = role

# Business logic in service layer
def can_access_resource(user: User, resource: Resource) -> bool:
    return user.role == "admin" or resource.owner == user.email
```
**Solution**: Rich domain model with behavior
```python
# GOOD: Business logic in domain entity
class User:
    def __init__(self, email: str, role: Role):
        self.email = email
        self.role = role
    
    def can_access(self, resource: Resource) -> bool:
        """Business logic belongs in domain."""
        return self.role.is_admin() or resource.is_owned_by(self)
```

### ❌ Hardcoded Dependencies
**Problem**: Direct instantiation of dependencies  
**Example**:
```python
# BAD: Hardcoded database connection
class UserService:
    def __init__(self):
        self.db = PostgresDatabase()  # Tight coupling
```
**Solution**: Dependency injection
```python
# GOOD: Injected dependency
class UserService:
    def __init__(self, db: Database):
        self.db = db  # Loose coupling, testable
```

### ❌ Leaky Abstractions
**Problem**: Infrastructure details leak into domain  
**Example**:
```python
# BAD: SQLAlchemy model in domain
from sqlalchemy import Column, String
class User(Base):  # Domain entity depends on SQLAlchemy
    __tablename__ = "users"
    id = Column(String, primary_key=True)
```
**Solution**: Pure domain entities
```python
# GOOD: Pure domain entity
@dataclass
class User:
    id: str
    email: str
    # No infrastructure dependencies
```

---

## Architecture Review Checklist

### Clean Architecture Compliance
- [ ] Domain layer has no external dependencies
- [ ] Application layer orchestrates use cases
- [ ] Infrastructure implements domain interfaces
- [ ] Dependency rule: outer layers depend on inner, never reverse

### SOLID Principles
- [ ] Each class has single responsibility (SRP)
- [ ] Classes open for extension, closed for modification (OCP)
- [ ] Subtypes can replace base types (LSP)
- [ ] Interfaces are client-specific, not general (ISP)
- [ ] Depend on abstractions, not concretions (DIP)

### Design Patterns
- [ ] Patterns solve real problems (not used for show)
- [ ] Repository pattern for data access
- [ ] Strategy pattern for interchangeable algorithms
- [ ] Factory pattern for object creation
- [ ] Decorator pattern for cross-cutting concerns

### Code Quality
- [ ] No God classes (> 300 lines)
- [ ] No anemic domain models
- [ ] No hardcoded dependencies
- [ ] No circular dependencies
- [ ] Cyclomatic complexity < 10 per function

### Documentation
- [ ] Architecture Decision Records (ADRs) exist
- [ ] C4 diagrams for system context
- [ ] README explains architecture layers
- [ ] API documentation generated (OpenAPI)

---

## Additional References

### Architecture Patterns
- **CNCF Cloud Native Landscape**: [landscape.cncf.io](https://landscape.cncf.io/)
    - *Best for*: Exploring cloud-native ecosystem
- **Software Architecture Patterns** (O'Reilly): [oreilly.com/library/view/software-architecture-patterns/](https://www.oreilly.com/library/view/software-architecture-patterns/)
    - *Best for*: Layered, event-driven, microkernel patterns
- **Hexagonal Architecture** (Alistair Cockburn): [alistair.cockburn.us/hexagonal-architecture/](https://alistair.cockburn.us/hexagonal-architecture/)
    - *Best for*: Ports and adapters pattern

### Python Architecture
- **Architecture Patterns with Python** (Cosmic Python): [cosmicpython.com](https://www.cosmicpython.com/)
    - *Best for*: DDD, Repository, Unit of Work in Python
- **FastAPI Best Practices**: [github.com/zhanymkanov/fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices)
    - *Best for*: Project structure, dependency injection

### Refactoring
- **Refactoring Catalog** (Martin Fowler): [refactoring.com/catalog/](https://refactoring.com/catalog/)
    - *Best for*: Specific refactoring techniques
- **Code Smells**: [refactoring.guru/refactoring/smells](https://refactoring.guru/refactoring/smells)
    - *Best for*: Identifying problematic code patterns
