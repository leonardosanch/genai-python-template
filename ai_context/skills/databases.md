---
name: Databases & Storage
description: Patterns and best practices for SQL, NoSQL, and vector storage.
---

# Databases & Storage

## Overview
A production GenAI system typically requires a polyglot persistence layer: SQL for structured business data, NoSQL for logs/metadata, and Vector Stores for embeddings.

## Core Technologies

### Relational (SQL)
- **PostgreSQL**: The default choice. Robust, ACID compliant, and extensible.
    - **pgvector**: Preferred extension for vector search within Postgres.
- **SQLAlchemy (Async)**: The standard ORM for Python. Use version 2.0+ patterns.
- **Alembic**: Essential for database migrations.

### NoSQL & Caching
- **MongoDB**: Ideal for storing unstructured LLM traces, chat history, and raw document metadata.
- **Redis**: The go-to solution for caching, rate limiting, and simple task queues.

### Vector Stores
- **Dedicated**: Pinecone, Weaviate, Qdrant.
- **Integrated**: pgvector (Postgres), Atlas Vector Search (MongoDB).

## Implementation Patterns

### 1. Repository Pattern
Abstract data access behind interfaces to decouple domain logic from persistence details.

```python
class DocumentRepository(ABC):
    @abstractmethod
    async def save(self, doc: Document) -> str: ...
    
    @abstractmethod
    async def search(self, query: str) -> list[Document]: ...
```

### 2. Unit of Work
Manage transactions across multiple repositories to ensure data consistency.

### 3. Connection Pooling
Always configure connection pools (e.g., `max_overflow`, `pool_size` in SQLAlchemy) to handle concurrent load.

## Best Practices
1.  **Async Drivers**: Always use async drivers (`asyncpg`, `motor`, `redis.asyncio`) to prevent blocking the event loop.
2.  **Migrations**: Database schema changes must be versioned and applied via CI/CD.
3.  **JSONB**: Use Postgres JSONB for flexible fields instead of adding NoSQL unnecessarily.
4.  **Read Replicas**: Offload heavy read queries (like reporting) to replicas.

## External Resources
- [SQLAlchemy 2.0 Docs](https://docs.sqlalchemy.org/en/20/)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [MongoDB Motor](https://motor.readthedocs.io/)
