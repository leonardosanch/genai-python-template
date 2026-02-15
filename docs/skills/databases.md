---
name: Databases & Storage
description: Patterns and best practices for SQL, NoSQL, and vector storage.
---

# Skill: Databases & Storage

## Description

Production GenAI systems require a polyglot persistence layer: SQL for structured business data, NoSQL for logs and metadata, and vector stores for embeddings and similarity search. This skill covers async data access patterns, connection management, migration strategies, and vector search integration within a Clean Architecture context.

## Executive Summary

**Critical database rules:**
- ALWAYS use async drivers (`asyncpg`, `motor`, `redis.asyncio`) — synchronous drivers block the event loop
- Repository pattern MANDATORY — all data access behind domain-defined ports (SQL never in domain layer)
- Connection pooling explicitly configured — set `pool_size`, `max_overflow`, `pool_timeout` (defaults fail in production)
- PostgreSQL JSONB before adding NoSQL — handles most semi-structured data needs without new infrastructure
- Consult Decision Tree 1 before adding databases — PostgreSQL + JSONB + pgvector covers 80% of cases

**Read full skill when:** Designing data access layers, choosing databases, implementing repositories, configuring connection pools, or integrating vector stores.

---

## Versiones y Pool Tuning

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| asyncpg | >= 0.28.0 | Driver SQL recomendado |
| motor | >= 3.3.0 | Driver NoSQL recomendado |
| redis | >= 5.0.0 | Driver Cache recomendado |
| pgvector | >= 0.2.0 | Soporte HNSW |

### SQLAlchemy Pool Config

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,           # Conexiones persistentes
    max_overflow=10,        # Conexiones extra bajo carga
    pool_timeout=30,        # Segundos de espera para conexión
    pool_recycle=1800,      # Reiniciar conexiones cada 30 min
)
```

---

## Deep Dive

## Core Concepts

1. **Repository Pattern** — All data access is abstracted behind domain-defined ports (interfaces). Infrastructure adapters implement these ports using specific drivers. The domain layer never imports SQLAlchemy, Motor, or any other persistence library.

2. **Async-First Data Access** — Every database driver must be async (`asyncpg`, `motor`, `redis.asyncio`). Synchronous drivers block the event loop and destroy throughput in FastAPI or any async application. There are no exceptions to this rule.

3. **Connection Pooling** — Every production deployment must configure explicit pool sizes (`pool_size`, `max_overflow`, `pool_timeout` in SQLAlchemy; `maxPoolSize` in Motor). Unconfigured pools lead to connection exhaustion under load.

4. **Schema Migrations as Code** — All schema changes go through Alembic migrations, versioned in Git, applied via CI/CD. Manual DDL in production is forbidden. Migrations must be backward-compatible to support rolling deployments.

5. **JSONB Before NoSQL** — PostgreSQL JSONB columns handle most semi-structured data needs. Only introduce MongoDB or DynamoDB when there is a genuine requirement that PostgreSQL cannot satisfy (e.g., document-level sharding, sub-millisecond key-value access at scale).

6. **Vector Search Integration** — pgvector is the default for teams already on PostgreSQL. Dedicated vector stores (Pinecone, Qdrant, Weaviate) are justified only at scale (>10M vectors) or when advanced features (hybrid search, filtering, managed infrastructure) are required.

## External Resources

### :zap: Core Databases

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
  *Best for:* Understanding the new 2.0-style query API, async session patterns, and ORM mapping.

- [asyncpg — Fast PostgreSQL Driver](https://magicstack.github.io/asyncpg/current/)
  *Best for:* Low-level async PostgreSQL operations and connection pool tuning.

- [Alembic Migration Tool](https://alembic.sqlalchemy.org/en/latest/)
  *Best for:* Writing and managing database migrations, autogenerate workflows.

- [PostgreSQL Official Docs](https://www.postgresql.org/docs/current/)
  *Best for:* Understanding JSONB, indexing strategies, and query optimization.

### :shield: NoSQL & Caching

- [Redis Documentation](https://redis.io/docs/)
  *Best for:* Cache patterns, data structures, TTL strategies, and Pub/Sub.

- [Motor — Async MongoDB Driver](https://motor.readthedocs.io/en/stable/)
  *Best for:* Async MongoDB operations with Tornado or asyncio.

- [DynamoDB with boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html)
  *Best for:* Serverless NoSQL on AWS, single-table design patterns.

### :books: Vector Stores

- [pgvector GitHub](https://github.com/pgvector/pgvector)
  *Best for:* Adding vector similarity search to existing PostgreSQL deployments.

- [Pinecone Documentation](https://docs.pinecone.io/)
  *Best for:* Managed vector database with filtering and serverless scaling.

- [Qdrant Documentation](https://qdrant.tech/documentation/)
  *Best for:* Self-hosted vector search with advanced filtering and payload indexing.

### :book: Tutorials & Guides

- [Real Python — SQLAlchemy Tutorial](https://realpython.com/python-sqlite-sqlalchemy/)
  *Best for:* Practical introduction to SQLAlchemy patterns and session management.

- [Connection Pooling Best Practices (PgBouncer)](https://www.pgbouncer.org/)
  *Best for:* External connection pooling for high-concurrency PostgreSQL deployments.

## Decision Trees

### Decision Tree 1: Qué base de datos usar

```
¿Qué tipo de datos almacenas?
├── Datos estructurados con relaciones (users, orders, products)
│   └── PostgreSQL (default para todo proyecto nuevo)
├── Datos semi-estructurados (LLM metadata, logs, flexible schema)
│   └── ¿Ya usas PostgreSQL?
│       ├── SÍ → PostgreSQL JSONB (no agregar otro DB)
│       └── NO → MongoDB (solo si hay un requisito real)
├── Cache / rate limiting / sessions
│   └── Redis (siempre async con redis.asyncio)
├── Embeddings / vector search
│   └── Ver Decision Tree 2
└── Key-value serverless (AWS)
    └── DynamoDB (single-table design)
```

### Decision Tree 2: pgvector vs Vector Store dedicado

```
¿Cuántos vectores vas a almacenar?
├── < 1M vectores
│   └── ¿Ya usas PostgreSQL?
│       ├── SÍ → pgvector (cero infra adicional)
│       └── NO → ChromaDB para dev, pgvector para prod
├── 1M - 10M vectores
│   └── ¿Necesitas hybrid search (keyword + vector)?
│       ├── SÍ → Weaviate (BM25 + vector nativo)
│       └── NO → pgvector con HNSW index (aún viable)
└── > 10M vectores
    └── ¿Necesitas managed / serverless?
        ├── SÍ → Pinecone (auto-scaling, zero ops)
        └── NO → Qdrant self-hosted (mejor filtering, open-source)
```

### Decision Tree 3: Estrategia de índice pgvector

```
¿Cuál es tu prioridad?
├── Recall alto (no perder resultados relevantes)
│   └── HNSW index (ef_construction=200, m=16)
│       ├── Más memoria, build más lento
│       └── Mejor recall que IVFFlat
├── Velocidad de build (re-indexar frecuentemente)
│   └── IVFFlat (lists = sqrt(n_vectors))
│       ├── Build más rápido
│       └── Requiere VACUUM después de bulk inserts
└── Dataset pequeño (< 50K vectores)
    └── Sin índice (exact search es suficiente)
```

---

## Instructions for the Agent

1. **Consultar decision tree antes de elegir DB.** No agregar MongoDB, DynamoDB, o
   un vector store dedicado sin justificación documentada. PostgreSQL + JSONB + pgvector
   cubre el 80% de los casos.

2. **Always use async drivers** — `asyncpg` para PostgreSQL, `motor` para MongoDB,
   `redis.asyncio` para Redis. Nunca `psycopg2` o `pymongo` en apps async.

3. **Repository pattern es obligatorio** — Todo data access a través de interfaces
   en domain layer. No importar SQLAlchemy en domain ni application layers.

4. **Migrations versionadas vía CI/CD** — Cada schema change requiere Alembic migration.
   Nunca `metadata.create_all()` en producción.

5. **Connection pooling explícito** — Configurar `pool_size`, `max_overflow`, `pool_timeout`.
   Defaults son insuficientes para producción.

6. **Nunca raw SQL en domain layer** — SQL vive solo en infrastructure adapters.

7. **JSONB antes de NoSQL** — PostgreSQL JSONB con GIN indexes cubre la mayoría de
   requerimientos semi-estructurados.

8. **Indexar columnas consultadas frecuentemente** — `EXPLAIN ANALYZE` para validar.
   Partial indexes para queries filtradas, GIN para JSONB.

9. **Evitar N+1 queries** — `selectinload` o `joinedload` en SQLAlchemy.

10. **Plan de re-indexación para embeddings.** Cuando cambias el modelo de embedding,
    TODOS los vectores deben re-generarse. Tener un pipeline de bulk re-indexing con
    zero-downtime (blue-green index swap).

## Code Examples

### Unit of Work Pattern (Multi-Repository Transactions)

```python
"""Unit of Work for transactions spanning multiple repositories."""
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class UnitOfWork:
    """Coordinates transactions across multiple repositories.

    Usage:
        async with uow:
            await uow.documents.save(doc)
            await uow.embeddings.upsert(chunks)
            await uow.outbox.publish(event)
            await uow.commit()
            # All three operations in ONE transaction
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def __aenter__(self) -> "UnitOfWork":
        self._session = self._session_factory()
        # Initialize repositories with shared session
        self.documents = DocumentRepository(self._session)
        self.embeddings = EmbeddingRepository(self._session)
        self.outbox = OutboxRepository(self._session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            await self.rollback()
        await self._session.close()

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()
```

### JSONB Patterns (Avoid Adding NoSQL)

```python
"""PostgreSQL JSONB patterns — covers most semi-structured data needs."""
from sqlalchemy import Column, String, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.infrastructure.data.models import Base


class LLMCallLog(Base):
    """Store LLM call metadata as JSONB — no need for MongoDB."""

    __tablename__ = "llm_call_logs"

    id = Column(String(36), primary_key=True)
    model = Column(String(100), nullable=False, index=True)
    # JSONB for flexible, queryable metadata
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
    # Example metadata: {"tokens": 150, "cost_usd": 0.003, "latency_ms": 450,
    #   "prompt_version": "v2.3", "tags": ["support", "refund"]}

    __table_args__ = (
        # GIN index for JSONB queries
        Index("ix_llm_logs_metadata", metadata_, postgresql_using="gin"),
        # Partial index: only high-cost calls
        Index(
            "ix_llm_logs_high_cost",
            metadata_,
            postgresql_using="gin",
            postgresql_where=text("(metadata->>'cost_usd')::float > 0.01"),
        ),
    )


class LLMCallLogRepository:
    """Query JSONB fields without leaving PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_tag(self, tag: str) -> list[LLMCallLog]:
        """Query JSONB array field."""
        stmt = select(LLMCallLog).where(
            LLMCallLog.metadata_["tags"].contains([tag])
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def find_expensive_calls(self, min_cost: float) -> list[LLMCallLog]:
        """Query JSONB numeric field."""
        stmt = select(LLMCallLog).where(
            LLMCallLog.metadata_["cost_usd"].as_float() > min_cost
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def aggregate_cost_by_model(self) -> list[dict]:
        """Aggregate JSONB fields with SQL."""
        stmt = text("""
            SELECT model,
                   COUNT(*) as call_count,
                   SUM((metadata->>'cost_usd')::float) as total_cost,
                   AVG((metadata->>'latency_ms')::float) as avg_latency
            FROM llm_call_logs
            GROUP BY model
            ORDER BY total_cost DESC
        """)
        result = await self._session.execute(stmt)
        return [dict(row._mapping) for row in result]
```

### Embedding Re-indexing with Zero Downtime

```python
"""Blue-green index swap for embedding model changes."""
from datetime import datetime, timezone


class EmbeddingReindexer:
    """Re-index all embeddings when changing embedding model.

    Strategy: blue-green swap
    1. Create new index table (embeddings_v2)
    2. Batch re-generate embeddings with new model
    3. Validate new index quality
    4. Swap alias to point to new table
    5. Drop old table after validation period
    """

    def __init__(self, db, old_table: str, new_table: str, embedding_model):
        self._db = db
        self._old = old_table
        self._new = new_table
        self._model = embedding_model

    async def reindex_batch(self, batch_size: int = 100, offset: int = 0) -> int:
        """Re-generate embeddings for a batch of documents."""
        async with self._db.session() as session:
            # Read from old table
            result = await session.execute(
                text(f"SELECT id, content, source FROM {self._old} LIMIT :limit OFFSET :offset"),
                {"limit": batch_size, "offset": offset},
            )
            rows = result.fetchall()

            if not rows:
                return 0

            # Generate new embeddings
            texts = [row.content for row in rows]
            new_embeddings = await self._model.embed(texts)

            # Write to new table
            for row, embedding in zip(rows, new_embeddings):
                await session.execute(
                    text(f"""
                        INSERT INTO {self._new} (id, content, source, embedding)
                        VALUES (:id, :content, :source, :embedding)
                        ON CONFLICT (id) DO UPDATE SET embedding = :embedding
                    """),
                    {"id": row.id, "content": row.content, "source": row.source, "embedding": embedding},
                )
            await session.commit()
            return len(rows)

    async def swap(self) -> None:
        """Atomic swap: rename tables."""
        async with self._db.session() as session:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            await session.execute(text(f"ALTER TABLE {self._old} RENAME TO {self._old}_{timestamp}_bak"))
            await session.execute(text(f"ALTER TABLE {self._new} RENAME TO {self._old}"))
            await session.commit()
```

### Async SQLAlchemy Repository with Session Management

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import TypeVar

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import select

from src.domain.entities.document import Document
from src.domain.ports.document_repository import DocumentRepositoryPort

T = TypeVar("T")


class AsyncDatabase:
    """Manages async SQLAlchemy engine and session factory."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(
            database_url,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class SQLAlchemyDocumentRepository(DocumentRepositoryPort):
    """Infrastructure adapter for document persistence."""

    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db

    async def save(self, document: Document) -> str:
        async with self._db.session() as session:
            model = DocumentModel.from_entity(document)
            session.add(model)
            await session.flush()
            return str(model.id)

    async def find_by_id(self, document_id: str) -> Document | None:
        async with self._db.session() as session:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            return model.to_entity() if model else None
```

### Alembic Migration with Async Engine

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from src.infrastructure.config.settings import get_settings
from src.infrastructure.data.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode (generates SQL script)."""
    url = get_settings().database_url
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in online mode with async engine."""
    engine = create_async_engine(get_settings().database_url)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

### Redis Cache with TTL and Serialization

```python
import json
from datetime import timedelta

import redis.asyncio as redis
from pydantic import BaseModel


class RedisCache:
    """Async Redis cache with Pydantic serialization and TTL."""

    def __init__(self, redis_url: str, default_ttl: timedelta = timedelta(hours=1)) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._default_ttl = default_ttl

    async def get(self, key: str, model_class: type[BaseModel]) -> BaseModel | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        return model_class.model_validate_json(raw)

    async def set(
        self,
        key: str,
        value: BaseModel,
        ttl: timedelta | None = None,
    ) -> None:
        serialized = value.model_dump_json()
        await self._client.set(key, serialized, ex=int((ttl or self._default_ttl).total_seconds()))

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def invalidate_pattern(self, pattern: str) -> None:
        """Delete all keys matching a glob pattern."""
        cursor = None
        while cursor != 0:
            cursor, keys = await self._client.scan(cursor=cursor or 0, match=pattern, count=100)
            if keys:
                await self._client.delete(*keys)

    async def close(self) -> None:
        await self._client.close()
```

### pgvector Similarity Search

```python
from sqlalchemy import Column, Integer, String, Text, Index
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pgvector.sqlalchemy import Vector

from src.infrastructure.data.models import Base


class EmbeddingModel(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String(512), nullable=False)
    embedding = Column(Vector(1536), nullable=False)

    __table_args__ = (
        Index(
            "ix_embeddings_vector",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class PgVectorRepository:
    """Vector similarity search using pgvector."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        async with self._session_factory() as session:
            stmt = (
                select(
                    EmbeddingModel.id,
                    EmbeddingModel.content,
                    EmbeddingModel.source,
                    EmbeddingModel.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .where(
                    EmbeddingModel.embedding.cosine_distance(query_embedding) < (1 - similarity_threshold)
                )
                .order_by("distance")
                .limit(top_k)
            )
            result = await session.execute(stmt)
            return [
                {"id": row.id, "content": row.content, "source": row.source, "score": 1 - row.distance}
                for row in result
            ]
```

## Anti-Patterns to Avoid

### :x: Synchronous Drivers in Async Applications

**Problem:** Using `psycopg2` or synchronous `pymongo` in a FastAPI app blocks the event loop, reducing throughput to a single concurrent request per worker.

**Example:**
```python
# BAD: sync driver in async app
from sqlalchemy import create_engine
engine = create_engine("postgresql://...")  # blocks event loop
```

**Solution:** Always use async engines and drivers.
```python
from sqlalchemy.ext.asyncio import create_async_engine
engine = create_async_engine("postgresql+asyncpg://...")
```

### :x: Raw SQL in Business Logic

**Problem:** SQL strings scattered across use cases create tight coupling to the database schema and make refactoring dangerous.

**Example:**
```python
# BAD: SQL in application layer
class ProcessOrderUseCase:
    async def execute(self, order_id: str):
        result = await self.db.execute("SELECT * FROM orders WHERE id = %s", order_id)
```

**Solution:** Use repository interfaces. SQL lives only in infrastructure adapters.
```python
class ProcessOrderUseCase:
    def __init__(self, order_repo: OrderRepositoryPort):
        self._repo = order_repo

    async def execute(self, order_id: str) -> Order:
        return await self._repo.find_by_id(order_id)
```

### :x: No Connection Pooling Configuration

**Problem:** Default pool settings cause connection exhaustion under moderate load, leading to `TimeoutError` or `TooManyConnections` in production.

**Example:**
```python
# BAD: default pool — pool_size=5, no overflow control
engine = create_async_engine("postgresql+asyncpg://localhost/db")
```

**Solution:** Explicitly configure pool parameters based on expected concurrency.
```python
engine = create_async_engine(
    "postgresql+asyncpg://localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)
```

### :x: N+1 Query Problem

**Problem:** Loading a list of entities and then issuing one query per entity for related data. Common with lazy-loaded relationships.

**Example:**
```python
# BAD: N+1 — one query for users, then N queries for orders
users = await session.execute(select(User))
for user in users.scalars():
    orders = await session.execute(select(Order).where(Order.user_id == user.id))
```

**Solution:** Use eager loading strategies.
```python
from sqlalchemy.orm import selectinload
stmt = select(User).options(selectinload(User.orders))
result = await session.execute(stmt)
```

### :x: Missing Indexes on Frequently Queried Columns

**Problem:** Full table scans on columns used in WHERE clauses, JOIN conditions, or ORDER BY. Performance degrades as data grows.

**Solution:** Analyze slow queries with `EXPLAIN ANALYZE`. Add indexes for filter and join columns. Use partial indexes for filtered subsets and GIN indexes for JSONB columns.

## Database Checklist

### Schema Design
- [ ] Primary keys use UUIDs or auto-increment based on use case
- [ ] Foreign keys have explicit ON DELETE behavior
- [ ] JSONB columns have GIN indexes for queried paths
- [ ] Enum columns use PostgreSQL ENUM types or CHECK constraints
- [ ] Table and column names follow snake_case convention

### Connection Management
- [ ] Async drivers configured (`asyncpg`, `motor`, `redis.asyncio`)
- [ ] Pool size, max overflow, and timeout explicitly set
- [ ] Pool recycle configured to avoid stale connections
- [ ] Health check queries enabled for connection validation
- [ ] External pooler (PgBouncer) evaluated for high-concurrency scenarios

### Migration Safety
- [ ] All schema changes go through Alembic migrations
- [ ] Migrations are backward-compatible (no column drops without deprecation)
- [ ] CI/CD pipeline applies migrations before deploying new code
- [ ] Migration rollback scripts exist for critical changes
- [ ] `metadata.create_all()` is never used in production

### Vector Store Configuration
- [ ] Embedding dimension matches the model output (e.g., 1536 for OpenAI)
- [ ] IVFFlat or HNSW index created with appropriate parameters
- [ ] Similarity metric matches use case (cosine, L2, inner product)
- [ ] Index rebuild scheduled after large bulk inserts
- [ ] Query performance validated with `EXPLAIN ANALYZE`

### Backup & Recovery
- [ ] Automated daily backups configured
- [ ] Point-in-time recovery (PITR) enabled for PostgreSQL
- [ ] Backup restoration tested at least quarterly
- [ ] Redis persistence strategy chosen (RDB, AOF, or hybrid)
- [ ] Disaster recovery runbook documented

## Additional References

- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [SQLAlchemy Async Session Patterns](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Redis Patterns — Caching Best Practices](https://redis.io/docs/manual/patterns/)
- [pgvector — Indexing Strategies](https://github.com/pgvector/pgvector#indexing)
- [Martin Fowler — Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)
