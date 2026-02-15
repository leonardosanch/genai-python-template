# Bases de Datos

## Bases de Datos Relacionales (SQL)

### PostgreSQL (Recomendado)

Base de datos relacional principal. Soporta pgvector para embeddings.

```python
# Port en domain
from abc import ABC, abstractmethod
from src.domain.entities.document import Document

class DocumentRepositoryPort(ABC):
    @abstractmethod
    async def save(self, document: Document) -> str: ...

    @abstractmethod
    async def get_by_id(self, id: str) -> Document | None: ...

    @abstractmethod
    async def list(self, offset: int = 0, limit: int = 20) -> list[Document]: ...

    @abstractmethod
    async def delete(self, id: str) -> None: ...
```

#### Con SQLAlchemy (async)

```python
# src/infrastructure/database/sqlalchemy_repository.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    content: Mapped[str]
    metadata_json: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class SQLAlchemyDocumentRepository(DocumentRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, document: Document) -> str:
        model = DocumentModel(
            id=document.id or str(uuid4()),
            title=document.metadata.get("title", ""),
            content=document.content,
            metadata_json=json.dumps(document.metadata),
        )
        self._session.add(model)
        await self._session.commit()
        return model.id

    async def get_by_id(self, id: str) -> Document | None:
        result = await self._session.get(DocumentModel, id)
        if not result:
            return None
        return Document(
            id=result.id,
            content=result.content,
            metadata=json.loads(result.metadata_json or "{}"),
        )

    async def list(self, offset: int = 0, limit: int = 20) -> list[Document]:
        stmt = select(DocumentModel).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return [
            Document(id=r.id, content=r.content, metadata=json.loads(r.metadata_json or "{}"))
            for r in result.scalars()
        ]

    async def delete(self, id: str) -> None:
        stmt = delete(DocumentModel).where(DocumentModel.id == id)
        await self._session.execute(stmt)
        await self._session.commit()
```

#### Con Django ORM

```python
# models.py
from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=500)
    content = models.TextField()
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["title"]),
            models.Index(fields=["created_at"]),
        ]

class ChatHistory(models.Model):
    session_id = models.UUIDField(db_index=True)
    role = models.CharField(max_length=20)  # user, assistant, system
    content = models.TextField()
    tokens_used = models.IntegerField(default=0)
    model = models.CharField(max_length=100)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
```

#### Migrations

```bash
# SQLAlchemy (con Alembic)
uv run alembic init migrations
uv run alembic revision --autogenerate -m "create documents table"
uv run alembic upgrade head

# Django
uv run python manage.py makemigrations
uv run python manage.py migrate
```

### MySQL / MariaDB

Alternativa a PostgreSQL. Usar cuando el equipo ya tiene expertise o infraestructura existente.

```python
# SQLAlchemy connection
engine = create_async_engine("mysql+aiomysql://user:pass@localhost/dbname")
```

### SQLite

Solo para desarrollo local y testing. Nunca en producción para APIs.

```python
engine = create_async_engine("sqlite+aiosqlite:///./dev.db")
```

### Comparativa SQL

| Feature | PostgreSQL | MySQL | SQLite |
|---------|-----------|-------|--------|
| Producción | Si | Si | No |
| JSON nativo | Si (JSONB) | Si (JSON) | Limitado |
| Vector search | Si (pgvector) | No nativo | No |
| Full-text search | Si | Si | Limitado |
| Async drivers | asyncpg | aiomysql | aiosqlite |
| Django support | Excelente | Excelente | Solo dev |

---

## Bases de Datos No Relacionales (NoSQL)

### MongoDB

Base de datos documental. Flexible schema, bueno para datos semi-estructurados.

```python
# src/infrastructure/database/mongo_repository.py
from motor.motor_asyncio import AsyncIOMotorClient

class MongoDocumentRepository(DocumentRepositoryPort):
    def __init__(self, client: AsyncIOMotorClient, db_name: str = "genai"):
        self._collection = client[db_name]["documents"]

    async def save(self, document: Document) -> str:
        doc_dict = {
            "_id": document.id or str(uuid4()),
            "content": document.content,
            "metadata": document.metadata,
            "created_at": datetime.utcnow(),
        }
        await self._collection.insert_one(doc_dict)
        return doc_dict["_id"]

    async def get_by_id(self, id: str) -> Document | None:
        doc = await self._collection.find_one({"_id": id})
        if not doc:
            return None
        return Document(id=doc["_id"], content=doc["content"], metadata=doc.get("metadata", {}))

    async def list(self, offset: int = 0, limit: int = 20) -> list[Document]:
        cursor = self._collection.find().skip(offset).limit(limit)
        return [
            Document(id=doc["_id"], content=doc["content"], metadata=doc.get("metadata", {}))
            async for doc in cursor
        ]

    async def delete(self, id: str) -> None:
        await self._collection.delete_one({"_id": id})
```

**Cuándo usar MongoDB:**
- Documentos con schema variable
- Logs, eventos, metadata de LLM
- Cuando la estructura de datos cambia frecuentemente
- Chat histories con contenido flexible

### Redis

Cache y message broker. In-memory, extremadamente rápido.

```python
# src/infrastructure/cache/redis_cache.py
import redis.asyncio as redis

class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379"):
        self._client = redis.from_url(url)

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        await self._client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)
```

**Casos de uso con GenAI:**
- Cache de respuestas de LLM (por hash del prompt)
- Rate limiting
- Session storage
- Task queues (con Redis Streams o Celery)
- Pub/Sub para eventos

### DynamoDB

NoSQL serverless de AWS. Pay-per-request, escalado automático.

```python
# src/infrastructure/database/dynamo_repository.py
import aioboto3

class DynamoDocumentRepository(DocumentRepositoryPort):
    def __init__(self, table_name: str = "documents"):
        self._table_name = table_name
        self._session = aioboto3.Session()

    async def save(self, document: Document) -> str:
        async with self._session.resource("dynamodb") as dynamo:
            table = await dynamo.Table(self._table_name)
            item = {
                "id": document.id or str(uuid4()),
                "content": document.content,
                "metadata": document.metadata,
            }
            await table.put_item(Item=item)
            return item["id"]
```

### Firestore

NoSQL serverless de Google Cloud. Real-time listeners, buena integración con GCP.

### Comparativa NoSQL

| Feature | MongoDB | Redis | DynamoDB | Firestore |
|---------|---------|-------|----------|-----------|
| Tipo | Documental | Key-Value / Cache | Key-Value / Documental | Documental |
| Schema | Flexible | N/A | Flexible | Flexible |
| Queries complejas | Si | Limitado | Limitado | Limitado |
| Real-time | Change Streams | Pub/Sub | Streams | Si (nativo) |
| Serverless | Atlas | Elasticache | Si (nativo) | Si (nativo) |
| Async driver | motor | redis-py async | aioboto3 | google-cloud-firestore async |

---

## Vector Databases

Ver [RAG.md](RAG.md) para detalle completo de Pinecone, Weaviate, Qdrant, ChromaDB y pgvector.

---

## Patrones de Acceso a Datos

### Repository Pattern

Todo acceso a datos pasa por un repository (port en domain, implementación en infrastructure).

```python
# Domain port
class UserRepositoryPort(ABC):
    @abstractmethod
    async def find_by_email(self, email: str) -> User | None: ...

# Infrastructure implementation
class PostgresUserRepository(UserRepositoryPort):
    async def find_by_email(self, email: str) -> User | None:
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return self._to_entity(row) if row else None
```

### Unit of Work

Para transacciones que abarcan múltiples repositories.

```python
class UnitOfWork:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def __aenter__(self):
        self._session = self._session_factory()
        self.documents = SQLAlchemyDocumentRepository(self._session)
        self.users = PostgresUserRepository(self._session)
        return self

    async def __aexit__(self, exc_type, *args):
        if exc_type:
            await self._session.rollback()
        await self._session.close()

    async def commit(self):
        await self._session.commit()
```

### Connection Pooling

```python
# SQLAlchemy
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
)

# MongoDB
client = AsyncIOMotorClient(MONGO_URL, maxPoolSize=50, minPoolSize=10)
```

---

## Integración con el Proyecto

```
src/infrastructure/
├── database/
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy models
│   ├── session.py             # Session factory, engine
│   ├── sqlalchemy_repository.py
│   ├── mongo_repository.py
│   └── migrations/            # Alembic migrations
├── cache/
│   ├── __init__.py
│   └── redis_cache.py
└── vectorstores/              # Ya existente
    ├── pinecone_store.py
    ├── qdrant_store.py
    └── chroma_store.py
```

---

## Reglas

1. **Repository pattern siempre** — nunca queries directas fuera de infrastructure
2. **Async drivers** — asyncpg, motor, redis.asyncio, nunca drivers sync
3. **Migrations versionadas** — Alembic o Django migrations, siempre en Git
4. **Connection pooling** — configurado para producción
5. **PostgreSQL por defecto** — salvo que haya razón para otra opción
6. **Redis para cache** — nunca como DB principal
7. **Un port, múltiples implementaciones** — cambiar de DB sin tocar domain

Ver también: [ARCHITECTURE.md](ARCHITECTURE.md), [API.md](API.md), [RAG.md](RAG.md), [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md)
