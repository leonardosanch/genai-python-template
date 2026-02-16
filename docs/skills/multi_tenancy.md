# Skill: Multi-Tenancy Patterns for GenAI

## Description

Architectural patterns and implementation strategies for building secure, scalable multi-tenant GenAI systems with proper isolation, cost attribution, and rate limiting.

## Executive Summary

**Critical rules (always enforce):**
- **NEVER share context/memory between tenants** — Complete isolation is mandatory
- **Tenant ID in EVERY request** — Via header, JWT claim, or API key
- **Rate limiting PER TENANT** — Not global, prevents noisy neighbor problem
- **Cost tracking POR TENANT** — Essential for billing and chargeback
- **Data isolation** — tenant_id in all database queries, vector store namespaces
- **Validate tenant access** — Middleware must verify tenant permissions on every request
- **Separate secrets per tenant** — API keys, credentials must be tenant-scoped

**Read full skill when:** Building SaaS GenAI products, implementing multi-tenant RAG, designing tenant isolation strategies, or debugging cross-tenant data leakage.

---

---

## Versiones y Aislamiento

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| sqlalchemy | >= 2.0.0 | Soporte async estable |
| pinecone-client | >= 3.0.0 | Soporte namespaces v3 |
| qdrant-client | >= 1.6.0 | Collections API estable |
| redis | >= 5.0.0 | Cache multi-tenant |

### Tenant Isolation Check (Middleware)

```python
async def verify_tenant_isolation(request: Request):
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        raise HTTPException(status_code=401)
    # ✅ Inyectar tenant_id en el contexto de ejecución
    request.state.tenant_id = tenant_id
```

---

## Deep Dive

## Isolation Strategies

### 1. Database-Level Isolation

**Schema per Tenant:**
```sql
-- Separate schema for each tenant
CREATE SCHEMA tenant_123;
CREATE TABLE tenant_123.documents (...);
CREATE TABLE tenant_123.conversations (...);
```

**Row-Level Security (PostgreSQL RLS):**
```sql
-- Single schema, RLS enforces isolation
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    content TEXT
);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

**Separate Databases:**
- Most isolated but operationally complex
- Use for high-value or regulated tenants

### 2. Vector Store Isolation

**Pinecone Namespaces:**
```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("main-index")

# Tenant-specific namespace
index.upsert(
    vectors=[...],
    namespace=f"tenant_{tenant_id}",
)

# Query within tenant namespace
results = index.query(
    vector=[...],
    namespace=f"tenant_{tenant_id}",
    top_k=5,
)
```

**Qdrant Collections:**
```python
# Requiere: qdrant-client >= 1.6.0
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Collection per tenant
client.create_collection(
    collection_name=f"tenant_{tenant_id}",
    vectors_config={"size": 1536, "distance": "Cosine"},
)
```

**Metadata Filtering:**
```python
# Single collection with tenant_id metadata
index.query(
    vector=[...],
    filter={"tenant_id": {"$eq": tenant_id}},
    top_k=5,
)
```

### 3. Memory/Context Isolation

**Conversation Memory:**
```python
from collections import defaultdict

class TenantMemoryManager:
    def __init__(self):
        self.memories: dict[str, dict[str, list]] = defaultdict(dict)
    
    def get_memory(self, tenant_id: str, user_id: str) -> list:
        return self.memories[tenant_id].get(user_id, [])
    
    def add_message(self, tenant_id: str, user_id: str, message: dict) -> None:
        if user_id not in self.memories[tenant_id]:
            self.memories[tenant_id][user_id] = []
        self.memories[tenant_id][user_id].append(message)
```

**Cache Keys:**
```python
# Prefix all cache keys with tenant_id
cache_key = f"tenant:{tenant_id}:user:{user_id}:session:{session_id}"
```

---

## Implementation Patterns

### Middleware Pattern

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()


@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    """Extract and validate tenant ID from request."""
    # Extract tenant ID from header
    tenant_id = request.headers.get("X-Tenant-ID")
    
    if not tenant_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing X-Tenant-ID header"},
        )
    
    # Validate tenant exists and is active
    tenant = await get_tenant(tenant_id)
    if not tenant or not tenant.is_active:
        return JSONResponse(
            status_code=403,
            content={"error": "Invalid or inactive tenant"},
        )
    
    # Store in request state
    request.state.tenant_id = tenant_id
    request.state.tenant = tenant
    
    response = await call_next(request)
    return response
```

### Repository Pattern with Tenant

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

class TenantAwareRepository:
    """Repository with automatic tenant filtering."""
    
    def __init__(self, session: AsyncSession, tenant_id: str):
        self.session = session
        self.tenant_id = tenant_id
    
    async def get_documents(self) -> list[Document]:
        """Get documents for current tenant only."""
        result = await self.session.execute(
            select(Document).where(Document.tenant_id == self.tenant_id)
        )
        return result.scalars().all()
    
    async def create_document(self, content: str) -> Document:
        """Create document with tenant_id."""
        doc = Document(
            tenant_id=self.tenant_id,
            content=content,
        )
        self.session.add(doc)
        await self.session.commit()
        return doc
```

### Vector Store with Tenant

```python
from pinecone import Pinecone

class TenantVectorStore:
    """Vector store with tenant isolation."""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.pc = Pinecone(api_key="...")
        self.index = self.pc.Index("main-index")
        self.namespace = f"tenant_{tenant_id}"
    
    async def upsert(self, vectors: list[dict]) -> None:
        """Upsert vectors in tenant namespace."""
        self.index.upsert(
            vectors=vectors,
            namespace=self.namespace,
        )
    
    async def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """Query within tenant namespace."""
        results = self.index.query(
            vector=vector,
            namespace=self.namespace,
            top_k=top_k,
        )
        return results.matches
```

---

## Rate Limiting per Tenant

### Token Budget

```python
import redis.asyncio as redis
from datetime import datetime, timedelta

class TenantRateLimiter:
    """Per-tenant rate limiting."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_limit(
        self,
        tenant_id: str,
        tokens: int,
        daily_limit: int = 1_000_000,
    ) -> bool:
        """Check if tenant is within token budget."""
        key = f"tenant:{tenant_id}:tokens:{datetime.now().date()}"
        
        current = await self.redis.get(key)
        current_tokens = int(current) if current else 0
        
        if current_tokens + tokens > daily_limit:
            return False
        
        # Increment and set expiry
        await self.redis.incrby(key, tokens)
        await self.redis.expire(key, timedelta(days=1))
        
        return True
```

### Request Limits

```python
from fastapi import HTTPException

async def rate_limit_middleware(request: Request, call_next):
    """Rate limit requests per tenant."""
    tenant_id = request.state.tenant_id
    
    # Check rate limit (e.g., 100 requests per minute)
    key = f"tenant:{tenant_id}:requests:{int(time.time() // 60)}"
    
    count = await redis_client.incr(key)
    await redis_client.expire(key, 60)
    
    if count > 100:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )
    
    return await call_next(request)
```

---

## Cost Attribution

### Per-Tenant Cost Tracking

```python
class TenantCostTracker:
    """Track costs per tenant."""
    
    async def log_llm_usage(
        self,
        tenant_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Log usage and return cost."""
        cost = calculate_cost(model, input_tokens, output_tokens)
        
        await db.execute(
            """
            INSERT INTO tenant_usage (tenant_id, model, input_tokens, output_tokens, cost)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tenant_id, model, input_tokens, output_tokens, cost),
        )
        
        return cost
    
    async def get_monthly_cost(self, tenant_id: str) -> float:
        """Get current month cost for tenant."""
        result = await db.execute(
            """
            SELECT SUM(cost)
            FROM tenant_usage
            WHERE tenant_id = ? AND MONTH(created_at) = MONTH(NOW())
            """,
            (tenant_id,),
        )
        return result.scalar() or 0.0
```

---

## Security Considerations

### Tenant Context Leakage

**Risk:** LLM retains information from previous tenant's context

**Mitigation:**
- Clear conversation history between tenants
- Use separate LLM sessions per tenant
- Never reuse prompts with tenant-specific data

### Cross-Tenant Prompt Injection

**Risk:** Malicious tenant injects prompts to access other tenant data

**Mitigation:**
- Validate and sanitize all inputs
- Use separate vector store namespaces
- Implement strict access controls
- Log all cross-tenant access attempts

---

## Decision Tree: Isolation Strategy

```
START: What is your isolation requirement?
│
├─ Regulatory compliance (HIPAA, GDPR)
│  └─ Use SEPARATE DATABASES
│     - Maximum isolation
│     - Easier compliance audits
│     - Higher operational cost
│
├─ High-value enterprise customers
│  └─ Use SCHEMA PER TENANT
│     - Strong isolation
│     - Easier backup/restore per tenant
│     - Moderate operational cost
│
├─ Standard SaaS (many small tenants)
│  └─ Use ROW-LEVEL SECURITY
│     - Cost-effective
│     - Easier to manage
│     - Requires careful implementation
│
└─ Vector Store Isolation
   ├─ < 100 tenants → Collection per tenant
   ├─ 100-1000 tenants → Namespace per tenant
   └─ > 1000 tenants → Metadata filtering

ALWAYS: Implement rate limiting and cost tracking per tenant
```

---

## External Resources

### Multi-Tenancy Architecture
- **PostgreSQL Row-Level Security**: [postgresql.org/docs/current/ddl-rowsecurity.html](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
    - *Best for*: Row-level tenant isolation in shared-schema approach
- **SQLAlchemy Multi-Tenancy**: [docs.sqlalchemy.org/en/20/orm/extensions/horizontal_shard.html](https://docs.sqlalchemy.org/en/20/orm/extensions/horizontal_shard.html)
    - *Best for*: Horizontal sharding across tenant databases
- **django-tenants**: [django-tenants.readthedocs.io](https://django-tenants.readthedocs.io/)
    - *Best for*: Schema-per-tenant in Django (PostgreSQL schemas)

### Vector Store Isolation
- **Pinecone Namespaces**: [docs.pinecone.io/guides/indexes/use-namespaces](https://docs.pinecone.io/guides/indexes/use-namespaces)
    - *Best for*: Tenant isolation via namespaces in Pinecone
- **Qdrant Collections**: [qdrant.tech/documentation/concepts/collections](https://qdrant.tech/documentation/concepts/collections/)
    - *Best for*: Collection-per-tenant strategy in Qdrant
- **Weaviate Multi-Tenancy**: [weaviate.io/developers/weaviate/manage-data/multi-tenancy](https://weaviate.io/developers/weaviate/manage-data/multi-tenancy)
    - *Best for*: Native multi-tenancy support with tenant-aware sharding

### Security & Compliance
- **OWASP Multi-Tenancy Security**: [cheatsheetseries.owasp.org/cheatsheets/Multi-Tenancy_Security_Cheat_Sheet.html](https://cheatsheetseries.owasp.org/cheatsheets/Multi-Tenancy_Security_Cheat_Sheet.html)
    - *Best for*: Security checklist for multi-tenant applications
- **GDPR Data Isolation**: [gdpr-info.eu/art-25-gdpr](https://gdpr-info.eu/art-25-gdpr/)
    - *Best for*: Data protection by design requirements

### Rate Limiting & Cost
- **Redis Rate Limiting Patterns**: [redis.io/glossary/rate-limiting](https://redis.io/glossary/rate-limiting)
    - *Best for*: Token bucket and sliding window per-tenant rate limiting
- **LiteLLM Budget Manager**: [docs.litellm.ai/docs/budget_manager](https://docs.litellm.ai/docs/budget_manager)
    - *Best for*: Per-tenant LLM cost tracking and budget enforcement

---

## Instructions for the Agent

1. **Tenant ID Mandatory**: EVERY request must include tenant_id via header (`X-Tenant-ID`), JWT claim, or API key. Reject requests without tenant identification.

2. **Complete Isolation**: NEVER share context, memory, or cache between tenants. Each tenant must have completely isolated:
   - Database queries (with `tenant_id` filter)
   - Vector store namespaces/collections
   - Cache keys (prefixed with `tenant:{tenant_id}:`)
   - Conversation memory

3. **Middleware Validation**: Implement middleware that:
   - Extracts tenant_id from request
   - Validates tenant exists and is active
   - Injects tenant_id into request.state
   - Logs all tenant access

4. **Repository Pattern**: All data access must use tenant-aware repositories that:
   - Accept tenant_id in constructor
   - Automatically filter queries by tenant_id
   - Prevent cross-tenant data access
   - Validate tenant ownership before writes

5. **Vector Store Isolation**: Choose strategy based on scale:
   - < 100 tenants: Collection per tenant
   - 100-1000 tenants: Namespace per tenant (Pinecone)
   - > 1000 tenants: Metadata filtering with strict validation

6. **Rate Limiting Per Tenant**: Implement separate rate limits for each tenant:
   - Requests per minute/hour
   - Tokens per day
   - Cost per month
   - Use Redis with tenant-scoped keys

7. **Cost Attribution**: Track costs per tenant for every LLM call:
   - Log model, input/output tokens, cost
   - Store in tenant_usage table
   - Calculate monthly costs for billing
   - Alert when tenant exceeds budget

8. **Security Hardening**:
   - Row-Level Security (RLS) in PostgreSQL
   - Separate secrets per tenant (API keys, credentials)
   - Validate tenant access on EVERY operation
   - Log cross-tenant access attempts
   - Clear context between tenant requests

9. **Isolation Strategy**: Choose based on requirements:
   - Regulatory compliance (HIPAA, GDPR) → Separate databases
   - Enterprise customers → Schema per tenant
   - Standard SaaS → Row-Level Security
   - Always document decision in ADR

10. **Prevent Context Leakage**:
    - Clear LLM conversation history between tenants
    - Use separate sessions per tenant
    - Never reuse prompts with tenant-specific data
    - Sanitize all inputs to prevent prompt injection
