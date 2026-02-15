# Skill: API & Streaming

## Description
This skill focuses on building high-performance, real-time APIs using modern Python frameworks and streaming protocols. Use this when implementing REST APIs, GraphQL, WebSockets, or streaming LLM responses.

## Executive Summary

**Critical async/streaming rules:**
- ALL I/O operations MUST be async — never block the event loop (no `time.sleep`, no `requests`)
- SSE for LLM streaming (unidirectional), WebSockets for chat (bidirectional) — consult Decision Tree 1
- Every endpoint needs a Pydantic model for input AND output — type safety is non-negotiable
- API versioning from day one (`/api/v1/`) — deprecation is easier than breaking changes
- Rate limiting mandatory on all public endpoints — use SlowAPI with per-IP and per-user tiers

**Read full skill when:** Implementing FastAPI endpoints, streaming LLM responses, setting up WebSockets, configuring rate limiting, or deploying async applications to production.

---

## Versiones y Rendimiento

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| uvicorn | >= 0.20.0 | Soporte mejorado para ASGI 3.0 |
| httpx | >= 0.25.0 | Cliente async estándar |
| strawberry-graphql | >= 0.200.0 | Soporte para Pydantic V2 |
| slowapi | >= 0.1.9 | Rate limiting para FastAPI |

### Async Safety

```python
import asyncio

# ❌ NUNCA usar time.sleep
# await asyncio.sleep(0)  # Cede el control al event loop

# ✅ SIEMPRE usar awaitable I/O
async def call_external_api():
    async with httpx.AsyncClient() as client:
        return await client.get("https://api.example.com")

# ✅ Ejecutar código síncrono bloqueante en threadpool
from fastapi.concurrency import run_in_threadpool
result = await run_in_threadpool(blocking_function, arg1, arg2)
```

---

## Deep Dive

## Core Concepts

1.  **Asynchronous Programming**: Non-blocking I/O using `asyncio` for high concurrency (C10K problem).
2.  **Streaming Protocols**: SSE (Server-Sent Events) for unidirectional updates, WebSockets for bidirectional.
3.  **Type Safety**: Leveraging Pydantic for request/response validation, serialization, and documentation.
4.  **Resilience**: Rate limiting, timeouts, circuit breakers, and graceful degradation.

---

## External Resources

### ⚡ Frameworks & Libraries

#### FastAPI Ecosystem
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
    - *Best for*: Modern, type-safe API development, auto-generated docs
- **Pydantic**: [docs.pydantic.dev](https://docs.pydantic.dev/)
    - *Best for*: Data validation, settings management, serialization
- **Starlette**: [www.starlette.io](https://www.starlette.io/)
    - *Best for*: Underlying ASGI toolkit (WebSockets, Background Tasks)
- **Uvicorn**: [www.uvicorn.org](https://www.uvicorn.org/)
    - *Best for*: ASIC server implementation

#### Async Python
- **Asyncio Documentation**: [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
    - *Best for*: Event loops, coroutines, tasks
- **AnyIO**: [anyio.readthedocs.io](https://anyio.readthedocs.io/)
    - *Best for*: High-level async compatibility (Trio/Asyncio)
- **HTTPX**: [www.python-httpx.org](https://www.python-httpx.org/)
    - *Best for*: Modern, async-native HTTP client

---

### 📡 Streaming & Real-Time Protocols

#### Server-Sent Events (SSE)
- **MDN SSE API**: [developer.mozilla.org/en-US/docs/Web/API/Server-sent_events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
    - *Best for*: Unidirectional updates (ideal for LLM streaming)
- **FastAPI StreamingResponse**: [fastapi.tiangolo.com/advanced/custom-response/#streamingresponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
    - *Best for*: Implementing SSE in FastAPI

#### WebSockets
- **WebSocket Protocol (RFC 6455)**: [datatracker.ietf.org/doc/html/rfc6455](https://datatracker.ietf.org/doc/html/rfc6455)
    - *Best for*: Full-duplex communication, low latency
- **FastAPI WebSockets**: [fastapi.tiangolo.com/advanced/websockets/](https://fastapi.tiangolo.com/advanced/websockets/)
    - *Best for*: Handling WebSocket connections, authentication

#### gRPC
- **gRPC Python**: [grpc.io/docs/languages/python/](https://grpc.io/docs/languages/python/)
    - *Best for*: High-performance microservices, strict contracts
- **Protobuf**: [protobuf.dev](https://protobuf.dev/)
    - *Best for*: Efficient binary serialization

---

### 🛡️ API Security & Performance

#### Security
- **OAuth 2.0 & OpenID Connect**: [oauth.net/2/](https://oauth.net/2/)
    - *Best for*: Authentication and authorization standards
- **FastAPI Security**: [fastapi.tiangolo.com/tutorial/security/](https://fastapi.tiangolo.com/tutorial/security/)
    - *Best for*: Implementing OAuth2, JWT, API Keys
- **Create a Rate Limiter**: [slowapi.readthedocs.io](https://slowapi.readthedocs.io/)
    - *Best for*: Rate limiting for FastAPI

#### Performance
- **Gunicorn**: [gunicorn.org](https://gunicorn.org/)
    - *Best for*: Production process manager
- **NGINX**: [nginx.org](https://nginx.org/)
    - *Best for*: Reverse proxy, load balancing, caching

---

### 🧪 Testing & Documentation

#### Testing Async APIs
- **Pytest Asyncio**: [pytest-asyncio.readthedocs.io](https://pytest-asyncio.readthedocs.io/)
    - *Best for*: Testing async code
- **Polyfactory**: [polyfactory.litestar.dev](https://polyfactory.litestar.dev/)
    - *Best for*: Generating mock data for Pydantic models

#### Documentation Standards
- **OpenAPI Specification (OAS)**: [swagger.io/specification/](https://swagger.io/specification/)
    - *Best for*: Standardizing API definitions
- **Swagger UI**: [swagger.io/tools/swagger-ui/](https://swagger.io/tools/swagger-ui/)
    - *Best for*: Interactive API documentation
- **Redoc**: [redocly.com/redoc/](https://redocly.com/redoc/)
    - *Best for*: Beautiful, readable API docs

---

### 📖 Books & Courses

#### Books
- **High Performance Python** (Micha Gorelick, Ian Ozsvald)
    - *Best for*: Profiling, optimizing, concurrency
- **Architecture Patterns with Python** (Harry Percival, Bob Gregory)
    - *Best for*: Test-Driven Development (TDD), Domain-Driven Design (DDD)
- **API Design Patterns** (JJ Geewax)
    - *Best for*: API resource layout, standard methods

#### Guides
- **The Twelve-Factor App**: [12factor.net](https://12factor.net/)
    - *Best for*: Methodology for building SaaS apps
- **Real Python - Async IO**: [realpython.com/async-io-python/](https://realpython.com/async-io-python/)
    - *Best for*: Deep dive into Python's async model

---

## Decision Trees

### Decision Tree 1: SSE vs WebSocket vs gRPC

```
¿Qué tipo de comunicación necesitas?
├── Unidireccional server → client (LLM streaming, notifications)
│   └── SSE (Server-Sent Events)
│       ├── Simple, HTTP nativo, auto-reconnect
│       ├── Ideal para: streaming tokens del LLM
│       └── No sirve si el client necesita enviar durante el stream
├── Bidireccional real-time (chat, colaboración)
│   └── WebSocket
│       ├── Full-duplex, baja latencia
│       ├── Ideal para: chat interactivo con agentes
│       └── Más complejo: reconexión, heartbeat, auth
├── Inter-service high performance
│   └── gRPC
│       ├── Protobuf, HTTP/2, streaming bidireccional
│       ├── Ideal para: microservicio → microservicio
│       └── No para browser clients (necesita grpc-web proxy)
└── No estoy seguro
    └── Empieza con SSE. Migrar a WebSocket solo si necesitas bidireccional.
```

### Decision Tree 2: Qué framework web usar

```
¿Qué tipo de aplicación construyes?
├── API async para GenAI (RAG, agents, streaming)
│   └── FastAPI (siempre para GenAI)
│       ├── Async-first, Pydantic nativo, SSE/WS
│       └── Auto-docs con OpenAPI
├── CRUD con admin, auth, ORM completo
│   └── Django + DRF
│       ├── Batteries included
│       └── No ideal para streaming LLM
├── Microservicio simple, mínima superficie
│   └── Flask
│       └── Solo si la simplicidad es requisito
└── GraphQL
    └── Strawberry (async, type-safe) sobre FastAPI
```

---

## Instructions for the Agent

1.  **Async First**: All I/O operations (Database, LLM calls, File I/O, Network) MUST be `async`. Never block the event loop.
2.  **Streaming**: 
    - For LLM text generation, use Server-Sent Events (SSE) via `StreamingResponse`.
    - Use `yield` generators for distinct "events" or "tokens".
    - Handle client functionality cleanly (disconnects).
3.  **Data Validation**: 
    - Every endpoint input (Request Body, Query Params) and output (Response Body) MUST have a typed Pydantic model.
    - Use `Field(..., description="...")` to document fields for OpenAPI.
4.  **Error Handling**: 
    - Use custom exception handlers to return consistent JSON error schemas (`{"code": "...", "message": "..."}`).
    - Never expose internal stack traces to the client.
5.  **Concurrency**:
    - Use `asyncio.gather` for parallel independent tasks.
    - Use `asyncio.Semaphore` to limit concurrency for external API calls (rate limits).
6.  **Versioning**: 
    - Always version APIs (e.g., `/api/v1/...`).
    - Deprecate old versions gracefully.

---

## Code Examples

### Example 1: SSE Streaming (see examples/fastapi_streaming.py)

Complete implementation available in `docs/skills/examples/fastapi_streaming.py`

### Example 2: SSE LLM Streaming with Token Events

```python
# src/interfaces/api/routes/stream_routes.py
import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])


class StreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default="gpt-4")
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


async def sse_generator(
    llm_client: Any,
    request: StreamRequest,
    client_request: Request,
) -> AsyncGenerator[str, None]:
    """SSE event generator with proper disconnect handling."""
    start_time = time.perf_counter()
    token_count = 0

    try:
        async for chunk in llm_client.astream(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            # Check if client disconnected
            if await client_request.is_disconnected():
                logger.info("Client disconnected, stopping stream")
                break

            token_count += 1
            event_data = json.dumps({
                "token": chunk,
                "index": token_count,
            })
            yield f"data: {event_data}\n\n"

        # Send final event with metadata
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        done_data = json.dumps({
            "done": True,
            "total_tokens": token_count,
            "duration_ms": round(elapsed_ms, 2),
        })
        yield f"event: done\ndata: {done_data}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        error_data = json.dumps({"error": str(e)})
        yield f"event: error\ndata: {error_data}\n\n"


@router.post("/generate")
async def stream_generate(
    request: StreamRequest,
    client_request: Request,
    llm_client: Any = Depends(get_llm_client),
):
    """Stream LLM response as Server-Sent Events."""
    return StreamingResponse(
        sse_generator(llm_client, request, client_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable NGINX buffering
        },
    )
```

### Example 3: WebSocket Chat with Heartbeat

```python
# src/interfaces/api/routes/ws_chat.py
import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()

HEARTBEAT_INTERVAL = 30  # seconds
MAX_MESSAGE_SIZE = 8192  # bytes


class ChatMessage(BaseModel):
    type: str  # "message", "ping"
    content: str = ""
    conversation_id: str | None = None


class ConnectionManager:
    """Manage active WebSocket connections per conversation."""

    def __init__(self) -> None:
        self._active: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, conversation_id: str) -> None:
        await ws.accept()
        self._active.setdefault(conversation_id, []).append(ws)

    def disconnect(self, ws: WebSocket, conversation_id: str) -> None:
        if conversation_id in self._active:
            self._active[conversation_id] = [
                c for c in self._active[conversation_id] if c != ws
            ]

    async def broadcast(self, conversation_id: str, message: dict) -> None:
        for ws in self._active.get(conversation_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws, conversation_id)


manager = ConnectionManager()


@router.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(
    ws: WebSocket,
    conversation_id: str,
    llm_client: Any = Depends(get_llm_client),
):
    await manager.connect(ws, conversation_id)
    heartbeat_task: asyncio.Task | None = None

    async def send_heartbeat():
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break

    try:
        heartbeat_task = asyncio.create_task(send_heartbeat())

        while True:
            raw = await ws.receive_text()
            if len(raw) > MAX_MESSAGE_SIZE:
                await ws.send_json({"type": "error", "content": "Message too large"})
                continue

            try:
                msg = ChatMessage.model_validate_json(raw)
            except ValidationError as e:
                await ws.send_json({"type": "error", "content": str(e)})
                continue

            if msg.type == "ping":
                await ws.send_json({"type": "pong"})
                continue

            # Stream LLM response token by token
            await ws.send_json({"type": "start", "content": ""})
            async for token in llm_client.astream(prompt=msg.content):
                await ws.send_json({"type": "token", "content": token})
            await ws.send_json({"type": "end", "content": ""})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from conversation {conversation_id}")
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        manager.disconnect(ws, conversation_id)
```

### Example 4: Rate Limiting with Redis Sliding Window

```python
# src/interfaces/api/middleware/rate_limit_middleware.py
from __future__ import annotations

import time
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitConfig:
    """Per-tier rate limiting configuration."""

    TIERS: dict[str, dict[str, int]] = {
        "free": {"requests_per_minute": 10, "requests_per_hour": 100},
        "pro": {"requests_per_minute": 60, "requests_per_hour": 1000},
        "enterprise": {"requests_per_minute": 300, "requests_per_hour": 10000},
    }


class SlidingWindowRateLimiter:
    """Redis-based sliding window rate limiter."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        now = time.time()
        window_start = now - window_seconds
        pipe = self._redis.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Count requests in window
        pipe.zcard(key)
        # Set expiry on the key
        pipe.expire(key, window_seconds)

        results = await pipe.execute()
        request_count = results[2]
        remaining = max(0, max_requests - request_count)

        return request_count <= max_requests, remaining


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, redis_client: Any) -> None:
        super().__init__(app)
        self._limiter = SlidingWindowRateLimiter(redis_client)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract client identity
        client_ip = request.client.host if request.client else "unknown"
        tier = getattr(request.state, "user_tier", "free")
        config = RateLimitConfig.TIERS.get(tier, RateLimitConfig.TIERS["free"])

        # Check per-minute limit
        key = f"rate:{client_ip}:minute"
        allowed, remaining = await self._limiter.is_allowed(
            key, config["requests_per_minute"], 60,
        )

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "rate_limit_exceeded", "retry_after_seconds": 60},
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = str(config["requests_per_minute"])
        return response
```

### Example 5: Dependency Injection with FastAPI

```python
# src/interfaces/api/dependencies.py
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, AsyncGenerator

from fastapi import Depends, Request

from src.application.services.llm_service import LLMService
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.llm.openai_adapter import OpenAIAdapter
from src.infrastructure.config import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


async def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[LLMPort, None]:
    """Provide LLM client with proper lifecycle management."""
    client = OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.default_model,
        timeout=settings.llm_timeout,
    )
    try:
        yield client
    finally:
        await client.close()


async def get_llm_service(
    llm_client: Annotated[LLMPort, Depends(get_llm_client)],
) -> LLMService:
    return LLMService(llm_client=llm_client)


# Usage in routes:
# @router.post("/generate")
# async def generate(
#     request: GenerateRequest,
#     service: Annotated[LLMService, Depends(get_llm_service)],
# ):
#     return await service.generate(request.prompt)
```

### Example 6: Background Tasks with Progress Tracking

```python
# src/interfaces/api/routes/task_routes.py
from __future__ import annotations

import asyncio
import uuid
from enum import Enum

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    result: str | None = None
    error: str | None = None


# In production, use Redis or a database
_task_store: dict[str, TaskResult] = {}


async def run_long_task(task_id: str, payload: dict) -> None:
    """Execute a long-running task with progress updates."""
    _task_store[task_id].status = TaskStatus.RUNNING
    try:
        total_steps = payload.get("steps", 10)
        for i in range(total_steps):
            await asyncio.sleep(0.5)  # Simulate work
            _task_store[task_id].progress = (i + 1) / total_steps

        _task_store[task_id].status = TaskStatus.COMPLETED
        _task_store[task_id].result = "Task completed successfully"
    except Exception as e:
        _task_store[task_id].status = TaskStatus.FAILED
        _task_store[task_id].error = str(e)


@router.post("/", response_model=TaskResult, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    payload: dict,
    background_tasks: BackgroundTasks,
):
    """Create a long-running task and return immediately."""
    task_id = str(uuid.uuid4())
    task = TaskResult(task_id=task_id, status=TaskStatus.PENDING)
    _task_store[task_id] = task
    background_tasks.add_task(run_long_task, task_id, payload)
    return task


@router.get("/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """Poll task status and progress."""
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_store[task_id]
```

### Example 7: Health Check and Readiness Probes

```python
# src/interfaces/api/routes/health_routes.py
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["health"])

START_TIME = time.time()


class HealthStatus(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    checks: dict[str, bool]


async def check_database(db: Any) -> bool:
    try:
        await db.execute("SELECT 1")
        return True
    except Exception:
        return False


async def check_redis(redis: Any) -> bool:
    try:
        await redis.ping()
        return True
    except Exception:
        return False


async def check_llm(llm_client: Any) -> bool:
    try:
        await llm_client.generate("ping", max_tokens=1)
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthStatus)
async def health():
    """Liveness probe — is the service running?"""
    return HealthStatus(
        status="healthy",
        uptime_seconds=round(time.time() - START_TIME, 2),
        checks={"alive": True},
    )


@router.get("/ready")
async def readiness(
    db: Any = Depends(get_database),
    redis: Any = Depends(get_redis),
):
    """Readiness probe — can the service handle requests?"""
    checks = {
        "database": await check_database(db),
        "redis": await check_redis(redis),
    }
    all_healthy = all(checks.values())

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "checks": checks,
        },
    )
```

### Example 8: Graceful Shutdown Handler

```python
# src/interfaces/api/lifecycle.py
from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle with graceful shutdown.

    Handles SIGTERM/SIGINT for clean shutdown of:
    - Active WebSocket connections
    - Background tasks
    - Database connection pools
    - Redis connections
    """
    # Startup
    logger.info("Starting application...")
    app.state.db_pool = await create_db_pool()
    app.state.redis = await create_redis_client()
    app.state.shutting_down = False
    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down gracefully...")
    app.state.shutting_down = True

    # Wait for active requests to complete (max 30s)
    shutdown_timeout = 30
    logger.info(f"Waiting up to {shutdown_timeout}s for active requests...")
    await asyncio.sleep(2)  # Grace period for in-flight requests

    # Close connection pools
    if hasattr(app.state, "db_pool"):
        await app.state.db_pool.close()
        logger.info("Database pool closed")

    if hasattr(app.state, "redis"):
        await app.state.redis.close()
        logger.info("Redis connection closed")

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="GenAI API",
        version="1.0.0",
        lifespan=lifespan,
    )
    return app
```

### Example 9: Global Error Handling

```python
# src/api/middleware/error_handler.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "validation_error", "message": str(e)}
            )
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "internal_error", "message": "An unexpected error occurred"}
            )
```

### Example 3: API Versioning Strategy

```python
# src/api/router.py
from fastapi import FastAPI, APIRouter

app = FastAPI()

# V1 Router - Deprecated but supported
router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])

@router_v1.post("/generate")
async def generate_v1(prompt: str):
    return {"text": f"V1: {prompt}"}

# V2 Router - Current
router_v2 = APIRouter(prefix="/api/v2", tags=["v2"])

@router_v2.post("/generate")
async def generate_v2(request: GenerationRequest):
    """Enhanced generation with parameters."""
    return {"text": f"V2: {request.prompt}", "meta": request.metadata}

app.include_router(router_v1)
app.include_router(router_v2)
```

---

## Anti-Patterns to Avoid

### ❌ Blocking I/O in Async
**Problem**: Blocks the event loop, killing concurrency  
**Example**:
```python
# BAD: Blocking I/O
@app.get("/")
async def root():
    time.sleep(1)  # Blocks ALL requests
    return "ok"
```
**Solution**: Use `await asyncio.sleep(1)` or run in threadpool

### ❌ No Rate Limiting
**Problem**: API abuse, potential DoS, high LLM costs  
**Solution**: Implement `SlowAPI` or Redis-based rate limiting

### ❌ Returning Raw LLM Errors
**Problem**: Leaks implementation details or confusing errors to users
**Solution**: Map LLM errors (ContextWindowExceeded, RateLimit) to clean HTTP 4xx/5xx responses

### ❌ No Client Disconnect Handling in SSE
**Problem**: LLM keeps generating tokens after client disconnects. Wastes money and compute.
**Solution**: Check `await request.is_disconnected()` in the SSE generator loop (see Example 2)

### ❌ Synchronous Dependencies in Async Routes
**Problem**: Using sync database drivers or sync HTTP clients in async FastAPI routes blocks the event loop.
```python
# BAD: sync driver in async route
@app.get("/users")
async def get_users():
    conn = psycopg2.connect(...)  # Blocks event loop
    return conn.execute("SELECT * FROM users")
```
**Solution**: Use async drivers (`asyncpg`, `httpx.AsyncClient`) or `run_in_threadpool` for unavoidable sync code

### ❌ Unbuffered Streaming Without Backpressure
**Problem**: SSE generator produces tokens faster than client can consume. Memory grows.
**Solution**: Use `asyncio.Queue` with maxsize as buffer between LLM stream and SSE response

### ❌ WebSocket Without Authentication
**Problem**: WebSocket endpoint accepts connections without validating auth tokens.
**Solution**: Validate JWT/API key during the WebSocket handshake (before `accept()`), not after

---

## API Deployment Checklist

### Security & Performance
- [ ] Rate limiting configured (Global + per usage tier)
- [ ] CORS policies restrictive (allow specific origins)
- [ ] Timeouts configured (Gunicorn/Uvicorn + Client side)
- [ ] Input validation (Pydantic models for everything)

### Streaming
- [ ] SSE/WebSockets keep-alive configured
- [ ] Client disconnect handling (cancellation of LLM tasks)
- [ ] Buffer flushing strategy suitable for token generation

### Observability
- [ ] Request ID generation middleware
- [ ] Structured logging for all requests
- [ ] Metrics (latency, error rate, throughput) exposed

### Documentation
- [ ] OpenAPI schema valid
- [ ] Examples provided for all endpoints
- [ ] Versioning strategy clear

---

## Additional References

- **FastAPI Advanced User Guide**: [fastapi.tiangolo.com/advanced/](https://fastapi.tiangolo.com/advanced/)
    - *Best for*: Middleware, testing, and security
- **Real Python Async IO**: [realpython.com/async-io-python/](https://realpython.com/async-io-python/)
    - *Best for*: Understanding the event loop
- **Server-Sent Events (MDN)**: [developer.mozilla.org/en-US/docs/Web/API/Server-sent_events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
    - *Best for*: Protocol understanding

---

## API Security Middleware Stack

The template includes a production-ready middleware stack applied in this order (outermost to innermost):

1. **LoggingMiddleware** — Structured request/response logging with correlation IDs
2. **SecurityHeadersMiddleware** — HSTS, X-Frame-Options, CSP, Referrer-Policy, Permissions-Policy
3. **RequestSizeMiddleware** — Rejects payloads exceeding `MAX_REQUEST_SIZE` (default 1MB) with 413
4. **CORSMiddleware** — Configurable origins via `CORS_ORIGINS` (no more `allow_origins=["*"]`)
5. **RateLimitMiddleware** — Token bucket (in-memory) or sliding window (Redis) per client IP
6. **JWTAuthMiddleware** — Validates `Authorization: Bearer <JWT>`, injects claims into `request.state.user`
7. **AuthMiddleware** — API key validation via `X-API-Key` header (optional, if `API_KEYS` configured)

### Key files
- `src/interfaces/api/middleware/security_headers_middleware.py`
- `src/interfaces/api/middleware/request_size_middleware.py`
- `src/interfaces/api/middleware/jwt_auth_middleware.py`
- `src/interfaces/api/middleware/rate_limit_middleware.py`
- `src/infrastructure/security/jwt_handler.py`
- `src/infrastructure/security/password_handler.py`
- `src/interfaces/api/routes/auth_routes.py`
