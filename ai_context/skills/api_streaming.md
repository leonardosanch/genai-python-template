# Skill: API & Streaming

## Description
This skill focuses on building high-performance, real-time APIs using modern Python frameworks and streaming protocols. Use this when implementing REST APIs, GraphQL, WebSockets, or streaming LLM responses.

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

### Example 2: Global Error Handling

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
