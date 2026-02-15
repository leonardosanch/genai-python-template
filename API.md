# APIs & Web Frameworks

## Frameworks Python

### FastAPI (Recomendado para GenAI)

Framework async-first. Ideal para sistemas GenAI por soporte nativo de SSE, WebSockets y async.

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GenAI API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    tokens_used: int

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await chat_use_case.execute(request.message)
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        tokens_used=result.tokens_used,
    )
```

**Cuándo usar FastAPI:**
- APIs async (LLM calls, I/O intensivo)
- Streaming (SSE, WebSockets)
- Microservicios
- Cuando Pydantic ya es parte del stack

### Django REST Framework (DRF)

Framework full-stack con ORM, admin, auth, y REST out-of-the-box. Ideal cuando necesitas un backend completo con base de datos relacional.

```python
# serializers.py
from rest_framework import serializers

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "title", "content", "created_at"]

class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=4096)
    model = serializers.CharField(default="gpt-4o")

class ChatResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    sources = serializers.ListField(child=serializers.CharField())
    tokens_used = serializers.IntegerField()
```

```python
# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from adrf.views import APIView as AsyncAPIView  # django-rest-framework async

class ChatView(AsyncAPIView):
    async def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = await chat_use_case.execute(serializer.validated_data["message"])
        return Response(
            ChatResponseSerializer(result).data,
            status=status.HTTP_200_OK,
        )
```

```python
# urls.py
from django.urls import path
from .views import ChatView, DocumentViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register("documents", DocumentViewSet)

urlpatterns = [
    path("api/chat/", ChatView.as_view()),
] + router.urls
```

**Cuándo usar Django + DRF:**
- Apps con modelos de datos complejos y relaciones
- Necesitas admin panel, auth, migrations out-of-the-box
- CRUD completo sobre base de datos relacional
- Apps monolíticas o modulares con Django apps
- Equipos familiarizados con Django

### Flask

Framework minimalista. Máxima flexibilidad, mínimas opiniones.

```python
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError

app = Flask(__name__)

@app.post("/api/chat")
def chat():
    try:
        data = ChatRequest(**request.json)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    result = chat_use_case.execute(data.message)
    return jsonify({
        "answer": result.answer,
        "sources": result.sources,
    })
```

**Cuándo usar Flask:**
- APIs simples o prototipos rápidos
- Microservicios ligeros
- Cuando necesitas control total sin opiniones del framework
- Integrar con async manualmente (Quart como alternativa async)

### Comparativa

| Característica | FastAPI | Django + DRF | Flask |
|---------------|---------|-------------|-------|
| Async nativo | Si | Parcial (ASGI) | No (Quart sí) |
| ORM integrado | No | Si (Django ORM) | No |
| Admin panel | No | Si | No |
| Auth built-in | No | Si | No |
| Streaming (SSE/WS) | Si | Limitado | Limitado |
| Validación | Pydantic | Serializers | Manual/Pydantic |
| OpenAPI/Swagger | Automático | drf-spectacular | Flask-RESTX |
| Ideal para | GenAI APIs, microservicios | Apps full-stack, CRUD | Microservicios simples |
| Curva aprendizaje | Baja | Media-Alta | Baja |

---

## Patrones de API

### REST

```python
# Recursos y verbos HTTP estándar
GET    /api/documents/          # Listar
POST   /api/documents/          # Crear
GET    /api/documents/{id}      # Obtener
PUT    /api/documents/{id}      # Actualizar completo
PATCH  /api/documents/{id}      # Actualizar parcial
DELETE /api/documents/{id}      # Eliminar

# GenAI-specific
POST   /api/chat                # Chat completion
POST   /api/chat/stream         # Chat con streaming (SSE)
POST   /api/embeddings          # Generar embeddings
POST   /api/documents/search    # Búsqueda semántica
```

### Pagination

```python
from pydantic import BaseModel

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    has_next: bool

@app.get("/api/documents")
async def list_documents(page: int = 1, page_size: int = 20):
    offset = (page - 1) * page_size
    items = await repo.list(offset=offset, limit=page_size)
    total = await repo.count()
    return PaginatedResponse(
        items=items, total=total, page=page,
        page_size=page_size, has_next=(offset + page_size < total),
    )
```

### Error Handling

```python
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: str

@app.exception_handler(DomainError)
async def domain_error_handler(request, exc: DomainError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error=str(exc), code="DOMAIN_ERROR").model_dump(),
    )

@app.exception_handler(LLMTimeoutError)
async def llm_timeout_handler(request, exc: LLMTimeoutError):
    return JSONResponse(
        status_code=504,
        content=ErrorResponse(error="LLM request timed out", code="LLM_TIMEOUT").model_dump(),
    )
```

### Middleware

```python
# Authentication
from fastapi import Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    user = await verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["https://app.example.com"])

# Request ID / Correlation ID
@app.middleware("http")
async def add_correlation_id(request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response
```

---

## Microservicios

### Principios

1. **Single responsibility**: Cada servicio tiene un dominio acotado
2. **Independently deployable**: Cada servicio se despliega por separado
3. **Database per service**: Cada servicio tiene su propia base de datos
4. **API contracts**: Comunicación via API bien definida
5. **Resilience**: Circuit breakers, retries, timeouts

### Patrones de Comunicación

| Patrón | Uso | Ejemplo |
|--------|-----|---------|
| REST sync | Request-response directo | CRUD, queries |
| gRPC | Alta performance, tipado fuerte | Inter-service calls |
| Event-driven (async) | Desacoplamiento, eventual consistency | Pub/sub con Kafka/RabbitMQ |
| API Gateway | Punto de entrada único | Kong, Traefik, AWS API Gateway |

### Arquitectura de Microservicios GenAI

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  API Gateway │────→│  Chat Service │────→│  LLM Service  │
│  (FastAPI)   │     │  (FastAPI)   │     │  (FastAPI)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  RAG Service  │────→ Vector DB
                    │  (FastAPI)   │
                    └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  Doc Service  │────→ PostgreSQL
                    │  (Django+DRF)│
                    └──────────────┘
```

### Service Communication

```python
# HTTP client con retries y circuit breaker
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMServiceClient:
    def __init__(self, base_url: str):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(self, prompt: str) -> str:
        response = await self._client.post(
            "/api/generate",
            json={"prompt": prompt},
        )
        response.raise_for_status()
        return response.json()["content"]
```

### Event-Driven

```python
# Publicar evento
import aio_pika

async def publish_event(event_type: str, data: dict):
    connection = await aio_pika.connect_robust("amqp://localhost")
    async with connection:
        channel = await connection.channel()
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps({"type": event_type, "data": data}).encode(),
            ),
            routing_key="events",
        )

# Uso
await publish_event("document.indexed", {"document_id": "123", "chunks": 15})
```

---

## GraphQL (Alternativa)

```python
# Con Strawberry (async-first, type-safe)
import strawberry
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class DocumentType:
    id: str
    title: str
    content: str

@strawberry.type
class Query:
    @strawberry.field
    async def documents(self, search: str | None = None) -> list[DocumentType]:
        if search:
            return await repo.search(search)
        return await repo.list()

schema = strawberry.Schema(query=Query)
app.include_router(GraphQLRouter(schema), prefix="/graphql")
```

---

## API Versioning

```python
# Por path prefix
app_v1 = FastAPI(prefix="/api/v1")
app_v2 = FastAPI(prefix="/api/v2")

# O por router
from fastapi import APIRouter
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.post("/chat")
async def chat_v1(request: ChatRequestV1): ...

@v2_router.post("/chat")
async def chat_v2(request: ChatRequestV2): ...
```

---

## Reglas

1. **FastAPI para APIs GenAI** — async, streaming, Pydantic nativo
2. **Django+DRF para CRUD/full-stack** — cuando necesitas ORM, admin, auth
3. **Flask para microservicios simples** — mínimo overhead
4. **Siempre validar inputs** — Pydantic o serializers, nunca confiar en el cliente
5. **API versioning desde el inicio** — evita breaking changes
6. **Health checks obligatorios** — `/health` y `/ready` en todo servicio
7. **Correlation ID en toda request** — para distributed tracing

8. **Security middleware stack** — headers, request size, JWT auth, rate limiting
9. **JWT authentication** — access + refresh tokens via `/api/v1/auth/token` y `/api/v1/auth/refresh`
10. **CORS restrictivo** — nunca `allow_origins=["*"]` en produccion

### Middleware Stack (orden de ejecucion)

| # | Middleware | Proposito |
|---|-----------|-----------|
| 1 | LoggingMiddleware | Structured logging con correlation ID |
| 2 | SecurityHeadersMiddleware | HSTS, CSP, X-Frame-Options |
| 3 | RequestSizeMiddleware | Rechaza payloads > MAX_REQUEST_SIZE |
| 4 | CORSMiddleware | Origins configurables via CORS_ORIGINS |
| 5 | RateLimitMiddleware | Token bucket (memory) o sliding window (Redis) |
| 6 | JWTAuthMiddleware | Valida Bearer token, inyecta claims |
| 7 | AuthMiddleware | API key validation (opcional) |

Ver también: [STREAMING.md](STREAMING.md), [SECURITY.md](SECURITY.md), [DEPLOYMENT.md](DEPLOYMENT.md)
