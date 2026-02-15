"""FastAPI application entry point.

Reference implementation showing:
- Health check endpoints
- CORS and security middleware
- JWT authentication
- Dependency injection at the composition root
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError as PydanticValidationError

from src.domain.exceptions import DomainError
from src.infrastructure.config import get_settings
from src.infrastructure.container import Container
from src.infrastructure.governance.in_memory_audit_trail import InMemoryAuditTrail
from src.infrastructure.observability.logging import setup_logging
from src.infrastructure.observability.tracing import setup_tracing
from src.interfaces.api.middleware import (
    AuditMiddleware,
    AuthMiddleware,
    InputSanitizationMiddleware,
    JWTAuthMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RequestSizeMiddleware,
    SecurityHeadersMiddleware,
    domain_error_exception_handler,
    http_exception_handler,
    pydantic_validation_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from src.interfaces.api.routes.analytics_routes import router as analytics_router
from src.interfaces.api.routes.auth_routes import router as auth_router
from src.interfaces.api.routes.data_routes import router as data_router
from src.interfaces.api.routes.pipeline_routes import router as pipeline_router
from src.interfaces.api.routes.rag_routes import router as rag_router
from src.interfaces.api.routes.spark_routes import router as spark_router
from src.interfaces.api.routes.stream_routes import router as stream_router
from src.interfaces.api.routes.ws_routes import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown."""
    settings = get_settings()
    setup_logging(settings)
    setup_tracing(settings)
    app.state.container = Container(settings=settings)
    yield
    await app.state.container.close()


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- Routes ---
    app.include_router(analytics_router)
    app.include_router(auth_router)
    app.include_router(pipeline_router)
    app.include_router(rag_router)
    app.include_router(stream_router)
    app.include_router(data_router)
    app.include_router(ws_router)
    app.include_router(spark_router)

    # --- Middleware (order: last added = outermost) ---
    # Outermost → innermost: logging → security_headers → request_size → CORS → rate_limit → auth

    # Input sanitization (detect prompt injection)
    app.add_middleware(InputSanitizationMiddleware)

    # Audit trail
    audit_trail = InMemoryAuditTrail()
    app.add_middleware(AuditMiddleware, audit_trail=audit_trail)

    # Auth (innermost — closest to route handlers)
    if settings.API_KEYS:
        app.add_middleware(AuthMiddleware, api_keys=settings.API_KEYS)
    app.add_middleware(JWTAuthMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        rpm=settings.RATE_LIMIT_RPM,
        backend=settings.RATE_LIMIT_BACKEND,
        redis_url=settings.redis.URL,
    )

    # CORS — restricted origins, methods, and headers from settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Tenant-ID",
            "X-Tenant-Name",
            "X-Tenant-Tier",
        ],
    )

    # Request size limit
    app.add_middleware(RequestSizeMiddleware, max_bytes=settings.MAX_REQUEST_SIZE)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Logging (outermost)
    app.add_middleware(LoggingMiddleware)

    # --- Exception handlers ---
    app.add_exception_handler(DomainError, domain_error_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        PydanticValidationError,
        pydantic_validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready() -> dict[str, object]:
        """Readiness probe — verify dependencies are available."""
        checks: dict[str, bool] = {}
        container = getattr(app.state, "container", None)

        # Redis check
        if container:
            try:
                redis = container.redis_cache
                await redis.set("_health_check", "ok", ttl=10)
                val = await redis.get("_health_check")
                checks["redis"] = val == "ok"
            except Exception:
                checks["redis"] = False

        all_ready = all(checks.values()) if checks else True
        return {"ready": all_ready, "checks": checks}

    return app


# Module-level app for backwards compatibility with `uvicorn src.interfaces.api.main:app`
app = create_app()
