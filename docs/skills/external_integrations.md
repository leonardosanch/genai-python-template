---
name: External Integrations
description: OAuth2 flows, external API clients, token management, webhooks, and resilience patterns.
---

# Skill: External Integrations

## Description

This skill covers integrating with external APIs and OAuth2 providers in Python backend applications. Use this when connecting to LinkedIn, Google, Microsoft, Slack, or any third-party API that requires OAuth2 authentication, token management, webhook handling, or resilient HTTP communication.

## Executive Summary

**Critical external integration rules:**
- ALWAYS use `httpx.AsyncClient` with explicit timeouts for all external HTTP calls — never use `requests` (synchronous, blocks event loop)
- Store OAuth2 tokens ENCRYPTED in the database — access tokens and refresh tokens are high-value secrets
- Implement automatic token refresh with retry — expired tokens are the #1 cause of integration failures
- Circuit breaker on all external API calls — prevent cascade failures when a third-party is down
- NEVER log access tokens or API keys — mask in all logging and telemetry
- Abstract each integration behind a port — swapping providers or mocking for tests must be trivial

**Read full skill when:** Integrating with LinkedIn, Google, Microsoft, or any OAuth2 provider, consuming REST APIs, handling webhooks, implementing token management, or building resilient API clients.

---

## Versiones y Dependencias

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| httpx | >= 0.27.0 | ✅ Async HTTP client — reemplazo de requests |
| authlib | >= 1.3.0 | ✅ OAuth2 client, JWT, OpenID Connect |
| tenacity | >= 8.2.0 | ✅ Retry with exponential backoff, circuit breaker |
| cryptography | >= 42.0.0 | ✅ Fernet encryption para tokens en DB |
| pydantic | >= 2.0.0 | ✅ Validación de responses externas |

> ⚠️ **httpx vs aiohttp**: `httpx` tiene API similar a `requests`, soporte sync y async, HTTP/2. `aiohttp` es más maduro para WebSockets client. Para REST APIs, preferir `httpx`.

> ⚠️ **authlib vs python-social-auth**: `authlib` es más moderno y ligero. `python-social-auth` es más framework-opinionated (Django). Para FastAPI, preferir `authlib`.

---

## Deep Dive

## Core Concepts

1. **OAuth2 Authorization Code Flow**: El flow estándar para aplicaciones web. El usuario autoriza en el proveedor externo, la app recibe un `code`, lo intercambia por `access_token` + `refresh_token`. Usado para LinkedIn, Google, Microsoft SSO.

2. **Token Lifecycle**: Access tokens expiran (típicamente 1 hora). Refresh tokens duran más (días/meses). La app DEBE refresh automáticamente antes de la expiración. Almacenar tokens encriptados en DB.

3. **Circuit Breaker**: Patrón que previene llamadas repetidas a un servicio caído. Después de N fallos consecutivos, el circuito se "abre" y las llamadas fallan inmediatamente sin intentar la conexión. Se "cierra" automáticamente después de un cooldown.

4. **Webhook Handler**: Endpoint que recibe notificaciones push de APIs externas. DEBE ser idempotente (el mismo webhook puede llegar más de una vez), validar la firma/HMAC del webhook, y procesar en background.

5. **Rate Limiting del Cliente**: Respetar los límites de la API externa. Implementar retry con backoff cuando se recibe HTTP 429. Usar `asyncio.Semaphore` para limitar concurrencia.

---

## External Resources

### 🔐 OAuth2

- **OAuth2 RFC 6749**: [datatracker.ietf.org/doc/html/rfc6749](https://datatracker.ietf.org/doc/html/rfc6749)
    - *Best for*: Especificación oficial OAuth2 — authorization code, client credentials, refresh
- **Authlib Documentation**: [docs.authlib.org](https://docs.authlib.org/)
    - *Best for*: OAuth2 client, OIDC, JWT — para Python
- **LinkedIn OAuth2 Guide**: [learn.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow](https://learn.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow)
    - *Best for*: LinkedIn-specific OAuth2 implementation
- **Google OAuth2 for Web**: [developers.google.com/identity/protocols/oauth2/web-server](https://developers.google.com/identity/protocols/oauth2/web-server)
    - *Best for*: Google OAuth2 Authorization Code Flow
- **Microsoft Identity Platform**: [learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow)
    - *Best for*: Azure AD / Microsoft OAuth2

### 🌐 HTTP Clients & Resilience

- **httpx Documentation**: [www.python-httpx.org](https://www.python-httpx.org/)
    - *Best for*: Async HTTP client, timeouts, retries, HTTP/2
- **tenacity Documentation**: [tenacity.readthedocs.io](https://tenacity.readthedocs.io/)
    - *Best for*: Retry library with exponential backoff, circuit breaker
- **Circuit Breaker Pattern**: [martinfowler.com/bliki/CircuitBreaker.html](https://martinfowler.com/bliki/CircuitBreaker.html)
    - *Best for*: Understanding the circuit breaker pattern

### 📡 Webhooks

- **Webhook Security Best Practices**: [webhooks.fyi/security](https://webhooks.fyi/security)
    - *Best for*: HMAC verification, replay prevention, idempotency
- **Standard Webhooks**: [standardwebhooks.com](https://www.standardwebhooks.com/)
    - *Best for*: Emerging standard for webhook signatures

### 🔗 API-Specific SDKs

- **LinkedIn API Documentation**: [learn.microsoft.com/en-us/linkedin/](https://learn.microsoft.com/en-us/linkedin/)
    - *Best for*: LinkedIn Marketing API, Profile API, Share API
- **Google API Python Client**: [github.com/googleapis/google-api-python-client](https://github.com/googleapis/google-api-python-client)
    - *Best for*: Google APIs (Calendar, Drive, Gmail, etc.)
- **Microsoft Graph SDK**: [github.com/microsoftgraph/msgraph-sdk-python](https://github.com/microsoftgraph/msgraph-sdk-python)
    - *Best for*: Microsoft 365 APIs (Users, Calendar, Mail)

---

## Decision Trees

### Decision Tree 1: Qué flow OAuth2 usar

```
¿Qué tipo de acceso necesitas?
├── Acceso en nombre del usuario (perfil, datos personales)
│   └── Authorization Code Flow
│       ├── LinkedIn: perfil, conexiones, shares
│       ├── Google: Calendar, Gmail, Drive
│       └── Microsoft: Outlook, Teams, OneDrive
├── Acceso server-to-server (sin usuario)
│   └── Client Credentials Flow
│       ├── APIs administrativas
│       ├── Service accounts
│       └── Cron jobs, background tasks
├── Login / SSO del usuario
│   └── Authorization Code Flow + OpenID Connect (OIDC)
│       ├── Recibes id_token con datos del usuario
│       └── No necesitas access_token a la API externa
└── Mobile / SPA (no aplica a backend puro)
    └── Authorization Code Flow with PKCE
        └── Solo si tu backend es un BFF (Backend for Frontend)
```

### Decision Tree 2: Cómo manejar resiliencia

```
¿Cómo protegerse contra fallos de APIs externas?
├── La API devuelve error transitorio (500, 502, 503)
│   └── Retry con exponential backoff (tenacity)
│       ├── Max 3 intentos
│       ├── Backoff: 1s → 2s → 4s
│       └── Jitter aleatorio para evitar thundering herd
├── La API devuelve 429 (rate limited)
│   └── Respetar header Retry-After
│       ├── Si tiene Retry-After: esperar ese tiempo
│       └── Si no: backoff exponencial con base 5s
├── La API está completamente caída (timeout, connection refused)
│   └── Circuit Breaker
│       ├── Después de 5 fallos consecutivos → circuito abierto
│       ├── Circuito abierto: fail fast por 60 segundos
│       └── Después de 60s → half-open (intenta 1 request)
├── Token expirado (401)
│   └── Refresh token automáticamente → retry el request original
│       ├── Si refresh también falla → re-autenticar al usuario
│       └── Lock para evitar refresh concurrentes
└── Respuesta inesperada (schema cambió)
    └── Pydantic validation → log warning + graceful degradation
        └── Feature flag para desactivar la integración
```

---

## Instructions for the Agent

1.  **Port para cada integración**: Definir un port en domain layer por integración (ej: `LinkedInClientPort`, `GoogleCalendarPort`). Nunca importar `httpx`, `authlib`, o SDKs fuera de infrastructure.

2.  **Token storage**: Tokens en tabla dedicada (`oauth_tokens`) con columnas: `provider`, `user_id`, `access_token_encrypted`, `refresh_token_encrypted`, `expires_at`, `scopes`. Encriptar con Fernet (symmetric key desde env var).

3.  **Refresh automático**: Antes de cada API call, verificar `expires_at`. Si quedan < 5 minutos, refresh primero. Usar lock (Redis o DB) para evitar refresh concurrentes del mismo token.

4.  **Timeouts explícitos**: SIEMPRE `httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=30, write=10, pool=5))`. Nunca usar timeout infinito (default).

5.  **Retry con tenacity**: Decorar métodos de API con `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(...))`. Solo retry en errores transitorios (5xx, timeout, connection error). NUNCA retry en 4xx (excepto 429).

6.  **Webhook validation**: Verificar HMAC/signature en CADA webhook recibido. Rechazar con 401 si inválido. Procesar el payload en Celery task (responder 200 inmediatamente).

7.  **Pydantic para responses externas**: Definir Pydantic models para las respuestas de APIs externas. Usar `model_validate` con `strict=False` para tolerar campos extra. Log warning si la respuesta no matchea el schema esperado.

8.  **Feature flags**: Cada integración debe poder desactivarse vía env var (`LINKEDIN_ENABLED=false`). Si está desactivada, el port devuelve un valor por defecto o raise un error específico.

---

## Code Examples

### Example 1: Port en Domain Layer

```python
# src/domain/ports/linkedin_client_port.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LinkedInProfile:
    """Domain value object for LinkedIn profile data."""

    linkedin_id: str
    first_name: str
    last_name: str
    headline: str | None = None
    profile_url: str | None = None
    email: str | None = None
    profile_picture_url: str | None = None


class LinkedInClientPort(ABC):
    """Domain port for LinkedIn API integration."""

    @abstractmethod
    async def get_authorization_url(self, state: str) -> str:
        """Generate OAuth2 authorization URL for user consent."""

    @abstractmethod
    async def exchange_code(self, code: str) -> dict:
        """Exchange authorization code for tokens.

        Returns: {"access_token": ..., "refresh_token": ..., "expires_in": ...}
        """

    @abstractmethod
    async def get_profile(self, access_token: str) -> LinkedInProfile:
        """Fetch user's LinkedIn profile."""

    @abstractmethod
    async def search_candidates(
        self,
        access_token: str,
        keywords: str,
        limit: int = 10,
    ) -> list[LinkedInProfile]:
        """Search LinkedIn profiles by keywords."""
```

### Example 2: OAuth2 Token Storage (Encrypted)

```python
# src/infrastructure/database/models/oauth_token_model.py
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, UniqueConstraint

from src.infrastructure.database.models.base import Base


class OAuthTokenModel(Base):
    """Encrypted OAuth2 token storage."""

    __tablename__ = "oauth_tokens"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # "linkedin", "google", "microsoft"
    access_token_encrypted = Column(Text, nullable=False)
    refresh_token_encrypted = Column(Text, nullable=True)
    token_type = Column(String(20), default="Bearer")
    scopes = Column(String(500), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_user_provider"),
    )


# src/infrastructure/security/token_encryption.py
from cryptography.fernet import Fernet


class TokenEncryptor:
    """Encrypt/decrypt OAuth tokens for database storage.

    Key MUST come from environment variable, never hardcoded.
    Generate key: Fernet.generate_key().decode()
    """

    def __init__(self, encryption_key: str) -> None:
        self._fernet = Fernet(encryption_key.encode())

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode()).decode()
```

### Example 3: Resilient HTTP Client Base

```python
# src/infrastructure/http/resilient_client.py
import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = structlog.get_logger()

# Default timeouts for external APIs
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=5.0,    # 5s to establish connection
    read=30.0,      # 30s to read response
    write=10.0,     # 10s to send request body
    pool=5.0,       # 5s to acquire connection from pool
)

# Transient errors that should be retried
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.PoolTimeout,
)

RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 429}


class ExternalAPIError(Exception):
    """Raised when external API call fails after retries."""

    def __init__(self, provider: str, status_code: int | None, message: str) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {status_code}: {message}")


class ResilientHttpClient:
    """HTTP client with retries, timeouts, and structured logging.

    Use as base class for all external API adapters.
    """

    def __init__(
        self,
        base_url: str,
        provider_name: str,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url
        self._provider = provider_name
        self._timeout = timeout
        self._max_retries = max_retries

    async def _request(
        self,
        method: str,
        path: str,
        headers: dict | None = None,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an HTTP request with retries and structured logging."""

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        )
        async def _do_request() -> httpx.Response:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            ) as client:
                response = await client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    json=json,
                    params=params,
                )
                return response

        try:
            response = await _do_request()
        except RETRYABLE_EXCEPTIONS as exc:
            logger.error(
                "external_api_timeout",
                provider=self._provider,
                path=path,
                error=str(exc),
            )
            raise ExternalAPIError(
                self._provider, None, f"Connection failed after {self._max_retries} retries"
            ) from exc

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            logger.warning(
                "external_api_rate_limited",
                provider=self._provider,
                retry_after=retry_after,
            )
            raise ExternalAPIError(self._provider, 429, f"Rate limited. Retry after {retry_after}s")

        # Handle non-retryable errors
        if response.status_code >= 400:
            logger.error(
                "external_api_error",
                provider=self._provider,
                path=path,
                status_code=response.status_code,
                body=response.text[:500],  # Truncate for logging
            )
            raise ExternalAPIError(
                self._provider,
                response.status_code,
                response.text[:200],
            )

        logger.info(
            "external_api_success",
            provider=self._provider,
            path=path,
            status_code=response.status_code,
        )

        return response.json()
```

### Example 4: LinkedIn API Adapter

```python
# src/infrastructure/integrations/linkedin_client.py
import structlog
from pydantic import BaseModel, Field

from src.domain.ports.linkedin_client_port import LinkedInClientPort, LinkedInProfile
from src.infrastructure.http.resilient_client import ResilientHttpClient

logger = structlog.get_logger()

LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_API_URL = "https://api.linkedin.com/v2"


class LinkedInProfileResponse(BaseModel):
    """Pydantic model to validate LinkedIn API response."""

    id: str
    localizedFirstName: str = ""
    localizedLastName: str = ""
    headline: dict | None = Field(None, alias="localizedHeadline")
    vanityName: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class LinkedInClient(LinkedInClientPort):
    """LinkedIn API adapter with OAuth2 and resilient HTTP."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._http = ResilientHttpClient(
            base_url=LINKEDIN_API_URL,
            provider_name="linkedin",
        )

    async def get_authorization_url(self, state: str) -> str:
        """Generate LinkedIn OAuth2 authorization URL."""
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "state": state,
            "scope": "openid profile email",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{LINKEDIN_AUTH_URL}?{query}"

    async def exchange_code(self, code: str) -> dict:
        """Exchange authorization code for access + refresh tokens."""
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                LINKEDIN_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": self._redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

        logger.info(
            "linkedin_token_exchanged",
            expires_in=data.get("expires_in"),
            # NEVER log the actual token
        )

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_in": data.get("expires_in", 3600),
        }

    async def get_profile(self, access_token: str) -> LinkedInProfile:
        """Fetch authenticated user's LinkedIn profile."""
        data = await self._http._request(
            method="GET",
            path="/me",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"projection": "(id,localizedFirstName,localizedLastName,vanityName)"},
        )

        # Validate external response with Pydantic
        profile_data = LinkedInProfileResponse.model_validate(data)

        return LinkedInProfile(
            linkedin_id=profile_data.id,
            first_name=profile_data.localizedFirstName,
            last_name=profile_data.localizedLastName,
            headline=None,
            profile_url=f"https://linkedin.com/in/{profile_data.vanityName}"
            if profile_data.vanityName
            else None,
        )

    async def search_candidates(
        self,
        access_token: str,
        keywords: str,
        limit: int = 10,
    ) -> list[LinkedInProfile]:
        """Search LinkedIn profiles (requires LinkedIn Recruiter or Talent Solutions API)."""
        # Note: LinkedIn's search APIs require specific product approval
        logger.warning("linkedin_search_not_implemented", keywords=keywords)
        return []
```

### Example 5: OAuth2 Callback Endpoint

```python
# src/interfaces/api/routes/oauth_routes.py
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response

router = APIRouter(prefix="/api/v1/oauth", tags=["oauth"])


@router.get("/{provider}/authorize")
async def oauth_authorize(
    provider: str,
    request: Request,
    current_user=Depends(get_current_user),
):
    """Start OAuth2 flow — redirect user to provider's consent page."""
    client = get_oauth_client(provider)  # Factory: returns LinkedInClient, GoogleClient, etc.

    # Generate CSRF state token, store in session/Redis
    state = secrets.token_urlsafe(32)
    await store_oauth_state(
        user_id=str(current_user.id),
        state=state,
        provider=provider,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )

    auth_url = await client.get_authorization_url(state=state)
    return {"authorization_url": auth_url}


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    current_user=Depends(get_current_user),
    token_repo=Depends(get_token_repository),
    encryptor=Depends(get_token_encryptor),
):
    """OAuth2 callback — exchange code for tokens and store them."""
    # 1. Validate CSRF state
    stored_state = await get_oauth_state(state)
    if not stored_state or stored_state["user_id"] != str(current_user.id):
        raise HTTPException(400, "Invalid or expired OAuth state")

    # 2. Exchange code for tokens
    client = get_oauth_client(provider)
    try:
        tokens = await client.exchange_code(code)
    except Exception as exc:
        raise HTTPException(502, f"Failed to exchange code with {provider}")

    # 3. Store tokens encrypted
    await token_repo.upsert(
        user_id=str(current_user.id),
        provider=provider,
        access_token=encryptor.encrypt(tokens["access_token"]),
        refresh_token=encryptor.encrypt(tokens["refresh_token"]) if tokens.get("refresh_token") else None,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=tokens["expires_in"]),
    )

    # 4. Cleanup state
    await delete_oauth_state(state)

    return {"status": "connected", "provider": provider}


@router.delete("/{provider}/disconnect")
async def oauth_disconnect(
    provider: str,
    current_user=Depends(get_current_user),
    token_repo=Depends(get_token_repository),
):
    """Disconnect OAuth2 integration — delete stored tokens."""
    await token_repo.delete(user_id=str(current_user.id), provider=provider)
    return {"status": "disconnected", "provider": provider}
```

### Example 6: Automatic Token Refresh Decorator

```python
# src/infrastructure/integrations/token_manager.py
import asyncio
import functools
from datetime import datetime, timedelta, timezone

import structlog

from src.infrastructure.security.token_encryption import TokenEncryptor

logger = structlog.get_logger()

# Lock to prevent concurrent token refreshes
_refresh_locks: dict[str, asyncio.Lock] = {}


def auto_refresh_token(func):
    """Decorator: automatically refresh expired OAuth tokens before API calls.

    The decorated method must be on a class with:
    - self._token_repo: repository for OAuth tokens
    - self._encryptor: TokenEncryptor
    - self._provider: provider name
    - self.exchange_refresh_token(refresh_token) -> dict
    """

    @functools.wraps(func)
    async def wrapper(self, user_id: str, *args, **kwargs):
        token_record = await self._token_repo.get(user_id=user_id, provider=self._provider)
        if not token_record:
            raise ValueError(f"No {self._provider} token found for user {user_id}")

        # Check if token needs refresh (< 5 min remaining)
        if token_record.expires_at < datetime.now(timezone.utc) + timedelta(minutes=5):
            lock_key = f"{user_id}:{self._provider}"
            if lock_key not in _refresh_locks:
                _refresh_locks[lock_key] = asyncio.Lock()

            async with _refresh_locks[lock_key]:
                # Re-check after acquiring lock (another coroutine may have refreshed)
                token_record = await self._token_repo.get(user_id=user_id, provider=self._provider)
                if token_record.expires_at < datetime.now(timezone.utc) + timedelta(minutes=5):
                    refresh_token = self._encryptor.decrypt(token_record.refresh_token_encrypted)

                    try:
                        new_tokens = await self._exchange_refresh_token(refresh_token)
                    except Exception as exc:
                        logger.error(
                            "token_refresh_failed",
                            provider=self._provider,
                            user_id=user_id,
                            error=str(exc),
                        )
                        raise

                    await self._token_repo.upsert(
                        user_id=user_id,
                        provider=self._provider,
                        access_token=self._encryptor.encrypt(new_tokens["access_token"]),
                        refresh_token=self._encryptor.encrypt(new_tokens.get("refresh_token", refresh_token)),
                        expires_at=datetime.now(timezone.utc) + timedelta(seconds=new_tokens["expires_in"]),
                    )

                    logger.info("token_refreshed", provider=self._provider, user_id=user_id)

                    token_record = await self._token_repo.get(user_id=user_id, provider=self._provider)

        # Decrypt and pass token to the actual method
        access_token = self._encryptor.decrypt(token_record.access_token_encrypted)
        return await func(self, access_token=access_token, *args, **kwargs)

    return wrapper
```

### Example 7: Webhook Handler

```python
# src/interfaces/api/routes/webhook_routes.py
import hashlib
import hmac

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str,
    algorithm: str = "sha256",
) -> bool:
    """Verify HMAC signature of webhook payload."""
    expected = hmac.new(
        secret.encode(),
        payload,
        getattr(hashlib, algorithm),
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/{provider}")
async def receive_webhook(
    provider: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Receive webhook from external provider.

    - Validates HMAC signature
    - Responds 200 immediately
    - Processes payload in background
    """
    body = await request.body()

    # 1. Validate signature (provider-specific header)
    signature_header = {
        "linkedin": "X-LinkedIn-Signature",
        "stripe": "Stripe-Signature",
        "github": "X-Hub-Signature-256",
    }.get(provider, "X-Webhook-Signature")

    signature = request.headers.get(signature_header)
    if not signature:
        raise HTTPException(401, "Missing webhook signature")

    webhook_secret = get_webhook_secret(provider)  # From settings
    if not verify_webhook_signature(body, signature, webhook_secret):
        logger.warning("webhook_invalid_signature", provider=provider)
        raise HTTPException(401, "Invalid webhook signature")

    # 2. Parse payload
    payload = await request.json()
    event_type = payload.get("event", payload.get("type", "unknown"))

    # 3. Idempotency check (deduplicate by event ID)
    event_id = payload.get("id", payload.get("event_id"))
    if event_id and await is_webhook_processed(event_id):
        logger.info("webhook_duplicate", provider=provider, event_id=event_id)
        return {"status": "already_processed"}

    # 4. Process in background (respond 200 immediately)
    background_tasks.add_task(
        process_webhook_event,
        provider=provider,
        event_type=event_type,
        payload=payload,
        event_id=event_id,
    )

    logger.info("webhook_received", provider=provider, event_type=event_type, event_id=event_id)
    return {"status": "accepted"}
```

### Example 8: Settings e Integración Config

```python
# src/infrastructure/config/integration_settings.py
from pydantic_settings import BaseSettings


class LinkedInSettings(BaseSettings):
    """LinkedIn OAuth2 configuration."""

    LINKEDIN_ENABLED: bool = False
    LINKEDIN_CLIENT_ID: str = ""
    LINKEDIN_CLIENT_SECRET: str = ""
    LINKEDIN_REDIRECT_URI: str = ""
    LINKEDIN_WEBHOOK_SECRET: str = ""

    model_config = {"env_prefix": "", "case_sensitive": True}


class IntegrationSettings(BaseSettings):
    """Master settings for all external integrations."""

    OAUTH_TOKEN_ENCRYPTION_KEY: str  # Fernet key for encrypting tokens in DB

    # Feature flags
    LINKEDIN_ENABLED: bool = False
    GOOGLE_ENABLED: bool = False
    MICROSOFT_ENABLED: bool = False

    model_config = {"env_prefix": "", "case_sensitive": True}
```

---

## Anti-Patterns to Avoid

### ❌ Using `requests` (Synchronous) for External APIs
**Problem**: Blocks the event loop, degrades FastAPI performance
**Example**:
```python
# BAD: Synchronous — blocks entire server
import requests
response = requests.get("https://api.linkedin.com/v2/me")
```
**Solution**: Always `httpx.AsyncClient`
```python
# GOOD: Non-blocking async client
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.linkedin.com/v2/me")
```

### ❌ Storing Tokens in Plain Text
**Problem**: Database breach exposes all OAuth tokens
**Solution**: Encrypt with Fernet before storing, decrypt only when needed
```python
# GOOD: Encrypted storage
encrypted = fernet.encrypt(access_token.encode())
# Store `encrypted` in database, not `access_token`
```

### ❌ No Timeout on External Calls
**Problem**: If LinkedIn is slow, your server hangs indefinitely
**Solution**: Explicit timeouts on every external call
```python
# GOOD: Explicit timeout — never waiting forever
async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=30)) as client:
    ...
```

### ❌ Logging Access Tokens
**Problem**: Tokens in logs = anyone with log access has API access
**Example**:
```python
# BAD: Token in logs
logger.info(f"LinkedIn response for token {access_token}: {data}")
```
**Solution**: Never log tokens, mask sensitive data
```python
# GOOD: No token in logs
logger.info("linkedin_profile_fetched", user_id=user_id, status_code=200)
```

### ❌ No Circuit Breaker
**Problem**: If LinkedIn is down, every user request waits 30s and fails — cascade failure
**Solution**: Circuit breaker fails fast after N consecutive failures

### ❌ Not Validating External Responses with Pydantic
**Problem**: External APIs change schemas without notice — your code crashes on `KeyError`
**Solution**: Pydantic model with `extra="ignore"` — tolerates new fields, catches removed fields

---

## External Integration Checklist

### OAuth2
- [ ] Authorization Code Flow implemented with CSRF state parameter
- [ ] State token stored in Redis/DB with short TTL (10 min)
- [ ] Code exchange happens server-side (never client-side)
- [ ] Tokens stored encrypted (Fernet) in dedicated table
- [ ] Automatic token refresh before expiration (< 5 min)
- [ ] Lock to prevent concurrent refreshes of same token
- [ ] Disconnect endpoint to revoke and delete tokens

### HTTP Client
- [ ] `httpx.AsyncClient` with explicit timeouts (connect, read, write)
- [ ] Retry with exponential backoff (tenacity) for 5xx errors
- [ ] HTTP 429 handling (respect Retry-After header)
- [ ] Circuit breaker for complete outages
- [ ] Structured logging for all requests (no tokens logged)

### Webhooks
- [ ] HMAC signature validation on every webhook
- [ ] Idempotency check (duplicate event detection)
- [ ] Immediate 200 response — process in background
- [ ] Dead letter queue for failed webhook processing
- [ ] Webhook secret stored in environment variable

### Architecture
- [ ] Port defined in domain layer per integration
- [ ] Adapter in infrastructure implements the port
- [ ] Pydantic models for external API responses (`extra="ignore"`)
- [ ] Feature flags to enable/disable each integration
- [ ] Integration settings via `pydantic-settings`

### Security
- [ ] CSRF protection on OAuth callback (state parameter)
- [ ] Token encryption key in env var (never in code)
- [ ] Scopes request least-privilege (only what's needed)
- [ ] Token revocation on user account deletion
- [ ] Audit log for OAuth connect/disconnect events

---

## Additional References

- [OAuth2 RFC 6749](https://datatracker.ietf.org/doc/html/rfc6749)
- [LinkedIn OAuth2 Documentation](https://learn.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow)
- [httpx Documentation](https://www.python-httpx.org/)
- [tenacity Documentation](https://tenacity.readthedocs.io/)
- [Martin Fowler — Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Authlib OAuth Client](https://docs.authlib.org/en/latest/client/index.html)
- [Webhook Security Best Practices](https://webhooks.fyi/security)
