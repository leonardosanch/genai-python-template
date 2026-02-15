"""Authentication routes — JWT token issuance and refresh."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.domain.value_objects.role import Role
from src.infrastructure.config import get_settings
from src.infrastructure.security.jwt_handler import JWTHandler
from src.infrastructure.security.password_handler import verify_api_key

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_jwt = JWTHandler()


class TokenRequest(BaseModel):
    """Login request body."""

    client_id: str
    client_secret: str
    roles: list[Role] = [Role.VIEWER]


class TokenResponse(BaseModel):
    """Token pair response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """Refresh token request body."""

    refresh_token: str


@router.post("/token", response_model=TokenResponse)
async def create_token(body: TokenRequest) -> TokenResponse:
    """Issue access + refresh tokens.

    For this template, ``client_secret`` is validated against ``API_KEYS``
    in settings. Replace with a real user store in production.
    """
    settings = get_settings()
    if not settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication not configured (API_KEYS is empty)",
        )

    # Check plain-text match (existing behaviour) or hashed match
    valid = body.client_secret in settings.API_KEYS or any(
        verify_api_key(body.client_secret, k)
        for k in settings.API_KEYS
        if k.startswith("$2b$")  # bcrypt prefix
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    data: dict[str, object] = {
        "sub": body.client_id,
        "roles": [r.value for r in body.roles],
    }
    return TokenResponse(
        access_token=_jwt.create_access_token(data),
        refresh_token=_jwt.create_refresh_token(data),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest) -> TokenResponse:
    """Rotate a refresh token into a new access + refresh pair."""
    from jose import JWTError

    try:
        claims = _jwt.decode_token(body.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if claims.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token",
        )

    data: dict[str, object] = {
        "sub": claims["sub"],
        "roles": claims.get("roles", [Role.VIEWER.value]),
    }
    return TokenResponse(
        access_token=_jwt.create_access_token(data),
        refresh_token=_jwt.create_refresh_token(data),
    )
