"""JWT token creation and validation using python-jose."""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from src.infrastructure.config import get_settings


class JWTHandler:
    """Handles JWT token lifecycle."""

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str | None = None,
    ) -> None:
        settings = get_settings().jwt
        self._secret_key = secret_key or settings.SECRET_KEY
        self._algorithm = algorithm or settings.ALGORITHM

    def create_access_token(
        self,
        data: dict[str, Any],
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a signed JWT access token."""
        settings = get_settings().jwt
        expire = datetime.now(UTC) + (
            expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode = {**data, "exp": expire, "type": "access"}
        return str(jwt.encode(to_encode, self._secret_key, algorithm=self._algorithm))

    def create_refresh_token(self, data: dict[str, Any]) -> str:
        """Create a signed JWT refresh token with longer TTL."""
        settings = get_settings().jwt
        expire = datetime.now(UTC) + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
        to_encode = {**data, "exp": expire, "type": "refresh"}
        return str(jwt.encode(to_encode, self._secret_key, algorithm=self._algorithm))

    def decode_token(self, token: str) -> dict[str, Any]:
        """Decode and validate a JWT token.

        Raises:
            JWTError: If token is invalid or expired.
        """
        payload: dict[str, Any] = jwt.decode(token, self._secret_key, algorithms=[self._algorithm])
        return payload


__all__ = ["JWTHandler", "JWTError"]
