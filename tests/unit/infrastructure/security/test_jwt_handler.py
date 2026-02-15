"""Tests for JWT handler."""

from datetime import timedelta

import pytest
from jose import JWTError

from src.infrastructure.security.jwt_handler import JWTHandler


@pytest.fixture
def handler() -> JWTHandler:
    return JWTHandler(secret_key="test-secret", algorithm="HS256")


class TestCreateAccessToken:
    def test_creates_valid_token(self, handler: JWTHandler) -> None:
        token = handler.create_access_token({"sub": "user1"})
        claims = handler.decode_token(token)
        assert claims["sub"] == "user1"
        assert claims["type"] == "access"

    def test_custom_expiry(self, handler: JWTHandler) -> None:
        token = handler.create_access_token({"sub": "u"}, expires_delta=timedelta(minutes=1))
        claims = handler.decode_token(token)
        assert claims["sub"] == "u"


class TestCreateRefreshToken:
    def test_creates_refresh_token(self, handler: JWTHandler) -> None:
        token = handler.create_refresh_token({"sub": "user1"})
        claims = handler.decode_token(token)
        assert claims["type"] == "refresh"


class TestDecodeToken:
    def test_invalid_token_raises(self, handler: JWTHandler) -> None:
        with pytest.raises(JWTError):
            handler.decode_token("invalid.token.here")

    def test_wrong_secret_raises(self, handler: JWTHandler) -> None:
        token = handler.create_access_token({"sub": "u"})
        other = JWTHandler(secret_key="other-secret")
        with pytest.raises(JWTError):
            other.decode_token(token)
