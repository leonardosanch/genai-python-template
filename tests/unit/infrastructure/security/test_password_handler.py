"""Tests for password handler."""

from src.infrastructure.security.password_handler import (
    hash_api_key,
    hash_password,
    verify_api_key,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify(self) -> None:
        hashed = hash_password("my-password")
        assert verify_password("my-password", hashed)

    def test_wrong_password_fails(self) -> None:
        hashed = hash_password("correct")
        assert not verify_password("wrong", hashed)


class TestApiKeyHashing:
    def test_hash_and_verify(self) -> None:
        hashed = hash_api_key("sk-abc123")
        assert verify_api_key("sk-abc123", hashed)

    def test_wrong_key_fails(self) -> None:
        hashed = hash_api_key("sk-abc123")
        assert not verify_api_key("sk-wrong", hashed)
