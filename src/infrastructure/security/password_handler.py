"""Password and API key hashing using bcrypt."""

import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return bcrypt.hashpw(key.encode(), bcrypt.gensalt()).decode()


def verify_api_key(key: str, hashed: str) -> bool:
    """Verify an API key against its stored hash."""
    return bcrypt.checkpw(key.encode(), hashed.encode())
