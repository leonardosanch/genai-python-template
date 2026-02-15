# src/infrastructure/config/__init__.py
"""Configuration module."""

from .settings import Settings, get_settings

__all__ = ["get_settings", "Settings"]
