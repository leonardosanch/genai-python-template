# src/domain/ports/pii_masking_port.py
"""Port for PII detection and masking."""

from abc import ABC, abstractmethod


class PIIMaskingPort(ABC):
    """Abstract interface for PII detection and masking.

    Used in logging, audit trails, and output sanitization.
    """

    @abstractmethod
    def mask(self, text: str) -> str:
        """Mask all detected PII in text. Returns sanitized text."""
        ...

    @abstractmethod
    def detect(self, text: str) -> list[str]:
        """Detect PII types present in text. Returns list of PII type names."""
        ...
