# src/infrastructure/governance/regex_pii_masker.py
"""Regex-based PII masker reusing patterns from pii_validators."""

import re

from src.domain.ports.pii_masking_port import PIIMaskingPort

# PII patterns: (pattern, replacement, pii_type_name)
_PII_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "<EMAIL>", "email"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "<PHONE>", "phone"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "<SSN>", "ssn"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "<API_KEY>", "api_key"),
    (r"\bAKIA[A-Z0-9]{16}\b", "<API_KEY>", "api_key"),
    (r"\bghp_[A-Za-z0-9]{36,}\b", "<API_KEY>", "api_key"),
]


class RegexPIIMasker(PIIMaskingPort):
    """PII masker using regex patterns.

    Detects and masks emails, phones, SSNs, and API keys.
    """

    def mask(self, text: str) -> str:
        for pattern, replacement, _ in _PII_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text

    def detect(self, text: str) -> list[str]:
        detected: list[str] = []
        for pattern, _, pii_type in _PII_PATTERNS:
            if re.search(pattern, text):
                if pii_type not in detected:
                    detected.append(pii_type)
        return detected
