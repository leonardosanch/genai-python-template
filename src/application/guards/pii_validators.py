# src/application/guards/pii_validators.py
"""Extended PII validators for output guard chain.

Each validator follows the ValidatorFunc signature:
    (text: str) -> tuple[bool, str, str | None]
"""

import re

from src.application.guards.output_validator import ValidatorFunc


def _luhn_check(card_number: str) -> bool:
    """Validate a card number using the Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def credit_card_validator(text: str) -> tuple[bool, str, str | None]:
    """Detect and redact credit card numbers (13-19 digits, Luhn-valid)."""
    pattern = r"\b(?:\d[ -]*?){13,19}\b"
    is_safe = True
    violation = None

    def _replace(match: re.Match[str]) -> str:
        nonlocal is_safe, violation
        raw = match.group(0)
        digits_only = re.sub(r"[ -]", "", raw)
        if digits_only.isdigit() and _luhn_check(digits_only):
            is_safe = False
            violation = "PII detected: Credit Card"
            return "<CREDIT_CARD>"
        return raw

    text = re.sub(pattern, _replace, text)
    return is_safe, text, violation


def ssn_validator(text: str) -> tuple[bool, str, str | None]:
    """Detect and redact US Social Security Numbers (XXX-XX-XXXX)."""
    pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(pattern, text):
        text = re.sub(pattern, "<SSN>", text)
        return False, text, "PII detected: SSN"
    return True, text, None


def api_key_validator(text: str) -> tuple[bool, str, str | None]:
    """Detect and redact common API key patterns."""
    patterns = [
        (r"\bsk-[A-Za-z0-9]{20,}\b", "OpenAI"),
        (r"\bAKIA[A-Z0-9]{16}\b", "AWS"),
        (r"\bghp_[A-Za-z0-9]{36,}\b", "GitHub"),
        (r"\bgho_[A-Za-z0-9]{36,}\b", "GitHub OAuth"),
        (r"\bglpat-[A-Za-z0-9\-]{20,}\b", "GitLab"),
        (r"\bxoxb-[A-Za-z0-9\-]{20,}\b", "Slack Bot"),
        (r"\bxoxp-[A-Za-z0-9\-]{20,}\b", "Slack User"),
    ]
    is_safe = True
    violation = None

    for pat, provider in patterns:
        if re.search(pat, text):
            text = re.sub(pat, "<API_KEY>", text)
            is_safe = False
            if not violation:
                violation = f"API key detected: {provider}"
            else:
                violation = f"{violation}, {provider}"

    return is_safe, text, violation


def ip_address_validator(text: str) -> tuple[bool, str, str | None]:
    """Detect and redact public IPv4 addresses (excludes private/localhost)."""
    ipv4_pattern = r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"
    is_safe = True
    violation = None

    def _replace(match: re.Match[str]) -> str:
        nonlocal is_safe, violation
        octets = [int(match.group(i)) for i in range(1, 5)]
        # Validate range
        if any(o > 255 for o in octets):
            return match.group(0)
        # Skip private/localhost
        if (
            octets[0] == 10
            or (octets[0] == 172 and 16 <= octets[1] <= 31)
            or (octets[0] == 192 and octets[1] == 168)
            or (octets[0] == 127)
            or (octets[0] == 0)
        ):
            return match.group(0)
        is_safe = False
        violation = "PII detected: Public IP Address"
        return "<IP_ADDRESS>"

    text = re.sub(ipv4_pattern, _replace, text)
    return is_safe, text, violation


def get_all_pii_validators() -> list[ValidatorFunc]:
    """Return all PII validators for convenient registration."""
    return [
        credit_card_validator,
        ssn_validator,
        api_key_validator,
        ip_address_validator,
    ]
