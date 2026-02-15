# src/application/guards/output_validator.py
import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class GuardResult:
    """Result of a guard validation."""

    is_safe: bool
    sanitized_text: str
    violations: list[str] = field(default_factory=list)


ValidatorFunc = Callable[[str], tuple[bool, str, str | None]]
# Returns (is_safe, sanitized_text, violation_msg_or_none)


class OutputGuard:
    """Guard for identifying and sanitizing unsafe LLM outputs."""

    def __init__(self) -> None:
        self.validators: list[ValidatorFunc] = []

    def add_validator(self, validator: ValidatorFunc) -> None:
        """Add a validator function to the chain."""
        self.validators.append(validator)

    def validate(self, text: str) -> GuardResult:
        """Run all validators on the text."""
        current_text = text
        all_violations = []
        is_safe_global = True

        for validator in self.validators:
            is_safe, sanitized, violation = validator(current_text)
            current_text = sanitized
            if not is_safe:
                is_safe_global = False
                if violation:
                    all_violations.append(violation)

        return GuardResult(
            is_safe=is_safe_global, sanitized_text=current_text, violations=all_violations
        )


# --- Validators ---


def no_pii_validator(text: str) -> tuple[bool, str, str | None]:
    """Detects and redacts emails and phone numbers."""
    # Simple regex for demonstration
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    phone_regex = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"

    violation = None
    is_safe = True

    if re.search(email_regex, text):
        text = re.sub(email_regex, "<EMAIL_REDACTED>", text)
        is_safe = False
        violation = "PII detected: Email"

    if re.search(phone_regex, text):
        text = re.sub(phone_regex, "<PHONE_REDACTED>", text)
        is_safe = False
        violation = "PII detected: Phone Number" if not violation else f"{violation} & Phone Number"

    return is_safe, text, violation


def no_system_prompt_leak_validator(text: str) -> tuple[bool, str, str | None]:
    """Detects if the model leaks its identity or instructions."""
    leak_phrases = [
        "Ignore previous instructions",
        "You are a helpful assistant",
        "system prompt",
        "You are an AI",
        "As an AI language model",
    ]

    lower_text = text.lower()
    for phrase in leak_phrases:
        if phrase.lower() in lower_text:
            return False, text, f"System prompt leak detected: '{phrase}'"

    return True, text, None


def max_length_validator(max_chars: int) -> ValidatorFunc:
    """Factory for max length validator."""

    def validator(text: str) -> tuple[bool, str, str | None]:
        if len(text) > max_chars:
            return False, text[:max_chars] + "...(truncated)", f"Output exceeded {max_chars} chars"
        return True, text, None

    return validator
