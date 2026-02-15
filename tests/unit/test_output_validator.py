# tests/unit/test_output_validator.py
from src.application.guards.output_validator import (
    OutputGuard,
    max_length_validator,
    no_pii_validator,
    no_system_prompt_leak_validator,
)


def test_guard_safe_text():
    """Test with safe text."""
    guard = OutputGuard()
    guard.add_validator(no_pii_validator)

    text = "Hello, I am a user."
    result = guard.validate(text)

    assert result.is_safe is True
    assert result.sanitized_text == text
    assert len(result.violations) == 0


def test_guard_pii_redaction():
    """Test PII redaction."""
    guard = OutputGuard()
    guard.add_validator(no_pii_validator)

    text = "Contact me at test@example.com or 555-123-4567."
    result = guard.validate(text)

    assert result.is_safe is False
    assert "<EMAIL_REDACTED>" in result.sanitized_text
    assert "<PHONE_REDACTED>" in result.sanitized_text
    assert "PII detected" in result.violations[0]


def test_guard_prompt_leak():
    """Test system prompt leak detection."""
    guard = OutputGuard()
    guard.add_validator(no_system_prompt_leak_validator)

    text = "Sure! As an AI language model, I can preserve context."
    result = guard.validate(text)

    assert result.is_safe is False
    assert "System prompt leak detected" in result.violations[0]


def test_guard_max_length():
    """Test max length truncation."""
    guard = OutputGuard()
    guard.add_validator(max_length_validator(10))

    text = "This is way too long."
    result = guard.validate(text)

    assert result.is_safe is False
    assert result.sanitized_text == "This is wa...(truncated)"
    assert "Output exceeded" in result.violations[0]


def test_guard_chaining():
    """Test multiple validators."""
    guard = OutputGuard()
    guard.add_validator(no_pii_validator)
    guard.add_validator(max_length_validator(50))

    text = "My email is test@example.com and this is a long sentence."
    result = guard.validate(text)

    assert result.is_safe is False
    assert "<EMAIL_REDACTED>" in result.sanitized_text
    # Check if length validation ran on the already redacted text
    # Original redacted: "My email is <EMAIL_REDACTED> and this is a long sentence." (54 chars)
    # Truncated to 50 chars + "..." (approx 63-64 chars depending on impl)
    assert len(result.sanitized_text) <= 65
    assert result.sanitized_text.endswith("(truncated)")
