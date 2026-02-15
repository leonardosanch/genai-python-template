# tests/unit/test_pii_validators.py
"""Tests for extended PII validators."""

from src.application.guards.pii_validators import (
    api_key_validator,
    credit_card_validator,
    get_all_pii_validators,
    ip_address_validator,
    ssn_validator,
)


class TestCreditCardValidator:
    def test_valid_visa_redacted(self) -> None:
        text = "Card: 4111 1111 1111 1111"
        is_safe, sanitized, violation = credit_card_validator(text)
        assert not is_safe
        assert "<CREDIT_CARD>" in sanitized
        assert "Credit Card" in (violation or "")

    def test_valid_mastercard_with_dashes(self) -> None:
        text = "Pay with 5500-0000-0000-0004"
        is_safe, sanitized, _ = credit_card_validator(text)
        assert not is_safe
        assert "<CREDIT_CARD>" in sanitized

    def test_invalid_luhn_not_redacted(self) -> None:
        text = "Not a card: 1234567890123"
        is_safe, sanitized, violation = credit_card_validator(text)
        assert is_safe
        assert "<CREDIT_CARD>" not in sanitized
        assert violation is None

    def test_no_card_safe(self) -> None:
        text = "Just a normal sentence."
        is_safe, sanitized, violation = credit_card_validator(text)
        assert is_safe
        assert sanitized == text
        assert violation is None


class TestSSNValidator:
    def test_ssn_redacted(self) -> None:
        text = "SSN: 123-45-6789"
        is_safe, sanitized, violation = ssn_validator(text)
        assert not is_safe
        assert "<SSN>" in sanitized
        assert "SSN" in (violation or "")

    def test_no_ssn_safe(self) -> None:
        text = "Phone: 555-1234"
        is_safe, sanitized, violation = ssn_validator(text)
        assert is_safe
        assert violation is None


class TestAPIKeyValidator:
    def test_openai_key_redacted(self) -> None:
        text = "Key: sk-abcdefghij1234567890"
        is_safe, sanitized, violation = api_key_validator(text)
        assert not is_safe
        assert "<API_KEY>" in sanitized
        assert "OpenAI" in (violation or "")

    def test_aws_key_redacted(self) -> None:
        text = "AWS: AKIAIOSFODNN7EXAMPLE"
        is_safe, sanitized, violation = api_key_validator(text)
        assert not is_safe
        assert "<API_KEY>" in sanitized
        assert "AWS" in (violation or "")

    def test_github_pat_redacted(self) -> None:
        text = "Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        is_safe, sanitized, violation = api_key_validator(text)
        assert not is_safe
        assert "<API_KEY>" in sanitized

    def test_no_key_safe(self) -> None:
        text = "No secrets here."
        is_safe, sanitized, violation = api_key_validator(text)
        assert is_safe
        assert violation is None


class TestIPAddressValidator:
    def test_public_ip_redacted(self) -> None:
        text = "Server: 8.8.8.8"
        is_safe, sanitized, violation = ip_address_validator(text)
        assert not is_safe
        assert "<IP_ADDRESS>" in sanitized

    def test_private_ip_not_redacted(self) -> None:
        text = "Local: 192.168.1.1"
        is_safe, sanitized, violation = ip_address_validator(text)
        assert is_safe
        assert "192.168.1.1" in sanitized

    def test_localhost_not_redacted(self) -> None:
        text = "Localhost: 127.0.0.1"
        is_safe, sanitized, violation = ip_address_validator(text)
        assert is_safe
        assert "127.0.0.1" in sanitized

    def test_10_x_not_redacted(self) -> None:
        text = "Private: 10.0.0.1"
        is_safe, sanitized, violation = ip_address_validator(text)
        assert is_safe

    def test_invalid_octet_not_redacted(self) -> None:
        text = "Not IP: 999.999.999.999"
        is_safe, sanitized, violation = ip_address_validator(text)
        assert is_safe


class TestGetAllValidators:
    def test_returns_four_validators(self) -> None:
        validators = get_all_pii_validators()
        assert len(validators) == 4

    def test_all_callable(self) -> None:
        for v in get_all_pii_validators():
            assert callable(v)
