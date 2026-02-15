# tests/unit/test_regex_pii_masker.py
"""Tests for regex PII masker."""

from src.infrastructure.governance.regex_pii_masker import RegexPIIMasker


class TestRegexPIIMasker:
    def setup_method(self) -> None:
        self.masker = RegexPIIMasker()

    def test_mask_email(self) -> None:
        result = self.masker.mask("Contact john@example.com for info")
        assert "<EMAIL>" in result
        assert "john@example.com" not in result

    def test_mask_phone(self) -> None:
        result = self.masker.mask("Call 555-123-4567 now")
        assert "<PHONE>" in result
        assert "555-123-4567" not in result

    def test_mask_ssn(self) -> None:
        result = self.masker.mask("SSN: 123-45-6789")
        assert "<SSN>" in result

    def test_mask_api_key_openai(self) -> None:
        result = self.masker.mask("Key: sk-abcdefghij1234567890")
        assert "<API_KEY>" in result

    def test_mask_api_key_aws(self) -> None:
        result = self.masker.mask("AWS: AKIAIOSFODNN7EXAMPLE")
        assert "<API_KEY>" in result

    def test_mask_no_pii(self) -> None:
        text = "Just a normal sentence."
        result = self.masker.mask(text)
        assert result == text

    def test_mask_multiple_pii(self) -> None:
        text = "Email: test@example.com, SSN: 123-45-6789"
        result = self.masker.mask(text)
        assert "<EMAIL>" in result
        assert "<SSN>" in result

    def test_detect_email(self) -> None:
        detected = self.masker.detect("Email: test@example.com")
        assert "email" in detected

    def test_detect_multiple(self) -> None:
        detected = self.masker.detect("test@example.com 123-45-6789")
        assert "email" in detected
        assert "ssn" in detected

    def test_detect_none(self) -> None:
        detected = self.masker.detect("Clean text")
        assert detected == []
