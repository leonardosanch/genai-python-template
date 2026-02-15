"""Tests for GuardService — static analysis and LLM-based checks."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.domain.exceptions import ValidationError as DomainValidationError
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.security.guard import (
    GuardService,
    HallucinationCheckResult,
    SecurityCheckResult,
)


@pytest.fixture()
def mock_llm() -> LLMPort:
    """Create a mock LLMPort for testing."""
    return create_autospec(LLMPort, instance=True)


@pytest.fixture()
def guard(mock_llm: LLMPort) -> GuardService:
    """Create a GuardService with a mocked LLM."""
    return GuardService(llm=mock_llm)


class TestValidatePythonCode:
    """Tests for the static Python code validator (no LLM calls)."""

    def test_valid_code_no_errors(self, guard: GuardService) -> None:
        code = "x = 1\ny = x + 2\nprint(y)"
        errors = guard.validate_python_code(code)
        assert errors == []

    def test_syntax_error_detected(self, guard: GuardService) -> None:
        code = "def foo(\n"
        errors = guard.validate_python_code(code)
        assert len(errors) == 1
        assert "Syntax Error" in errors[0]

    def test_hardcoded_secret_detected(self, guard: GuardService) -> None:
        code = 'api_key = "sk-1234567890abcdef"'
        errors = guard.validate_python_code(code)
        assert any("secret" in e.lower() or "api_key" in e.lower() for e in errors)

    def test_short_value_not_flagged(self, guard: GuardService) -> None:
        code = 'password = "test"'
        errors = guard.validate_python_code(code)
        assert errors == []

    def test_non_string_value_not_flagged(self, guard: GuardService) -> None:
        code = "password = get_from_env()"
        errors = guard.validate_python_code(code)
        assert errors == []


class TestCheckSecurity:
    """Tests for the LLM-based security check."""

    @pytest.mark.asyncio()
    async def test_safe_content_passes(self, guard: GuardService, mock_llm: LLMPort) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=SecurityCheckResult(is_safe=True, issues=[], score=0.95),
        )
        result = await guard.check_security("print('hello')")
        assert result.is_safe is True
        assert result.score == 0.95

    @pytest.mark.asyncio()
    async def test_unsafe_content_detected(self, guard: GuardService, mock_llm: LLMPort) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=SecurityCheckResult(
                is_safe=False,
                issues=["SQL injection"],
                score=0.2,
            ),
        )
        result = await guard.check_security("SELECT * FROM users WHERE id = '{user_input}'")
        assert result.is_safe is False
        assert "SQL injection" in result.issues


class TestCheckHallucination:
    """Tests for the LLM-based hallucination check."""

    @pytest.mark.asyncio()
    async def test_grounded_response_passes(
        self,
        guard: GuardService,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=HallucinationCheckResult(
                is_grounded=True,
                deviations=[],
                citations=["source1"],
            ),
        )
        result = await guard.check_hallucination(
            response="The sky is blue.",
            context="The sky is blue during the day.",
        )
        assert result.is_grounded is True

    @pytest.mark.asyncio()
    async def test_hallucinated_response_detected(
        self,
        guard: GuardService,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=HallucinationCheckResult(
                is_grounded=False,
                deviations=["claim not in context"],
                citations=[],
            ),
        )
        result = await guard.check_hallucination(
            response="The sky is green.",
            context="The sky is blue.",
        )
        assert result.is_grounded is False
        assert len(result.deviations) == 1


class TestGuardOutput:
    """Tests for the full guard_output pipeline."""

    @pytest.mark.asyncio()
    async def test_safe_and_grounded_returns_true(
        self,
        guard: GuardService,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                SecurityCheckResult(is_safe=True, issues=[], score=0.9),
                HallucinationCheckResult(is_grounded=True, deviations=[], citations=[]),
            ],
        )
        result = await guard.guard_output(response="answer", context="context")
        assert result is True

    @pytest.mark.asyncio()
    async def test_unsafe_raises_validation_error(
        self,
        guard: GuardService,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=SecurityCheckResult(is_safe=False, issues=["XSS"], score=0.1),
        )
        with pytest.raises(DomainValidationError, match="Security Guardrail Failed"):
            await guard.guard_output(response="<script>alert('xss')</script>")

    @pytest.mark.asyncio()
    async def test_hallucinated_raises_validation_error(
        self,
        guard: GuardService,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                SecurityCheckResult(is_safe=True, issues=[], score=0.9),
                HallucinationCheckResult(
                    is_grounded=False,
                    deviations=["fabricated"],
                    citations=[],
                ),
            ],
        )
        with pytest.raises(DomainValidationError, match="Hallucination Guardrail Failed"):
            await guard.guard_output(response="invented fact", context="real context")

    @pytest.mark.asyncio()
    async def test_no_context_skips_hallucination_check(
        self,
        guard: GuardService,
        mock_llm: LLMPort,
    ) -> None:
        mock_llm.generate_structured = AsyncMock(
            return_value=SecurityCheckResult(is_safe=True, issues=[], score=0.95),
        )
        result = await guard.guard_output(response="safe answer")
        assert result is True
        # Only called once (security check), no hallucination check
        assert mock_llm.generate_structured.call_count == 1
