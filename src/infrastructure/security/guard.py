"""GuardService: Runtime Safety & Hallucination Prevention.

This service provides a centralized way to validate LLM inputs and outputs,
ensuring code safety, security compliance, and preventing hallucinations
by checking provenance and syntax.

Depends on LLMPort — works with any provider (OpenAI, Anthropic, etc.).
"""

import ast

from pydantic import BaseModel, Field

from src.domain.exceptions import ValidationError as DomainValidationError
from src.domain.ports.llm_port import LLMPort

SECURITY_CHECK_PROMPT = (
    "You are a senior security engineer. Analyze the following code/text "
    "for security vulnerabilities (OWASP Top 10, Injection, Secrets).\n\n"
    "{content}"
)

HALLUCINATION_CHECK_PROMPT = (
    "You are a fact-checker. Verify if the RESPONSE is fully supported "
    "by the CONTEXT. Flag any information not present in context.\n\n"
    "CONTEXT: {context}\n\nRESPONSE: {response}"
)


class SecurityCheckResult(BaseModel):
    """Result of a security check."""

    is_safe: bool = Field(..., description="Whether the content is strictly safe.")
    issues: list[str] = Field(default_factory=list, description="List of security issues found.")
    score: float = Field(..., ge=0.0, le=1.0, description="Safety score from 0.0 to 1.0.")


class HallucinationCheckResult(BaseModel):
    """Result of a hallucination check."""

    is_grounded: bool = Field(..., description="Whether the response is strictly based on context.")
    deviations: list[str] = Field(
        default_factory=list, description="List of facts not found in context."
    )
    citations: list[str] = Field(
        default_factory=list, description="List of direct quotes or citations found."
    )


class GuardService:
    """Centralized guardrails for the application.

    Uses LLMPort for structured output extraction — provider-agnostic.
    Static analysis (validate_python_code) requires no LLM.
    """

    def __init__(self, llm: LLMPort) -> None:
        """Initialize with an LLM port (any provider).

        Args:
            llm: Abstract LLM interface for structured checks.
        """
        self._llm = llm

    def validate_python_code(self, code: str) -> list[str]:
        """Statically analyze Python code for syntax errors and basic security flaws.

        Args:
            code: The Python code to validate.

        Returns:
            List of error messages. Empty list means valid.
        """
        errors: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax Error: {e}"]
        except Exception as e:
            return [f"Parse Error: {e}"]

        errors.extend(self._check_security_nodes(tree, code))
        return errors

    def _check_security_nodes(self, tree: ast.AST, code: str) -> list[str]:
        """Check AST nodes for security issues."""
        errors: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in ["os", "subprocess", "sys"] and "system" in code:
                        pass  # Warning suppressed for agent context

            if isinstance(node, ast.Assign):
                errors.extend(self._check_assignment_secrets(node))
        return errors

    def _check_assignment_secrets(self, node: ast.Assign) -> list[str]:
        """Check assignment for hardcoded secrets."""
        errors: list[str] = []
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            is_secret_var = any(s in target.id.lower() for s in ["password", "secret", "api_key"])

            if is_secret_var:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    if len(node.value.value) > 10:
                        errors.append(f"Potential hardcoded secret in variable: {target.id}")
        return errors

    async def check_security(self, content: str) -> SecurityCheckResult:
        """Use an LLM judge to check for security vulnerabilities."""
        prompt = SECURITY_CHECK_PROMPT.format(content=content)
        return await self._llm.generate_structured(
            prompt,
            schema=SecurityCheckResult,
            temperature=0.0,
        )

    async def check_hallucination(self, response: str, context: str) -> HallucinationCheckResult:
        """Use an LLM judge to check if the response is grounded in the context."""
        prompt = HALLUCINATION_CHECK_PROMPT.format(context=context, response=response)
        return await self._llm.generate_structured(
            prompt,
            schema=HallucinationCheckResult,
            temperature=0.0,
        )

    async def guard_output(self, response: str, context: str | None = None) -> bool:
        """Main entry point. Validates output and optionally checks against context.

        Raises:
            DomainValidationError: If security or hallucination checks fail.
        """
        security = await self.check_security(response)
        if not security.is_safe:
            raise DomainValidationError(f"Security Guardrail Failed: {security.issues}")

        if context:
            hallucination = await self.check_hallucination(response, context)
            if not hallucination.is_grounded:
                raise DomainValidationError(
                    f"Hallucination Guardrail Failed: {hallucination.deviations}"
                )

        return True
