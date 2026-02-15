# src/domain/ports/context_assembler_port.py
"""Port for context assembly — building LLM prompts within a token budget."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Document
from src.domain.value_objects.context_budget import ContextBudget


class ContextAssemblerPort(ABC):
    """Abstract interface for assembling context within a token budget.

    Implementations decide how to select and truncate documents
    to fit within the budget.
    """

    @abstractmethod
    async def assemble(
        self,
        query: str,
        documents: list[Document],
        budget: ContextBudget,
        system_prompt: str = "",
    ) -> tuple[str, ContextBudget]:
        """Assemble context from documents within the token budget.

        Args:
            query: The user query.
            documents: Retrieved documents sorted by relevance.
            budget: Available token budget.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Tuple of (assembled context string, updated budget).
        """
        ...
