# src/application/services/context_assembler.py
"""Context assembler — builds LLM context within a token budget."""

import structlog

from src.domain.entities.document import Document
from src.domain.ports.context_assembler_port import ContextAssemblerPort
from src.domain.value_objects.context_budget import ContextBudget

logger = structlog.get_logger(__name__)

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


class ContextAssembler(ContextAssemblerPort):
    """Assembles context by selecting documents by score until budget exhausted.

    Strategy:
    1. Reserve tokens for system prompt and query
    2. Add documents in score order until budget is full
    3. Truncate last document if needed to stay within budget
    """

    async def assemble(
        self,
        query: str,
        documents: list[Document],
        budget: ContextBudget,
        system_prompt: str = "",
    ) -> tuple[str, ContextBudget]:
        parts: list[str] = []
        current_budget = budget

        # Reserve space for system prompt
        if system_prompt:
            sys_tokens = _estimate_tokens(system_prompt)
            current_budget = current_budget.consume(sys_tokens)
            parts.append(system_prompt)

        # Reserve space for query
        query_section = f"\nQuestion: {query}\n"
        query_tokens = _estimate_tokens(query_section)
        current_budget = current_budget.consume(query_tokens)

        # Add documents by relevance
        parts.append("\nContext:")
        docs_added = 0
        for doc in documents:
            doc_text = f"\n---\n{doc.content}"
            doc_tokens = _estimate_tokens(doc_text)

            if doc_tokens <= current_budget.remaining:
                parts.append(doc_text)
                current_budget = current_budget.consume(doc_tokens)
                docs_added += 1
            elif current_budget.remaining > 10:
                # Truncate to fit remaining budget
                max_chars = current_budget.remaining * CHARS_PER_TOKEN
                truncated = doc_text[:max_chars]
                trunc_tokens = _estimate_tokens(truncated)
                parts.append(truncated)
                current_budget = current_budget.consume(trunc_tokens)
                docs_added += 1
                break
            else:
                break

        parts.append(query_section)
        parts.append("Answer:")

        logger.info(
            "context_assembled",
            docs_added=docs_added,
            docs_total=len(documents),
            tokens_used=current_budget.used_tokens,
            tokens_remaining=current_budget.remaining,
        )

        return "\n".join(parts), current_budget
