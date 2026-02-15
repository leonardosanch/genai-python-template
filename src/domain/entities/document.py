"""Document entity — represents a piece of content in the system."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Document:
    """Immutable document entity used across the system.

    Used in RAG pipelines, vector stores, and LLM context building.
    Frozen to ensure immutability — domain entities should not be
    mutated after creation.
    """

    content: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    id: str | None = None
    score: float | None = None
    created_at: datetime | None = None

    def chunk(self, max_length: int = 1000, overlap: int = 200) -> list["Document"]:
        """Split document into smaller chunks with overlap.

        Args:
            max_length: Maximum characters per chunk.
            overlap: Characters of overlap between consecutive chunks.

        Returns:
            List of Document chunks preserving metadata.
        """
        chunks = []
        start = 0
        while start < len(self.content):
            end = start + max_length
            chunk_content = self.content[start:end]
            chunks.append(
                Document(
                    content=chunk_content,
                    metadata={**self.metadata, "chunk_start": start},
                )
            )
            start += max_length - overlap
        return chunks
