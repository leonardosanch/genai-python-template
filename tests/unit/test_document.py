"""Unit tests for Document entity.

Reference test showing:
- Pure domain logic testing (no mocks, no I/O)
- Testing business rules
"""

from dataclasses import FrozenInstanceError

from src.domain.entities.document import Document


class TestDocumentChunk:
    def test_chunk_respects_max_length(self) -> None:
        doc = Document(content="a" * 1000)
        chunks = doc.chunk(max_length=200, overlap=0)
        assert all(len(c.content) <= 200 for c in chunks)

    def test_chunk_preserves_metadata(self) -> None:
        doc = Document(content="hello world", metadata={"source": "test"})
        chunks = doc.chunk(max_length=5, overlap=0)
        assert all(c.metadata["source"] == "test" for c in chunks)

    def test_chunk_adds_position_metadata(self) -> None:
        doc = Document(content="a" * 100)
        chunks = doc.chunk(max_length=30, overlap=0)
        assert chunks[0].metadata["chunk_start"] == 0
        assert chunks[1].metadata["chunk_start"] == 30

    def test_chunk_with_overlap(self) -> None:
        doc = Document(content="a" * 100)
        chunks = doc.chunk(max_length=50, overlap=10)
        assert len(chunks) >= 2
        # With overlap, chunks start closer together
        assert chunks[1].metadata["chunk_start"] == 40

    def test_single_chunk_when_content_fits(self) -> None:
        doc = Document(content="short")
        chunks = doc.chunk(max_length=1000)
        assert len(chunks) == 1
        assert chunks[0].content == "short"


class TestDocumentImmutability:
    def test_document_is_frozen(self) -> None:
        doc = Document(content="test")
        try:
            doc.content = "changed"  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except FrozenInstanceError:
            pass
