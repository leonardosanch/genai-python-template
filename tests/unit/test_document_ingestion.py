"""Unit tests for DocumentIngestionPipeline.

Tests the complete ETL flow for document ingestion with mocked dependencies.
"""

from unittest.mock import AsyncMock

from src.application.pipelines.document_ingestion import DocumentIngestionPipeline
from src.domain.entities.document import Document


class TestDocumentIngestionPipeline:
    """Test document ingestion pipeline."""

    async def test_complete_pipeline_execution(self) -> None:
        """Test full ETL flow with successful execution."""
        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["doc1.txt", "doc2.txt"]
        mock_storage.read.side_effect = [
            b"This is document one content.",
            b"This is document two content.",
        ]

        # Mock vector store
        mock_vector_store = AsyncMock()

        # Create pipeline
        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="documents/",
            chunk_size=20,
            chunk_overlap=5,
        )

        # Run pipeline
        result = await pipeline.run()

        # Verify storage interactions
        mock_storage.list.assert_called_once_with("documents/")
        assert mock_storage.read.call_count == 2

        # Verify vector store was called
        mock_vector_store.upsert.assert_called_once()
        upserted_docs = mock_vector_store.upsert.call_args[0][0]

        # Verify documents were chunked (content > chunk_size)
        assert len(upserted_docs) > 2  # More chunks than original files
        assert all(isinstance(doc, Document) for doc in upserted_docs)
        assert all(doc.id is not None for doc in upserted_docs)

        # Verify result
        assert result.status == "success"
        assert result.records_processed > 0
        assert result.records_failed == 0
        assert result.duration_seconds > 0

    async def test_chunking_logic(self) -> None:
        """Verify chunking respects size and overlap parameters."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["test.txt"]
        # Content: 50 characters
        mock_storage.read.return_value = b"a" * 50

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
            chunk_size=20,
            chunk_overlap=5,
        )

        await pipeline.run()

        upserted_docs = mock_vector_store.upsert.call_args[0][0]

        # Verify chunks were created
        assert len(upserted_docs) > 1

        # Verify chunk sizes
        for doc in upserted_docs:
            assert len(doc.content) <= 20

    async def test_deterministic_ids(self) -> None:
        """Verify IDs are deterministic based on content hash."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["doc.txt"]
        mock_storage.read.return_value = b"Same content"

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
            chunk_size=100,
            chunk_overlap=0,
        )

        # Run twice
        await pipeline.run()
        first_docs = mock_vector_store.upsert.call_args[0][0]

        mock_vector_store.reset_mock()
        await pipeline.run()
        second_docs = mock_vector_store.upsert.call_args[0][0]

        # Same content should produce same IDs
        assert first_docs[0].id == second_docs[0].id

    async def test_metadata_preservation(self) -> None:
        """Verify source metadata is preserved in chunks."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["source.txt"]
        mock_storage.read.return_value = b"Content"

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="docs/",
            chunk_size=100,
            chunk_overlap=0,
        )

        await pipeline.run()

        upserted_docs = mock_vector_store.upsert.call_args[0][0]
        assert all(doc.metadata.get("source") == "source.txt" for doc in upserted_docs)

    async def test_empty_directory(self) -> None:
        """Test pipeline with no files to process."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = []

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="empty/",
        )

        result = await pipeline.run()

        # Should complete successfully with 0 records
        assert result.status == "success"
        assert result.records_processed == 0
        mock_vector_store.upsert.assert_called_once_with([])

    async def test_extraction_error_handling(self) -> None:
        """Test error handling during extraction phase."""
        mock_storage = AsyncMock()
        mock_storage.list.side_effect = Exception("Storage unavailable")

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
        )

        result = await pipeline.run()

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "ExtractionError" in result.errors[0]

    async def test_partial_file_read_failure(self) -> None:
        """Test pipeline continues when some files fail to read."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["good.txt", "bad.txt"]
        mock_storage.read.side_effect = [
            b"Good content",
            Exception("Read failed"),
        ]

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
        )

        result = await pipeline.run()

        # Should succeed with partial data
        assert result.status == "success"
        assert result.records_processed > 0

        # Only good file should be upserted
        upserted_docs = mock_vector_store.upsert.call_args[0][0]
        assert all("Good content" in doc.content for doc in upserted_docs)

    async def test_load_error_handling(self) -> None:
        """Test error handling during load phase."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["doc.txt"]
        mock_storage.read.return_value = b"Content"

        mock_vector_store = AsyncMock()
        mock_vector_store.upsert.side_effect = Exception("Vector store down")

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
        )

        result = await pipeline.run()

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "LoadError" in result.errors[0]

    async def test_whitespace_cleaning(self) -> None:
        """Test that whitespace is cleaned from content."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["doc.txt"]
        mock_storage.read.return_value = b"  \n  Content with whitespace  \n  "

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
        )

        await pipeline.run()

        upserted_docs = mock_vector_store.upsert.call_args[0][0]
        assert upserted_docs[0].content == "Content with whitespace"

    async def test_empty_file_skipped(self) -> None:
        """Test that empty files are skipped."""
        mock_storage = AsyncMock()
        mock_storage.list.return_value = ["empty.txt", "content.txt"]
        mock_storage.read.side_effect = [
            b"   \n   ",  # Only whitespace
            b"Real content",
        ]

        mock_vector_store = AsyncMock()

        pipeline = DocumentIngestionPipeline(
            storage=mock_storage,
            vector_store=mock_vector_store,
            source_prefix="test/",
        )

        await pipeline.run()

        upserted_docs = mock_vector_store.upsert.call_args[0][0]
        # Only one document should be upserted (the non-empty one)
        assert all("Real content" in doc.content for doc in upserted_docs)
