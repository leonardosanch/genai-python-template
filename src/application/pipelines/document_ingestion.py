"""Document ingestion pipeline — ETL for loading documents into vector store.

This pipeline reads text files from storage, chunks them, and loads them
into the vector store for retrieval.
"""

import hashlib

import structlog

from src.application.pipelines.base import Pipeline
from src.domain.entities.document import Document
from src.domain.exceptions import ExtractionError, LoadError, TransformationError
from src.domain.ports.storage_port import StoragePort
from src.domain.ports.vector_store_port import VectorStorePort

logger = structlog.get_logger(__name__)


class DocumentIngestionPipeline(Pipeline):
    """Pipeline for ingesting documents from storage into vector store.

    ETL phases:
    - Extract: Read text files from storage
    - Transform: Chunk documents, generate IDs, clean content
    - Load: Upsert chunks to vector store
    """

    def __init__(
        self,
        storage: StoragePort,
        vector_store: VectorStorePort,
        source_prefix: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize document ingestion pipeline.

        Args:
            storage: Storage adapter for reading files
            vector_store: Vector store for loading embeddings
            source_prefix: Path prefix or pattern for files to ingest
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.storage = storage
        self.vector_store = vector_store
        self.source_prefix = source_prefix
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def extract(self) -> list[dict[str, object]]:
        """Extract text files from storage.

        Returns:
            List of records with 'path' and 'content' keys

        Raises:
            ExtractionError: If file reading fails
        """
        try:
            # List all files matching the prefix
            file_paths = await self.storage.list(self.source_prefix)
            logger.info("extraction.files_found", count=len(file_paths))

            records = []
            for path in file_paths:
                try:
                    # Read file content
                    content_bytes = await self.storage.read(path)
                    content = content_bytes.decode("utf-8")

                    record: dict[str, object] = {
                        "path": path,
                        "content": content,
                    }
                    records.append(record)
                except Exception as e:
                    logger.warning(
                        "extraction.file_read_failed",
                        path=path,
                        error=str(e),
                    )
                    # Continue with other files
                    continue

            return records

        except Exception as e:
            raise ExtractionError(f"Failed to extract documents: {e}") from e

    async def transform(self, data: list[dict[str, object]]) -> list[dict[str, object]]:
        """Transform extracted data into chunks.

        Args:
            data: Raw records from extraction

        Returns:
            List of chunk records with 'id', 'content', and 'metadata'

        Raises:
            TransformationError: If transformation fails
        """
        try:
            chunks = []

            for record in data:
                path = str(record["path"])
                content = str(record["content"])

                # Clean whitespace
                content = content.strip()
                if not content:
                    continue

                # Create document and chunk it
                doc = Document(
                    content=content,
                    metadata={"source": path},
                )

                doc_chunks = doc.chunk(
                    max_length=self.chunk_size,
                    overlap=self.chunk_overlap,
                )

                # Generate deterministic IDs and prepare for loading
                for chunk in doc_chunks:
                    # Deterministic ID based on content hash
                    chunk_id = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]

                    chunk_record: dict[str, object] = {
                        "id": chunk_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    }
                    chunks.append(chunk_record)

            logger.info("transformation.chunks_created", count=len(chunks))
            return chunks

        except Exception as e:
            raise TransformationError(f"Failed to transform documents: {e}") from e

    async def load(self, data: list[dict[str, object]]) -> None:
        """Load chunks into vector store.

        Args:
            data: Transformed chunk records

        Raises:
            LoadError: If loading to vector store fails
        """
        try:
            # Convert to Document entities
            documents = []
            for record in data:
                metadata_obj = record.get("metadata", {})
                # Ensure metadata is the correct type
                if isinstance(metadata_obj, dict):
                    metadata: dict[str, str | int | float | bool] = {
                        k: v
                        for k, v in metadata_obj.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                else:
                    metadata = {}

                documents.append(
                    Document(
                        id=str(record["id"]),
                        content=str(record["content"]),
                        metadata=metadata,
                    )
                )

            # Upsert to vector store
            await self.vector_store.upsert(documents)

            logger.info("load.documents_upserted", count=len(documents))

        except Exception as e:
            raise LoadError(f"Failed to load documents to vector store: {e}") from e
