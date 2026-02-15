---
description: Scaffold complete RAG pipeline (ingestion + retrieval + generation)
---

1. Ask the user for:
   - Pipeline name
   - Document types to support (PDF, HTML, Markdown, etc.)
   - Vector store (Pinecone, Qdrant, pgvector, ChromaDB)
   - Chunking strategy (recursive, semantic, fixed)
   - Reranking (yes/no)

2. Read `docs/skills/genai_rag.md` for RAG patterns.

3. Read `docs/skills/databases.md` for vector store guidance.

4. Create the following files:

   **a. Document Loader**
   - Path: `src/infrastructure/document_loaders/<pipeline_name>_loader.py`
   - Support configured document types
   - Return `list[Document]` with metadata

   **b. Chunking Strategy**
   - Path: `src/application/rag/<pipeline_name>/chunker.py`
   - Implement selected chunking strategy
   - Configurable chunk_size and overlap

   **c. Embedding Service**
   - Path: `src/infrastructure/embeddings/<pipeline_name>_embeddings.py`
   - Abstract embedding provider
   - Async batch processing

   **d. Vector Store Adapter**
   - Path: `src/infrastructure/vector_stores/<pipeline_name>_store.py`
   - Implement selected vector store
   - CRUD operations for documents

   **e. Retriever**
   - Path: `src/application/rag/<pipeline_name>/retriever.py`
   - Hybrid search if reranking enabled
   - Return `list[RetrievedDocument]` with scores

   **f. RAG Chain**
   - Path: `src/application/rag/<pipeline_name>/chain.py`
   - Complete RAG pipeline
   - Streaming support
   - Source attribution

   **g. Ingestion Pipeline**
   - Path: `src/application/rag/<pipeline_name>/ingestion.py`
   - Batch document processing
   - Progress tracking
   - Idempotent (skip already indexed)

   **h. Tests**
   - Path: `tests/application/rag/<pipeline_name>/`
   - Unit tests for each component
   - Integration test for full pipeline

5. Apply these rules:
   - Reranking is MANDATORY for production (use Cohere or cross-encoder)
   - All operations are async
   - Structured output with Pydantic
   - Comprehensive error handling
   - Logging with document IDs for tracing

6. Output all files ready to use with `uv run`.
