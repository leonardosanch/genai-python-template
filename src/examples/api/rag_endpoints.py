"""
FastAPI RAG Endpoints Example

Demonstrates:
- REST API for RAG system
- Document upload and indexing
- Query endpoint with streaming
- Error handling and validation
- CORS configuration
- API documentation

Usage:
    export OPENAI_API_KEY="sk-..."

    # Run server
    uvicorn src.examples.api.rag_endpoints:app --reload

    # Or with uv
    uv run uvicorn src.examples.api.rag_endpoints:app --reload

    # Access docs at http://localhost:8000/docs
"""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Request/Response Models


class QueryRequest(BaseModel):
    """RAG query request."""

    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(3, ge=1, le=10)
    stream: bool = Field(False, description="Stream response tokens")


class QueryResponse(BaseModel):
    """RAG query response."""

    question: str
    answer: str
    sources: list[str]
    tokens_used: int
    cost_usd: float


class Document(BaseModel):
    """Document for indexing."""

    content: str = Field(..., min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    """Document indexing response."""

    document_id: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    documents_indexed: int
    version: str


# RAG Service


class RAGService:
    """RAG service for API."""

    def __init__(self) -> None:
        """Initialize RAG service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"

        # ChromaDB
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = self.chroma_client.get_or_create_collection(name="rag_api")

        # Pricing
        self.input_cost_per_1m = 0.15
        self.output_cost_per_1m = 0.60

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Chunk text into smaller pieces."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period > chunk_size * 0.5:
                    end = start + last_period + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings."""
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def index_document(self, doc_id: str, content: str, metadata: dict[str, str]) -> int:
        """
        Index a document.

        Returns:
            Number of chunks created
        """
        chunks = self.chunk_text(content)

        # Generate IDs and metadata for chunks
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "doc_id": doc_id, "chunk": str(i)} for i in range(len(chunks))]

        # Generate embeddings
        embeddings = await self.embed_texts(chunks)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=cast(Any, embeddings),
            metadatas=metadatas,  # type: ignore
        )

        return len(chunks)

    async def query(self, question: str, top_k: int = 3) -> tuple[str, list[str], int, float]:
        """
        Query RAG system.

        Returns:
            (answer, sources, tokens_used, cost_usd)
        """
        # Embed query
        query_embedding = (await self.embed_texts([question]))[0]

        # Search
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
        )

        # Build context
        if not results["documents"] or not results["documents"][0]:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please index documents first.",
            )

        context = "\n\n".join(f"[{i + 1}]: {doc}" for i, doc in enumerate(results["documents"][0]))

        # Generate answer
        prompt = f"""Answer based on context. Be concise.

Context:
{context}

Question: {question}

Answer:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=False,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        if not usage:
            raise ValueError("No usage data")

        # Calculate cost
        tokens_used = usage.total_tokens
        cost = (
            usage.prompt_tokens * self.input_cost_per_1m / 1_000_000
            + usage.completion_tokens * self.output_cost_per_1m / 1_000_000
        )

        # Extract sources
        sources: list[str] = []
        if results["metadatas"] and results["metadatas"][0]:
            sources = [
                str(meta.get("doc_id", "unknown"))
                for meta in (results["metadatas"][0] or [])
                if meta
            ]

        return answer, list(set(sources)), tokens_used, cost

    async def stream_query(self, question: str, top_k: int = 3) -> AsyncIterator[str]:
        """Stream query response."""
        # Embed and retrieve (same as query)
        query_embedding = (await self.embed_texts([question]))[0]

        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
        )

        if not results["documents"] or not results["documents"][0]:
            yield "Error: No documents indexed"
            return

        context = "\n\n".join(f"[{i + 1}]: {doc}" for i, doc in enumerate(results["documents"][0]))

        prompt = f"""Answer based on context.

Context:
{context}

Question: {question}

Answer:"""

        # Stream response
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        return self.collection.count()


# FastAPI Application


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager."""
    # Startup
    print("Starting RAG API...")
    app.state.rag = RAGService()
    print(f"Indexed documents: {app.state.rag.get_document_count()}")

    yield

    # Shutdown
    print("Shutting down RAG API...")


app = FastAPI(
    title="RAG API",
    description="Production-ready RAG API with FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    rag: RAGService = app.state.rag
    return HealthResponse(
        status="healthy",
        documents_indexed=rag.get_document_count(),
        version="1.0.0",
    )


@app.post("/documents", response_model=IndexResponse, status_code=201)
async def index_document(document: Document) -> IndexResponse:
    """
    Index a document.

    The document will be chunked and embedded automatically.
    """
    import hashlib
    import time

    # Generate document ID
    doc_id = hashlib.md5(f"{document.content[:100]}{time.time()}".encode()).hexdigest()[:12]

    try:
        rag: RAGService = app.state.rag
        chunks_created = await rag.index_document(
            doc_id=doc_id,
            content=document.content,
            metadata=document.metadata,
        )

        return IndexResponse(
            document_id=doc_id,
            chunks_created=chunks_created,
            message=f"Document indexed successfully with {chunks_created} chunks",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_document(file: UploadFile) -> IndexResponse:
    """Upload and index a text file."""
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported",
        )

    try:
        content = await file.read()
        text = content.decode("utf-8")

        doc = Document(
            content=text,
            metadata={"filename": file.filename},
        )

        return await index_document(doc)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be valid UTF-8 text",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system.

    Returns an answer based on indexed documents.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /query/stream for streaming responses",
        )

    try:
        rag: RAGService = app.state.rag
        answer, sources, tokens, cost = await rag.query(
            question=request.question,
            top_k=request.top_k,
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            tokens_used=tokens,
            cost_usd=cost,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def stream_query(request: QueryRequest) -> StreamingResponse:
    """
    Stream query response.

    Returns Server-Sent Events with answer tokens.
    """

    async def generate() -> AsyncIterator[str]:
        try:
            rag: RAGService = app.state.rag
            async for chunk in rag.stream_query(
                question=request.question,
                top_k=request.top_k,
            ):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)  # Small delay for demo
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# Example usage in docstring
"""
Example API Usage:

# 1. Index a document
curl -X POST http://localhost:8000/documents \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "FastAPI is a modern web framework for Python...",
    "metadata": {"source": "docs", "topic": "fastapi"}
  }'

# 2. Query
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is FastAPI?",
    "top_k": 3
  }'

# 3. Stream query
curl -X POST http://localhost:8000/query/stream \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "Explain FastAPI",
    "top_k": 3
  }'

# 4. Upload file
curl -X POST http://localhost:8000/documents/upload \\
  -F "file=@document.txt"
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
