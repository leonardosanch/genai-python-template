"""
Complete Application Example

Demonstrates:
- Full-stack RAG application
- FastAPI backend
- Document management
- Query interface
- Monitoring
- Production patterns

Usage:
    export OPENAI_API_KEY="sk-..."

    uvicorn src.examples.integration.complete_app:app --reload

    # Access at http://localhost:8000
"""

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Models


class Document(BaseModel):
    """Document model."""

    content: str = Field(..., min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Query request."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class QueryResponse(BaseModel):
    """Query response."""

    answer: str
    sources: list[str]
    latency_ms: float
    tokens: int


class Stats(BaseModel):
    """System statistics."""

    total_documents: int
    total_queries: int
    avg_latency_ms: float
    total_tokens: int


# RAG System


class RAGSystem:
    """Complete RAG system."""

    def __init__(self):
        """Initialize system."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # ChromaDB
        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = chroma_client.get_or_create_collection(name="complete_app")

        # Stats
        self.total_queries = 0
        self.total_tokens = 0
        self.latencies: list[float] = []

    async def add_document(self, content: str, metadata: dict[str, str]) -> str:
        """Add document."""
        # Generate embedding
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[content],
        )
        embedding = response.data[0].embedding

        # Generate ID
        doc_id = f"doc_{int(time.time() * 1000)}"

        # Add to collection
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],  # type: ignore
            metadatas=[metadata],  # type: ignore
        )

        return doc_id

    async def query(self, question: str, top_k: int = 3) -> QueryResponse:
        """Query system."""
        start = time.time()

        # Embed query
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[question],
        )
        query_embedding = response.data[0].embedding

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore
            n_results=top_k,
        )

        if not results["documents"] or not results["documents"][0]:
            raise HTTPException(
                status_code=404,
                detail="No documents found",
            )

        # Build context
        context = "\n\n".join(f"[{i + 1}]: {doc}" for i, doc in enumerate(results["documents"][0]))

        # Generate
        llm_response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Answer based on context:\n\n{context}\n\nQuestion: {question}",
                }
            ],
            temperature=0.3,
        )

        answer = llm_response.choices[0].message.content or ""
        usage = llm_response.usage
        tokens = usage.total_tokens if usage else 0

        # Calculate latency
        latency_ms = (time.time() - start) * 1000

        # Update stats
        self.total_queries += 1
        self.total_tokens += tokens
        self.latencies.append(latency_ms)

        # Extract sources
        sources = [meta.get("filename", "unknown") for meta in (results["metadatas"][0] or [])]

        return QueryResponse(
            answer=answer,
            sources=list(set(sources)),
            latency_ms=latency_ms,
            tokens=tokens,
        )

    def get_stats(self) -> Stats:
        """Get statistics."""
        return Stats(
            total_documents=self.collection.count(),
            total_queries=self.total_queries,
            avg_latency_ms=(sum(self.latencies) / len(self.latencies) if self.latencies else 0.0),
            total_tokens=self.total_tokens,
        )


# FastAPI App


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan."""
    print("Starting Complete RAG Application...")
    app.state.rag = RAGSystem()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Complete RAG Application",
    description="Full-featured RAG system",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """Home page."""
    return HTMLResponse("""
    <html>
        <head>
            <title>RAG Application</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 50px auto; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>🚀 Complete RAG Application</h1>
            <p>Full-featured RAG system with FastAPI</p>

            <h2>Endpoints:</h2>
            <div class="endpoint">
                <b>POST /documents</b> - Add document
            </div>
            <div class="endpoint">
                <b>POST /query</b> - Query system
            </div>
            <div class="endpoint">
                <b>GET /stats</b> - View statistics
            </div>
            <div class="endpoint">
                <b>GET /docs</b> - API documentation
            </div>
        </body>
    </html>
    """)


@app.post("/documents", status_code=201)
async def add_document(doc: Document) -> dict[str, str]:
    """Add document."""
    doc_id = await app.state.rag.add_document(doc.content, doc.metadata)
    return {"document_id": doc_id, "message": "Document added"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Query system."""
    return await app.state.rag.query(request.question, request.top_k)


@app.get("/stats", response_model=Stats)
async def get_stats() -> Stats:
    """Get statistics."""
    return app.state.rag.get_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
