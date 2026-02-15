"""
Naive RAG Example

Demonstrates:
- Document loading and chunking
- Embedding generation
- Vector search with ChromaDB
- Basic RAG generation

This is the simplest RAG implementation. For production, see advanced_rag.py.

Usage:
    export OPENAI_API_KEY="sk-..."

    python -m src.examples.rag.naive_rag
"""

import asyncio
import os

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from pydantic import BaseModel


class Document(BaseModel):
    """A document chunk."""

    id: str
    content: str
    metadata: dict[str, str] = {}


class RAGResponse(BaseModel):
    """RAG query response."""

    question: str
    answer: str
    sources: list[str]
    context_used: list[str]


class NaiveRAG:
    """
    Naive RAG implementation.

    Pipeline: Query → Retrieve → Generate

    Limitations:
    - No query transformation
    - No reranking
    - Simple chunking strategy
    - No evaluation
    """

    def __init__(self, collection_name: str = "naive_rag"):
        """
        Initialize Naive RAG.

        Args:
            collection_name: ChromaDB collection name
        """
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.llm_client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Naive RAG example collection"},
        )

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Simple fixed-size chunking.

        Args:
            text: Text to chunk
            chunk_size: Characters per chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period > chunk_size * 0.5:  # At least 50% into chunk
                    end = start + last_period + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        response = await self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    async def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to vector store.

        Args:
            documents: Documents to add
        """
        if not documents:
            return

        # Extract content and metadata
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = await self.embed_texts(contents)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,  # type: ignore
            metadatas=metadatas,  # type: ignore
        )

        print(f"Added {len(documents)} documents to vector store")

    async def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        # Embed query
        query_embedding = (await self.embed_texts([query]))[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore
            n_results=top_k,
        )

        # Convert to Document objects
        documents = []
        if (
            results["ids"]
            and results["ids"][0]
            and results["documents"]
            and results["documents"][0]
            and results["metadatas"]
            and results["metadatas"][0]
        ):
            for i, doc_id in enumerate(results["ids"][0]):
                meta_raw = results["metadatas"][0][i] or {}
                meta: dict[str, str] = {k: str(v) for k, v in meta_raw.items()}
                documents.append(
                    Document(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=meta,
                    )
                )

        return documents

    async def generate(self, query: str, context_docs: list[Document]) -> str:
        """
        Generate answer from query and context.

        Args:
            query: User question
            context_docs: Retrieved context documents

        Returns:
            Generated answer
        """
        # Build context
        context = "\n\n".join(
            f"[Source {i + 1}]: {doc.content}" for i, doc in enumerate(context_docs)
        )

        # Build prompt
        prompt = f"""Answer the question based on the context below.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        system_msg = "You are a helpful assistant that answers questions based on provided context."
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content or ""

    async def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """
        Complete RAG query pipeline.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            RAG response with answer and sources
        """
        # 1. Retrieve
        docs = await self.retrieve(question, top_k=top_k)

        # 2. Generate
        answer = await self.generate(question, docs)

        # 3. Build response
        return RAGResponse(
            question=question,
            answer=answer,
            sources=[doc.metadata.get("source", doc.id) for doc in docs],
            context_used=[doc.content[:100] + "..." for doc in docs],
        )


async def main() -> None:
    """Run example demonstration."""
    print("=" * 60)
    print("Naive RAG Example")
    print("=" * 60)

    # Initialize RAG
    rag = NaiveRAG(collection_name="naive_rag_demo")

    # Sample documents about Clean Architecture
    sample_docs = [
        """
        Clean Architecture is a software design philosophy that separates concerns
        into layers. The core principle is the Dependency Rule: source code dependencies
        must point only inward, toward higher-level policies. The innermost layer contains
        business logic (entities), surrounded by use cases, then interface adapters,
        and finally frameworks and drivers on the outside.
        """,
        """
        The main benefit of Clean Architecture is independence. The business rules don't
        know anything about the UI, database, or external frameworks. This makes the
        system testable, as you can test business rules without the UI, database, or
        any external element. It also makes the system flexible - you can change
        frameworks, databases, or UI without affecting business logic.
        """,
        """
        In Clean Architecture, entities are enterprise-wide business rules. Use cases
        contain application-specific business rules. Interface adapters convert data
        between the format most convenient for use cases and entities, and the format
        most convenient for external agencies like databases and web frameworks.
        """,
        """
        Dependency Inversion Principle is crucial in Clean Architecture. High-level
        modules should not depend on low-level modules. Both should depend on abstractions.
        This is achieved through interfaces (ports) that define contracts between layers.
        """,
    ]

    # Chunk and add documents
    print("\n1. Indexing Documents")
    print("-" * 60)

    all_chunks = []
    for i, doc_text in enumerate(sample_docs):
        chunks = rag.chunk_text(doc_text.strip(), chunk_size=200, overlap=20)
        for j, chunk in enumerate(chunks):
            all_chunks.append(
                Document(
                    id=f"doc_{i}_chunk_{j}",
                    content=chunk,
                    metadata={"source": f"document_{i}", "chunk": str(j)},
                )
            )

    await rag.add_documents(all_chunks)
    print(f"Total chunks indexed: {len(all_chunks)}")

    # Query examples
    print("\n2. Querying RAG System")
    print("-" * 60)

    questions = [
        "What is Clean Architecture?",
        "What are the benefits of Clean Architecture?",
        "What is the Dependency Rule?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("─" * 60)

        response = await rag.query(question, top_k=2)

        print(f"Answer: {response.answer}\n")
        print(f"Sources used: {', '.join(response.sources)}")
        print("\nContext snippets:")
        for i, context in enumerate(response.context_used, 1):
            print(f"  {i}. {context}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nLimitations of Naive RAG:")
    print("- No query transformation or expansion")
    print("- No reranking of retrieved documents")
    print("- Simple fixed-size chunking")
    print("- No evaluation metrics")
    print("\nSee advanced_rag.py for production-ready implementation")


if __name__ == "__main__":
    asyncio.run(main())
