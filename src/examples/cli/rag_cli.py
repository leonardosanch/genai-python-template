"""
RAG CLI Tool Example

Demonstrates:
- CLI application with Typer
- Document management commands
- Interactive query mode
- Progress indicators
- Rich console output
- Configuration management

Usage:
    export OPENAI_API_KEY="sk-..."

    # Index a document
    uv run python -m src.examples.cli.rag_cli index document.txt

    # Query
    uv run python -m src.examples.cli.rag_cli query "What is RAG?"

    # Interactive mode
    uv run python -m src.examples.cli.rag_cli interactive

    # List documents
    uv run python -m src.examples.cli.rag_cli list
"""

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import chromadb
import typer
from chromadb.config import Settings
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="rag-cli",
    help="RAG CLI - Manage documents and query knowledge base",
    add_completion=False,
)

console = Console()


# RAG Client


class RAGClient:
    """RAG client for CLI."""

    def __init__(self) -> None:
        """Initialize RAG client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY environment variable required[/red]")
            raise typer.Exit(1)

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"

        # ChromaDB
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = self.chroma_client.get_or_create_collection(name="rag_cli")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Chunk text."""
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

    async def index_file(self, file_path: Path) -> int:
        """Index a file."""
        # Read file
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Chunk
        chunks = self.chunk_text(content)

        # Generate IDs
        doc_id = file_path.stem
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"filename": file_path.name, "doc_id": doc_id, "chunk": str(i)}
            for i in range(len(chunks))
        ]

        # Embed
        embeddings = await self.embed_texts(chunks)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=cast(Any, embeddings),
            metadatas=metadatas,  # type: ignore
        )

        return len(chunks)

    async def query(self, question: str, top_k: int = 3) -> tuple[str, list[str]]:
        """Query RAG system."""
        # Embed query
        query_embedding = (await self.embed_texts([question]))[0]

        # Search
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
        )

        if not results["documents"] or not results["documents"][0]:
            return "No documents indexed yet.", []

        # Build context
        context = "\n\n".join(f"[{i + 1}]: {doc}" for i, doc in enumerate(results["documents"][0]))

        # Generate
        prompt = f"""Answer based on context.

Context:
{context}

Question: {question}

Answer:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content or ""

        # Extract sources
        sources: list[str] = []
        if results["metadatas"] and results["metadatas"][0]:
            sources = list(
                {
                    str(meta.get("filename", "unknown"))
                    for meta in (results["metadatas"][0] or [])
                    if meta
                }
            )

        return answer, sources

    def get_stats(self) -> dict[str, int]:
        """Get collection statistics."""
        count = self.collection.count()

        # Get unique documents
        if count > 0:
            results = self.collection.get()
            unique_docs = {meta.get("doc_id", "unknown") for meta in (results["metadatas"] or [])}
            num_docs = len(unique_docs)
        else:
            num_docs = 0

        return {"total_chunks": count, "unique_documents": num_docs}

    def clear(self) -> None:
        """Clear all documents."""
        self.chroma_client.delete_collection("rag_cli")
        self.collection = self.chroma_client.get_or_create_collection(name="rag_cli")


# Commands


@app.command()
def index(
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to text file to index",
    ),
) -> None:
    """Index a document."""
    console.print(f"\n[bold]Indexing:[/bold] {file_path.name}")

    client = RAGClient()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)

        async def _index() -> int:
            return await client.index_file(file_path)

        chunks = asyncio.run(_index())
        progress.update(task, completed=True)

    console.print(f"[green]✓[/green] Indexed {chunks} chunks from {file_path.name}")

    # Show stats
    stats = client.get_stats()
    console.print(
        f"\n[dim]Total: {stats['unique_documents']} documents, {stats['total_chunks']} chunks[/dim]"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(3, help="Number of chunks to retrieve"),
) -> None:
    """Query the knowledge base."""
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    client = RAGClient()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)

        async def _query() -> tuple[str, list[str]]:
            return await client.query(question, top_k=top_k)

        answer, sources = asyncio.run(_query())
        progress.update(task, completed=True)

    # Display answer
    console.print(
        Panel(
            answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )

    # Display sources
    if sources:
        console.print(f"\n[dim]Sources: {', '.join(sources)}[/dim]")


@app.command()
def interactive() -> None:
    """Interactive query mode."""
    console.print(
        Panel(
            "[bold]RAG Interactive Mode[/bold]\n\n"
            "Ask questions about your indexed documents.\n"
            "Type 'exit' or 'quit' to leave.",
            border_style="blue",
        )
    )

    client = RAGClient()

    # Show stats
    stats = client.get_stats()
    if stats["unique_documents"] == 0:
        console.print(
            "\n[yellow]Warning: No documents indexed yet. "
            "Use 'rag-cli index <file>' first.[/yellow]\n"
        )

    while True:
        try:
            question = console.input("\n[bold cyan]Question:[/bold cyan] ")

            if question.lower() in ["exit", "quit", "q"]:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not question.strip():
                continue

            async def _query() -> tuple[str, list[str]]:
                return await client.query(question)

            answer, sources = asyncio.run(_query())

            console.print(f"\n[bold green]Answer:[/bold green] {answer}")

            if sources:
                console.print(f"[dim]Sources: {', '.join(sources)}[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.command("list")
def list_docs() -> None:
    """List indexed documents."""
    client = RAGClient()
    stats = client.get_stats()

    if stats["unique_documents"] == 0:
        console.print("\n[yellow]No documents indexed yet.[/yellow]")
        return

    # Get all documents
    results = client.collection.get()

    # Group by document
    docs: dict[str, int] = {}
    for meta in results["metadatas"] or []:
        doc_id = meta.get("doc_id", "unknown")
        filename = meta.get("filename", "unknown")
        key = f"{doc_id} ({filename})"
        docs[key] = docs.get(key, 0) + 1

    # Create table
    table = Table(title="Indexed Documents", show_header=True)
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right", style="green")

    for doc, chunks in sorted(docs.items()):
        table.add_row(doc, str(chunks))

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Total: {stats['unique_documents']} documents, {stats['total_chunks']} chunks[/dim]"
    )


@app.command()
def clear(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Clear all indexed documents."""
    if not force:
        confirm = typer.confirm("Are you sure you want to clear all documents?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Abort()

    client = RAGClient()
    client.clear()

    console.print("[green]✓[/green] All documents cleared")


@app.command()
def stats() -> None:
    """Show statistics."""
    client = RAGClient()
    stats = client.get_stats()

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Documents", str(stats["unique_documents"]))
    table.add_row("Chunks", str(stats["total_chunks"]))

    console.print()
    console.print(table)


if __name__ == "__main__":
    app()
