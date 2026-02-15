"""CLI interface for genai-python-template.

Provides commands for pipeline management, queries, and health checks.
"""

import asyncio
import json as json_lib

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.infrastructure.config import get_settings

app = typer.Typer(
    name="genai",
    help="GenAI Python Template CLI",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    source_path: str = typer.Option(
        ...,
        "--source-path",
        "-s",
        help="Path prefix or pattern for files to ingest",
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        "-c",
        help="Maximum characters per chunk",
    ),
    chunk_overlap: int = typer.Option(
        200,
        "--chunk-overlap",
        "-o",
        help="Overlap between consecutive chunks",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what would be ingested without executing",
    ),
) -> None:
    """Run document ingestion pipeline.

    Reads text files from storage, chunks them, and loads them into the vector store.
    """
    if dry_run:
        console.print("[bold yellow]DRY RUN — no data will be ingested[/bold yellow]")
        console.print(f"Source path: {source_path}")
        console.print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        return

    async def run_pipeline() -> None:
        # Lazy imports to avoid heavy dependencies at startup
        from src.application.pipelines.document_ingestion import DocumentIngestionPipeline
        from src.infrastructure.storage.local_storage import LocalStorage
        from src.infrastructure.vectorstores.chromadb_adapter import ChromaDBAdapter

        settings = get_settings()
        storage_path = getattr(settings, "STORAGE_PATH", "./data")
        storage = LocalStorage(base_path=storage_path)
        vector_store = ChromaDBAdapter()

        pipeline = DocumentIngestionPipeline(
            storage=storage,
            vector_store=vector_store,
            source_prefix=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Ingesting documents...", total=None)
            result = await pipeline.run()

        table = Table(title="Pipeline Execution Result")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Status", result.status)
        table.add_row("Records Processed", str(result.records_processed))
        table.add_row("Records Failed", str(result.records_failed))
        table.add_row("Duration (seconds)", f"{result.duration_seconds:.2f}")
        console.print(table)

        if result.errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in result.errors:
                console.print(f"  - {error}")

    asyncio.run(run_pipeline())


@app.command()
def health(
    api_url: str = typer.Option(
        "http://localhost:8000",
        "--api-url",
        "-u",
        help="Base URL of the API",
    ),
) -> None:
    """Check API health status.

    Makes a GET request to the /health endpoint.
    """
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        response.raise_for_status()

        data = response.json()
        status = data.get("status", "unknown")

        if status == "healthy":
            console.print(f"[bold green]OK[/bold green] API is {status}")
        else:
            console.print(f"[bold yellow]WARN[/bold yellow] API status: {status}")

    except httpx.HTTPError as e:
        console.print(f"[bold red]FAIL[/bold red] Failed to connect: {e}")
        raise typer.Exit(code=1)


@app.command()
def etl(
    source: str = typer.Option(..., "--source", "-s", help="Source URI for data extraction"),
    sink: str = typer.Option(..., "--sink", "-k", help="Sink URI for data loading"),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip data validation step"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing"),
    output_format: str = typer.Option("table", "--output-format", "-f", help="Output: table|json"),
) -> None:
    """Run an ETL pipeline: extract -> validate -> load."""
    if dry_run:
        console.print("[bold yellow]DRY RUN — no data will be processed[/bold yellow]")
        console.print(f"Source: {source}")
        console.print(f"Sink: {sink}")
        console.print(f"Validation: {'disabled' if no_validate else 'enabled'}")
        return

    console.print("[bold blue]Starting ETL pipeline...[/bold blue]")

    async def run_etl() -> None:
        from src.application.dtos.data_engineering import ETLRunRequest
        from src.application.use_cases.run_etl import RunETLUseCase
        from src.infrastructure.data.local_file_sink import LocalFileSink
        from src.infrastructure.data.local_file_source import LocalFileSource
        from src.infrastructure.data.pydantic_data_validator import PydanticDataValidator
        from src.infrastructure.events.in_memory_event_bus import InMemoryEventBus

        use_case = RunETLUseCase(
            source=LocalFileSource(),
            sink=LocalFileSink(),
            validator=PydanticDataValidator(),
            event_bus=InMemoryEventBus(),
        )
        request = ETLRunRequest(
            source_uri=source,
            sink_uri=sink,
            run_validation=not no_validate,
        )
        result = await use_case.execute(request)

        if output_format == "json":
            console.print(
                json_lib.dumps(
                    {
                        "status": result.status,
                        "records_extracted": result.records_extracted,
                        "records_loaded": result.records_loaded,
                        "duration_seconds": result.duration_seconds,
                        "errors": result.errors,
                    },
                    indent=2,
                )
            )
        else:
            table = Table(title="ETL Result")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Status", result.status)
            table.add_row("Extracted", str(result.records_extracted))
            table.add_row("Loaded", str(result.records_loaded))
            table.add_row("Duration (s)", f"{result.duration_seconds:.2f}")
            console.print(table)

            if result.errors:
                console.print("\n[bold red]Errors:[/bold red]")
                for error in result.errors:
                    console.print(f"  - {error}")

    asyncio.run(run_etl())


@app.command()
def validate(
    source: str = typer.Option(..., "--source", "-s", help="Source URI for data validation"),
    output_format: str = typer.Option("table", "--output-format", "-f", help="Output: table|json"),
) -> None:
    """Validate a dataset and report quality."""
    console.print("[bold blue]Validating dataset...[/bold blue]")

    async def run_validate() -> None:
        from src.application.use_cases.validate_dataset import ValidateDatasetUseCase
        from src.infrastructure.data.local_file_source import LocalFileSource
        from src.infrastructure.data.pydantic_data_validator import PydanticDataValidator
        from src.infrastructure.events.in_memory_event_bus import InMemoryEventBus

        use_case = ValidateDatasetUseCase(
            source=LocalFileSource(),
            validator=PydanticDataValidator(),
            event_bus=InMemoryEventBus(),
        )
        result = await use_case.execute(source_uri=source)

        if output_format == "json":
            console.print(
                json_lib.dumps(
                    {
                        "is_valid": result.is_valid,
                        "total_records": result.total_records,
                        "valid_records": result.valid_records,
                        "invalid_records": result.invalid_records,
                        "errors": result.errors,
                    },
                    indent=2,
                )
            )
        else:
            table = Table(title="Validation Result")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Valid", str(result.is_valid))
            table.add_row("Total Records", str(result.total_records))
            table.add_row("Valid Records", str(result.valid_records))
            table.add_row("Invalid Records", str(result.invalid_records))
            console.print(table)

            if result.errors:
                console.print("\n[bold red]Errors:[/bold red]")
                for error in result.errors:
                    console.print(f"  - {error}")

    asyncio.run(run_validate())


@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Question to ask the RAG pipeline"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of documents to retrieve"),
    output_format: str = typer.Option("table", "--output-format", "-f", help="Output: table|json"),
) -> None:
    """Query the RAG pipeline from the command line."""
    console.print("[bold blue]Running RAG query...[/bold blue]")

    async def run_query() -> None:
        from src.application.use_cases.query_rag import QueryRAGUseCase
        from src.infrastructure.container import Container

        settings = get_settings()
        container = Container(settings=settings)

        use_case = QueryRAGUseCase(
            llm=container.llm_adapter,
            retriever=container.retriever_adapter,
        )
        result = await use_case.execute(query=question, top_k=top_k)

        if output_format == "json":
            console.print(
                json_lib.dumps(
                    {
                        "answer": result.answer,
                        "model": result.model,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"\n[bold green]Answer:[/bold green] {result.answer}\n")
            if result.model:
                console.print(f"[dim]Model: {result.model}[/dim]")

        await container.close()

    asyncio.run(run_query())


@app.command()
def cost(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to look back"),
    output_format: str = typer.Option("table", "--output-format", "-f", help="Output: table|json"),
    api_url: str = typer.Option(
        "http://localhost:8000",
        "--api-url",
        "-u",
        help="Base URL of the API",
    ),
) -> None:
    """Show LLM cost summary from the API."""
    try:
        response = httpx.get(
            f"{api_url}/api/v1/analytics/costs",
            params={"days": days},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        if output_format == "json":
            console.print(json_lib.dumps(data, indent=2))
        else:
            total = data["total_cost_usd"]
            console.print(f"\n[bold]Total Cost (last {days} days):[/bold] ${total:.4f}\n")
            if data.get("breakdown"):
                table = Table(title="Cost Breakdown")
                table.add_column("Group", style="cyan")
                table.add_column("Calls", style="magenta")
                table.add_column("Tokens", style="yellow")
                table.add_column("Cost (USD)", style="green")
                for item in data["breakdown"]:
                    table.add_row(
                        item["group"],
                        str(item["call_count"]),
                        str(item["total_tokens"]),
                        f"${item['total_cost']:.4f}",
                    )
                console.print(table)
            else:
                console.print("[dim]No usage data found.[/dim]")

    except httpx.HTTPError as e:
        console.print(f"[bold red]FAIL[/bold red] Failed to fetch costs: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
