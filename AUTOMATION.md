# Automatización & Scripting

## Python como Pegamento del Sistema

Python conecta sistemas, automatiza procesos repetitivos y construye herramientas internas.

---

## CLI Tools

### Typer (Recomendado)

CLI moderno con type hints. Auto-genera help y validación.

```python
# src/interfaces/cli/main.py
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="genai", help="GenAI system CLI")
console = Console()

@app.command()
def index(
    source: str = typer.Argument(..., help="Path or URL to index"),
    chunk_size: int = typer.Option(1000, help="Chunk size in characters"),
    dry_run: bool = typer.Option(False, help="Preview without indexing"),
):
    """Index documents into the vector store."""
    documents = load_documents(source)
    chunks = chunk_documents(documents, max_length=chunk_size)

    if dry_run:
        table = Table(title=f"Preview: {len(chunks)} chunks")
        table.add_column("ID")
        table.add_column("Length")
        table.add_column("Preview")
        for i, chunk in enumerate(chunks[:10]):
            table.add_row(str(i), str(len(chunk.content)), chunk.content[:80] + "...")
        console.print(table)
        return

    vector_store.upsert(chunks)
    console.print(f"[green]Indexed {len(chunks)} chunks[/green]")

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    model: str = typer.Option("gpt-4o", help="LLM model to use"),
    top_k: int = typer.Option(5, help="Number of documents to retrieve"),
):
    """Query the RAG system."""
    import asyncio
    result = asyncio.run(rag_pipeline.query(question, model=model, top_k=top_k))
    console.print(f"\n[bold]Answer:[/bold] {result.answer}")
    console.print(f"\n[dim]Sources: {', '.join(result.sources)}[/dim]")
    console.print(f"[dim]Confidence: {result.confidence:.2f}[/dim]")

@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="Path to evaluation dataset"),
    output: str = typer.Option("results.json", help="Output file"),
):
    """Run evaluation on a dataset."""
    import asyncio
    results = asyncio.run(run_evaluation(dataset))
    save_results(results, output)
    console.print(f"[green]Evaluation complete. Results saved to {output}[/green]")

if __name__ == "__main__":
    app()
```

### Click

Alternativa madura. Más explícito, más control.

```python
import click

@click.group()
def cli():
    """GenAI system CLI."""
    pass

@cli.command()
@click.argument("source")
@click.option("--chunk-size", default=1000)
def index(source: str, chunk_size: int):
    """Index documents."""
    # ...

@cli.command()
@click.argument("question")
@click.option("--model", default="gpt-4o")
def query(question: str, model: str):
    """Query the system."""
    # ...
```

---

## Web Scraping & Browser Automation

### Playwright (Recomendado)

Automatización de browsers moderna. Async nativo.

```python
from playwright.async_api import async_playwright

async def scrape_documentation(url: str) -> list[Document]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        # Extraer contenido
        content = await page.evaluate("""
            () => {
                const article = document.querySelector('article') || document.body;
                return article.innerText;
            }
        """)

        # Screenshot para debugging
        await page.screenshot(path="debug.png")
        await browser.close()

    return [Document(content=content, metadata={"source": url})]

async def scrape_multiple(urls: list[str]) -> list[Document]:
    """Scraping paralelo de múltiples URLs."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [scrape_page(browser, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()
    return [r for r in results if isinstance(r, Document)]
```

### Selenium

Alternativa más antigua. Usar cuando Playwright no está disponible.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get(url)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
content = driver.find_element(By.TAG_NAME, "article").text
driver.quit()
```

---

## Integraciones entre Sistemas

### HTTP APIs

```python
import httpx

class ExternalServiceClient:
    """Cliente para integración con servicio externo."""

    def __init__(self, base_url: str, api_key: str):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def fetch_data(self, endpoint: str, params: dict | None = None) -> dict:
        response = await self._client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    async def push_data(self, endpoint: str, data: dict) -> dict:
        response = await self._client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()
```

### Webhooks

```python
from fastapi import FastAPI, Request, Header, HTTPException
import hmac
import hashlib

@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(...),
):
    body = await request.body()

    # Verificar signature
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(x_hub_signature_256, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event")
    await process_github_event(event_type, payload)
    return {"status": "ok"}
```

---

## Scheduled Tasks & Cron Jobs

### Con APScheduler

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", hours=1)
async def sync_documents():
    """Sincronizar documentos nuevos cada hora."""
    new_docs = await source.fetch_new()
    if new_docs:
        await vector_store.upsert(new_docs)
        logger.info("documents_synced", count=len(new_docs))

@scheduler.scheduled_job("cron", hour=2, minute=0)
async def daily_evaluation():
    """Evaluación diaria del sistema RAG."""
    results = await run_evaluation(EVAL_DATASET)
    if results.average_score < QUALITY_THRESHOLD:
        await alert_team("RAG quality degraded", results)

scheduler.start()
```

---

## Bots Internos

```python
# Bot de Slack para consultas al sistema GenAI
from slack_bolt.async_app import AsyncApp

slack_app = AsyncApp(token=SLACK_BOT_TOKEN)

@slack_app.message("ask:")
async def handle_question(message, say):
    question = message["text"].replace("ask:", "").strip()
    result = await rag_pipeline.query(question)
    await say(
        f"*Answer:* {result.answer}\n"
        f"_Sources: {', '.join(result.sources)}_\n"
        f"_Confidence: {result.confidence:.0%}_"
    )
```

---

## Scripts de Proyecto

```
scripts/
├── index_documents.py    # Indexar documentos en vector store
├── run_evaluation.py     # Ejecutar evaluación de calidad
├── migrate_data.py       # Migración de datos
├── sync_embeddings.py    # Re-generar embeddings
├── benchmark.py          # Benchmark de modelos
└── seed_data.py          # Datos de prueba para desarrollo
```

---

## Reglas

1. **Typer para CLIs** — auto-help, validación, rich output
2. **Playwright sobre Selenium** — async, más rápido, mejor API
3. **httpx sobre requests** — async nativo, HTTP/2
4. **Idempotencia** — ejecutar un script dos veces no rompe nada
5. **Dry-run siempre** — opción para previsualizar sin ejecutar
6. **Logging, no prints** — structlog para scripts también

Ver también: [TOOLS.md](TOOLS.md), [DATA_ENGINEERING.md](DATA_ENGINEERING.md)
