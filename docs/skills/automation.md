---
name: Automation & Scripting
description: Best practices for building CLI tools, web scrapers, and automation scripts.
---

# Skill: Automation & Scripting

## Description

Automation and scripting form the operational backbone of GenAI systems, handling data ingestion,
periodic maintenance, external integrations, and developer tooling. This skill covers building
robust CLI tools, async web scrapers, scheduled tasks, and HTTP automation with production-grade
error handling, idempotency, and observability.

## Executive Summary

**Critical automation rules:**
- ALL scripts MUST be idempotent — safe to re-run without side effects (use upserts, check-before-write)
- `--dry-run` flag MANDATORY for destructive operations — preview changes without applying
- Graceful shutdown required (SIGTERM/SIGINT handlers) — clean up resources before exit
- Meaningful exit codes documented (0=success, 1=error, 2=usage, 3+=domain-specific) — callers rely on these
- Load ALL config from environment variables — never hardcode URLs, credentials, or paths

**Read full skill when:** Building CLI tools, implementing web scrapers, scheduling periodic tasks, or automating data pipelines with retry logic and error handling.

---

## Versiones y Robustez

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| playwright | >= 1.40.0 | Scraper async estable |
| typer | >= 0.9.0 | CLI moderno |
| tenacity | >= 8.2.0 | Retries configurables |

### Dry Run Implementation

```python
import typer

app = typer.Typer()

@app.command()
def delete_records(dry_run: bool = typer.Option(False, "--dry-run")):
    if dry_run:
        print("SIMULACIÓN: Se borrarían 50 registros.")
        return
    # logic...
```

---

## Deep Dive

## Core Concepts

1. **Idempotent Execution** — Every script must produce the same result whether run once or ten
   times. Use upserts instead of inserts, check-before-write patterns, and state tracking to
   guarantee safe re-execution after partial failures.

2. **Dry-Run by Default** — Destructive operations (delete, overwrite, migrate) must support a
   `--dry-run` flag that previews changes without applying them. This is non-negotiable for
   production scripts.

3. **Graceful Shutdown** — Scripts must handle SIGTERM and SIGINT signals to clean up resources
   (close browsers, flush buffers, release locks) before exiting. This is critical for
   containerized and orchestrated environments.

4. **Structured Exit Codes** — Use meaningful exit codes: 0 for success, 1 for general errors,
   2 for usage errors, 3+ for domain-specific failures. Callers (CI, cron, orchestrators) rely
   on exit codes to determine next steps.

5. **Configuration from Environment** — Never hardcode URLs, credentials, or environment-specific
   values. Load from environment variables with sensible defaults, using Pydantic Settings or
   similar for validation.

6. **Observability in Scripts** — Even short-lived scripts need structured logging with
   correlation IDs, execution duration, and outcome status. This enables debugging when scripts
   run unattended in production.

## External Resources

### ⚡ CLI Frameworks
- [Typer Documentation](https://typer.tiangolo.com/)
  *Best for*: Modern CLI apps with type hints, auto-completion, and automatic help generation.
- [Click Documentation](https://click.palletsprojects.com/)
  *Best for*: Complex CLI tools requiring granular control over argument parsing and plugin systems.
- [Rich Library](https://rich.readthedocs.io/)
  *Best for*: Beautiful terminal output — tables, progress bars, syntax highlighting, tracebacks.
- [Real Python — Building CLI Apps with Typer](https://realpython.com/python-typer-cli/)
  *Best for*: Tutorial-level walkthrough of Typer patterns and best practices.

### 🌐 Web Scraping & HTTP
- [Playwright for Python](https://playwright.dev/python/)
  *Best for*: Async browser automation, dynamic content scraping, and E2E testing.
- [httpx Documentation](https://www.python-httpx.org/)
  *Best for*: Modern async HTTP client with HTTP/2 support, timeouts, and retry integration.
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
  *Best for*: Lightweight parsing of static HTML when a full browser is unnecessary.

### ⏰ Scheduling & Orchestration
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
  *Best for*: In-process task scheduling with cron-like triggers and persistent job stores.
- [Invoke Documentation](https://www.pyinvoke.org/)
  *Best for*: Task runner for build/deploy scripts, alternative to Makefiles.
- [Fabric Documentation](https://www.fabfile.org/)
  *Best for*: Remote server execution over SSH for deployment automation.

### 📖 General
- [Tenacity — Retry Library](https://tenacity.readthedocs.io/)
  *Best for*: Configurable retry logic with exponential backoff, jitter, and custom stop conditions.
- [structlog Documentation](https://www.structlog.org/)
  *Best for*: Structured, contextual logging for scripts and services.

## Instructions for the Agent

1. **Always implement `--dry-run`** for any CLI command that modifies state (files, databases,
   external services). The dry-run path must execute the same validation logic as the real path.

2. **Use structured logging with structlog** in all scripts. Include execution context: script
   name, run ID, start time, arguments. Log the outcome (success/failure/skip) at the end.

3. **Handle signals explicitly.** Register handlers for SIGTERM and SIGINT that set a shutdown
   flag. Check this flag in loops and long-running operations to exit cleanly.

4. **Return meaningful exit codes.** Map domain outcomes to specific codes. Document exit codes
   in the CLI help text. Never swallow exceptions silently — propagate them as non-zero exits.

5. **Load all configuration from environment variables** using Pydantic `BaseSettings` or
   equivalent. Provide CLI overrides via Typer options. Never hardcode URLs, paths, or credentials.

6. **Enforce rate limiting and backoff** in all HTTP and scraping operations. Use `tenacity` for
   retries with exponential backoff and jitter. Respect `robots.txt` and rate headers.

7. **Make scripts idempotent.** Use upserts, check-before-write, and state files to ensure safe
   re-execution. Document idempotency guarantees in docstrings.

8. **Test CLI commands** with `typer.testing.CliRunner` and mock external dependencies. Verify
   exit codes, output messages, and dry-run behavior in unit tests.

## Code Examples

### Robust CLI with Typer, Rich, Error Handling, and Exit Codes

```python
"""CLI tool for syncing data from an external API."""

import signal
import sys
from enum import IntEnum

import structlog
import typer
from rich.console import Console
from rich.progress import Progress

logger = structlog.get_logger()
console = Console(stderr=True)
app = typer.Typer(help="Data sync CLI for GenAI pipeline.")

_shutdown_requested = False


class ExitCode(IntEnum):
    SUCCESS = 0
    GENERAL_ERROR = 1
    USAGE_ERROR = 2
    CONNECTION_ERROR = 3
    PARTIAL_FAILURE = 4


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.warning("shutdown_requested", signal=signal.Signals(signum).name)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


@app.command()
def sync(
    source_url: str = typer.Option(..., envvar="SYNC_SOURCE_URL", help="Source API URL"),
    batch_size: int = typer.Option(100, help="Records per batch"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without applying"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Sync records from external source into local store."""
    log = logger.bind(source_url=source_url, dry_run=dry_run)
    log.info("sync_started", batch_size=batch_size)

    if dry_run:
        console.print("[yellow]DRY RUN — no changes will be applied[/yellow]")

    try:
        records = _fetch_records(source_url, batch_size)
        processed = 0

        with Progress(console=console) as progress:
            task = progress.add_task("Syncing...", total=len(records))

            for record in records:
                if _shutdown_requested:
                    log.warning("sync_interrupted", processed=processed)
                    raise typer.Exit(code=ExitCode.PARTIAL_FAILURE)

                if not dry_run:
                    _upsert_record(record)
                processed += 1
                progress.advance(task)

        log.info("sync_completed", processed=processed)
        console.print(f"[green]Synced {processed} records[/green]")

    except ConnectionError as exc:
        log.error("connection_failed", error=str(exc))
        console.print(f"[red]Connection error: {exc}[/red]")
        raise typer.Exit(code=ExitCode.CONNECTION_ERROR) from exc
    except Exception as exc:
        log.error("sync_failed", error=str(exc))
        raise typer.Exit(code=ExitCode.GENERAL_ERROR) from exc


def _fetch_records(url: str, batch_size: int) -> list[dict]:
    """Fetch records from source API."""
    ...


def _upsert_record(record: dict) -> None:
    """Idempotent upsert of a single record."""
    ...


if __name__ == "__main__":
    app()
```

### Async Web Scraper with Playwright, Retry, and Rate Limiting

```python
"""Async scraper with retry logic and rate limiting."""

import asyncio

import structlog
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = structlog.get_logger()

# Rate limiter: max 5 concurrent requests
_semaphore = asyncio.Semaphore(5)
_request_delay_seconds = 1.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((PlaywrightTimeout, ConnectionError)),
    before_sleep=lambda retry_state: logger.warning(
        "scrape_retry",
        attempt=retry_state.attempt_number,
        url=retry_state.args[1] if len(retry_state.args) > 1 else "unknown",
    ),
)
async def _scrape_page(page, url: str) -> dict:
    """Scrape a single page with retry and rate limiting."""
    async with _semaphore:
        await asyncio.sleep(_request_delay_seconds)
        log = logger.bind(url=url)

        await page.goto(url, timeout=30_000, wait_until="networkidle")
        title = await page.title()
        content = await page.inner_text("body")

        log.info("page_scraped", title=title, content_length=len(content))
        return {"url": url, "title": title, "content": content}


async def scrape_urls(urls: list[str]) -> list[dict]:
    """Scrape multiple URLs concurrently with managed browser lifecycle."""
    results: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        try:
            context = await browser.new_context(
                user_agent="GenAI-Bot/1.0 (contact@example.com)",
            )
            page = await context.new_page()

            for url in urls:
                try:
                    result = await _scrape_page(page, url)
                    results.append(result)
                except Exception:
                    logger.error("scrape_failed_permanently", url=url)
        finally:
            await browser.close()

    logger.info("scrape_batch_completed", total=len(urls), success=len(results))
    return results
```

### Scheduled Task with APScheduler and Graceful Shutdown

```python
"""Background scheduler with graceful shutdown for periodic tasks."""

import signal
import asyncio
from datetime import datetime

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = structlog.get_logger()


async def cleanup_stale_sessions() -> None:
    """Periodic task: remove sessions older than 24 hours."""
    log = logger.bind(task="cleanup_stale_sessions", run_at=datetime.utcnow().isoformat())
    log.info("task_started")

    try:
        deleted_count = await _delete_stale_sessions(max_age_hours=24)
        log.info("task_completed", deleted=deleted_count)
    except Exception as exc:
        log.error("task_failed", error=str(exc))


async def sync_embeddings_index() -> None:
    """Periodic task: rebuild vector index from updated documents."""
    log = logger.bind(task="sync_embeddings_index", run_at=datetime.utcnow().isoformat())
    log.info("task_started")

    try:
        indexed = await _rebuild_index()
        log.info("task_completed", documents_indexed=indexed)
    except Exception as exc:
        log.error("task_failed", error=str(exc))


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the scheduler with all periodic tasks."""
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        cleanup_stale_sessions,
        trigger=CronTrigger(hour=3, minute=0),
        id="cleanup_sessions",
        name="Cleanup stale sessions",
        replace_existing=True,
    )
    scheduler.add_job(
        sync_embeddings_index,
        trigger=CronTrigger(hour="*/6"),
        id="sync_embeddings",
        name="Sync embeddings index",
        replace_existing=True,
    )
    return scheduler


async def run_scheduler() -> None:
    """Run the scheduler with graceful shutdown on signals."""
    scheduler = create_scheduler()
    shutdown_event = asyncio.Event()

    def _request_shutdown(signum: int, _frame: object) -> None:
        logger.info("shutdown_signal_received", signal=signal.Signals(signum).name)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    scheduler.start()
    logger.info("scheduler_started", jobs=len(scheduler.get_jobs()))

    await shutdown_event.wait()

    scheduler.shutdown(wait=True)
    logger.info("scheduler_stopped_gracefully")


async def _delete_stale_sessions(max_age_hours: int) -> int:
    """Delete stale sessions — placeholder."""
    ...


async def _rebuild_index() -> int:
    """Rebuild embeddings index — placeholder."""
    ...
```

### HTTP Automation with httpx, Retry, and Structured Responses

```python
"""HTTP automation client with retry, timeout, and structured logging."""

from dataclasses import dataclass

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = structlog.get_logger()

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
MAX_RETRIES = 3


@dataclass(frozen=True)
class ApiResponse:
    """Structured response from an external API call."""

    status_code: int
    data: dict | list | None
    elapsed_ms: float
    success: bool


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
)
async def api_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: dict | None = None,
    params: dict | None = None,
) -> ApiResponse:
    """Execute an HTTP request with retry and structured logging."""
    log = logger.bind(method=method, url=url)
    log.info("http_request_started")

    response = await client.request(method, url, json=json, params=params)
    elapsed_ms = response.elapsed.total_seconds() * 1000

    log.info(
        "http_request_completed",
        status_code=response.status_code,
        elapsed_ms=round(elapsed_ms, 2),
    )

    return ApiResponse(
        status_code=response.status_code,
        data=response.json() if response.is_success else None,
        elapsed_ms=elapsed_ms,
        success=response.is_success,
    )


async def sync_external_data(base_url: str, api_key: str) -> list[dict]:
    """Sync data from a paginated external API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    all_records: list[dict] = []

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    ) as client:
        page = 1
        while True:
            result = await api_request(client, "GET", "/records", params={"page": page})

            if not result.success or not result.data:
                break

            all_records.extend(result.data)
            logger.info("page_fetched", page=page, records=len(result.data))

            if len(result.data) < 100:
                break
            page += 1

    logger.info("sync_completed", total_records=len(all_records))
    return all_records
```

## Anti-Patterns to Avoid

### ❌ Blocking Synchronous Calls in Async Scripts

**Problem:** Mixing `requests` or `time.sleep()` inside async code blocks the event loop,
negating all concurrency benefits and causing timeouts in other coroutines.

**Example:**
```python
import requests  # synchronous!

async def fetch_data(url: str):
    response = requests.get(url)  # blocks the entire event loop
    return response.json()
```

**Solution:** Use `httpx.AsyncClient` or `aiohttp` for HTTP calls. Use `asyncio.sleep()` for
delays. If a sync library is unavoidable, run it in `asyncio.to_thread()`.

### ❌ No Dry-Run for Destructive Operations

**Problem:** Scripts that delete files, drop tables, or modify external state without a preview
mode lead to irreversible mistakes in production.

**Example:**
```python
@app.command()
def purge(days: int):
    db.execute(f"DELETE FROM logs WHERE age > {days}")  # no preview, no confirmation
```

**Solution:** Add `--dry-run` that runs the same query with `SELECT COUNT(*)` first. Log what
would be affected. Require explicit `--confirm` for production targets.

### ❌ Hardcoded URLs and Credentials

**Problem:** Embedding API endpoints, file paths, or secrets directly in code makes scripts
non-portable and creates security risks.

**Example:**
```python
API_URL = "https://prod.internal.company.com/api/v1"
API_KEY = "sk-abc123secret"
```

**Solution:** Use environment variables loaded via Pydantic `BaseSettings`. Provide CLI
overrides. Never commit credentials to version control.

### ❌ No Signal Handling in Long-Running Scripts

**Problem:** Scripts that ignore SIGTERM leave resources in an inconsistent state when
terminated by orchestrators (Kubernetes, systemd, CI runners).

**Example:**
```python
for item in huge_dataset:
    process(item)  # SIGTERM kills mid-iteration, data half-written
```

**Solution:** Register signal handlers that set a shutdown flag. Check the flag between
iterations. Flush pending work and release locks before exiting.

## Automation Checklist

### Script Reliability
- [ ] All scripts are idempotent — safe to re-run without side effects
- [ ] Retry logic with exponential backoff on transient failures
- [ ] Timeouts configured for all external calls
- [ ] Meaningful exit codes documented and tested
- [ ] Graceful shutdown on SIGTERM/SIGINT

### CLI Quality
- [ ] `--dry-run` flag on all destructive commands
- [ ] `--verbose` / `-v` flag for debug logging
- [ ] Auto-generated help text with examples
- [ ] Configuration from environment variables with CLI overrides
- [ ] Unit tests with `typer.testing.CliRunner`

### Scraping Ethics & Reliability
- [ ] Respect `robots.txt` and crawl-delay headers
- [ ] User-Agent identifies your bot and provides contact info
- [ ] Rate limiting with semaphores and inter-request delays
- [ ] Retry with backoff on transient errors (timeouts, 429s, 503s)
- [ ] Data extraction validated against expected schema

### Scheduling
- [ ] Jobs registered with unique IDs and `replace_existing=True`
- [ ] Scheduler handles graceful shutdown on signals
- [ ] Failed jobs logged with full context (task name, error, duration)
- [ ] No overlapping executions for long-running jobs (use locks or `max_instances=1`)
- [ ] Job execution history persisted for debugging

## Additional References

- [Typer — CLI Tutorial](https://typer.tiangolo.com/tutorial/)
- [Playwright Python — Getting Started](https://playwright.dev/python/docs/intro)
- [httpx — Advanced Usage](https://www.python-httpx.org/advanced/)
- [Tenacity — Retry Patterns](https://tenacity.readthedocs.io/en/latest/)
- [structlog — Getting Started](https://www.structlog.org/en/stable/getting-started.html)
