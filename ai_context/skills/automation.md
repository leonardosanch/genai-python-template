---
name: Automation & Scripting
description: Best practices for building CLI tools, web scrappers, and automation scripts.
---

# Automation & Scripting

## Core Philosophy
Python serves as the "glue" language for GenAI systems, connecting disparate components, automating workflows, and enabling rapid prototyping.

## Key Technologies

### CLI Tools
- **Typer**: The standard for building modern CLIs. Features automatic help generation, type validation, and rich output.
    - *Best for*: Administrative tools, data ingestion scripts, dev utilities.
- **Click**: Mature alternative when granular control is needed.

### Web Scraping & Browser Automation
- **Playwright**: Modern, async-native browser automation. Superior to Selenium for handling dynamic content and SPAs.
    - *Best for*: Data collection, end-to-end testing, UI automation.
- **BeautifulSoup**: Lightweight parsing for static HTML.

### Task Scheduling
- **APScheduler**: In-process scheduling for background tasks.
    - *Best for*: Periodic sync jobs, cleanup tasks, simple cron replacements.

## Implementation Patterns

### 1. The Robust CLI Pattern
Always use `Typer` with structured logging and error handling.

```python
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def sync(
    dry_run: bool = typer.Option(False, help="Preview changes only"),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
):
    """Sync data from external source."""
    if dry_run:
        console.print("[yellow]Dry run active[/yellow]")
    # Logic here
```

### 2. Async Scraping
Use `async_playwright` to handle concurrent scraping efficiently.

```python
from playwright.async_api import async_playwright

async def scrape(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        content = await page.content()
        await browser.close()
        return content
```

## Best Practices
1.  **Idempotency**: Scripts should be safe to run multiple times without side effects (upserts > inserts).
2.  **Dry-Runs**: Always implement a `--dry-run` flag for destructive operations.
3.  **Structured Logging**: Use `structlog` even in scripts for better observability.
4.  **Signal Handling**: Handle SIGTERM/SIGINT gracefully to clean up resources.

## External Resources
- [Typer Documentation](https://typer.tiangolo.com/)
- [Playwright Python](https://playwright.dev/python/)
- [APScheduler User Guide](https://apscheduler.readthedocs.io/)
