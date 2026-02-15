# Skill: Performance Profiling & Optimization

## Description
This skill covers application performance analysis, profiling, load testing, and optimization strategies for Python backend and GenAI systems. Use this when diagnosing bottlenecks, memory leaks, slow queries, async performance issues, or planning capacity.

## Executive Summary

**Critical performance rules:**
- **Profile before optimizing** — Never guess; use `py-spy`, `scalene`, or `cProfile` to identify the real bottleneck
- **Async I/O is mandatory for LLM calls** — A single blocking call can stall the entire event loop
- **Monitor memory in long-running services** — Use `tracemalloc` or `memray` to detect leaks in workers and agents
- **Load test with realistic LLM latency** — Mock LLM responses with realistic delays (500ms-5s), not instant returns
- **PostgreSQL: EXPLAIN ANALYZE before indexing** — Adding indexes without analysis can hurt write performance
- **Token throughput > raw latency** — For GenAI, optimize tokens/second and batch efficiency, not just p99

**Read full skill when:** Diagnosing slow endpoints, memory growth, event loop blocking, database bottlenecks, or planning load tests for GenAI APIs.

---

## Versiones y Herramientas

| Herramienta | Versión Mínima | Tipo | Notas |
|-------------|----------------|------|-------|
| py-spy | >= 0.3.14 | CPU profiler (sampling) | No requiere instrumentación, attach a proceso en ejecución |
| scalene | >= 1.5.0 | CPU + Memory + GPU | Perfil combinado, ideal para ML/GenAI |
| memray | >= 1.10.0 | Memory profiler | Flamegraphs de memoria, detecta leaks nativos |
| tracemalloc | stdlib | Memory tracker | Incluido en Python, sin dependencias |
| locust | >= 2.20.0 | Load testing | Python-native, distribuido, ideal para APIs |
| k6 | >= 0.47.0 | Load testing | Go-based, alto rendimiento, scripting en JS |
| pytest-benchmark | >= 4.0.0 | Microbenchmarks | Integrado con pytest |
| line_profiler | >= 4.1.0 | Line-by-line CPU | Decorador `@profile` por función |
| pyinstrument | >= 4.6.0 | Call stack profiler | Output legible, ideal para web requests |

---

## Core Concepts

1. **Profile First, Optimize Second**: Medir antes de cambiar. La intuición falla ~70% de las veces.
2. **Amdahl's Law**: Optimizar el 5% del código que consume el 95% del tiempo.
3. **Latency vs Throughput**: En GenAI, throughput (requests/sec) importa más que latencia individual.
4. **Memory vs CPU tradeoff**: Caching reduce CPU pero aumenta memoria. Medir ambos.
5. **Async is not parallel**: `asyncio` es concurrencia cooperativa, no paralelismo. CPU-bound necesita `ProcessPoolExecutor`.

---

## Decision Trees

### Decision Tree 1: Qué profiler usar

```
¿Qué tipo de problema tienes?
├── Endpoint lento (no sé dónde)
│   └── pyinstrument (call stack legible, bajo overhead)
│       └── uv run pyinstrument -r html -o profile.html my_script.py
├── CPU alto en producción (no puedo reiniciar)
│   └── py-spy (attach sin restart, sampling)
│       └── py-spy record -o profile.svg --pid <PID>
├── Función específica lenta (ya sé cuál)
│   └── line_profiler (línea por línea)
│       └── @profile decorator + kernprof -lv script.py
├── Memory leak / crecimiento
│   ├── Python objects → tracemalloc (stdlib, sin deps)
│   └── Native + Python → memray (flamegraphs, C extensions)
├── GPU + CPU + Memory combinado
│   └── scalene (todo en uno, ideal para ML)
│       └── uv run scalene my_script.py
└── Microbenchmark (comparar implementaciones)
    └── pytest-benchmark
        └── def test_perf(benchmark): benchmark(my_function)
```

### Decision Tree 2: Load testing tool

```
¿Qué necesitas testear?
├── API Python (FastAPI/Django) con lógica custom
│   └── Locust (Python-native, custom user behavior)
│       ├── Simula usuarios concurrentes
│       ├── Soporte para WebSockets y SSE
│       └── Dashboard web incluido
├── API de alto throughput (> 10K rps)
│   └── k6 (Go-based, bajo overhead)
│       ├── Scripting en JavaScript
│       └── Integración con Grafana Cloud
├── Test simple de endpoint
│   └── wrk o hey (CLI one-liner)
│       └── wrk -t4 -c100 -d30s http://localhost:8000/api/v1/health
└── LLM API con streaming
    └── Locust + custom SSE client
        └── Ver ejemplo en sección de código
```

---

## External Resources

### Profiling
- **py-spy**: [github.com/benfred/py-spy](https://github.com/benfred/py-spy)
    - *Best for*: Production profiling sin reinicio, flamegraphs SVG
- **Scalene**: [github.com/plasma-umass/scalene](https://github.com/plasma-umass/scalene)
    - *Best for*: CPU + memory + GPU profiling combinado para ML/AI
- **memray**: [github.com/bloomberg/memray](https://github.com/bloomberg/memray)
    - *Best for*: Memory profiling con soporte para C extensions y flamegraphs
- **pyinstrument**: [github.com/joerick/pyinstrument](https://github.com/joerick/pyinstrument)
    - *Best for*: Call stack profiling legible para web requests

### Load Testing
- **Locust**: [locust.io](https://locust.io/)
    - *Best for*: Load testing Python-native con scenarios programables
- **k6**: [k6.io](https://k6.io/)
    - *Best for*: Load testing de alto rendimiento con integración Grafana
- **wrk**: [github.com/wg/wrk](https://github.com/wg/wrk)
    - *Best for*: Benchmarks HTTP rápidos desde CLI

### Database Performance
- **pgMustard**: [pgmustard.com](https://www.pgmustard.com/)
    - *Best for*: Visualización de EXPLAIN ANALYZE
- **pg_stat_statements**: [postgresql.org/docs/current/pgstatstatements.html](https://www.postgresql.org/docs/current/pgstatstatements.html)
    - *Best for*: Tracking de queries lentas en producción

### Python Performance
- **Python Performance Tips**: [wiki.python.org/moin/PythonSpeed/PerformanceTips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
    - *Best for*: Referencia oficial de optimización Python
- **High Performance Python (Book)**: Micha Gorelick & Ian Ozsvald
    - *Best for*: Deep dive en profiling, Cython, concurrencia

---

## Code Examples

### Example 1: Profiling FastAPI endpoints con pyinstrument

```python
# src/infrastructure/middleware/profiling.py
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Middleware that profiles requests when ?profile=1 is present.

    ONLY enable in dev/staging. Never in production.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.query_params.get("profile") != "1":
            return await call_next(request)

        profiler = Profiler(interval=0.001, async_mode="enabled")
        profiler.start()

        response = await call_next(request)

        profiler.stop()

        return HTMLResponse(profiler.output_html())


# Usage in main.py (dev only):
# if settings.environment == "development":
#     app.add_middleware(ProfilingMiddleware)
```

### Example 2: Memory leak detection con tracemalloc

```python
# src/infrastructure/diagnostics/memory_tracker.py
import linecache
import tracemalloc
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    label: str
    current_mb: float
    peak_mb: float
    top_allocations: list[dict]


class MemoryTracker:
    """Track memory allocations to detect leaks in long-running services."""

    def __init__(self, top_n: int = 10):
        self._top_n = top_n
        self._snapshots: list[tracemalloc.Snapshot] = []

    def start(self) -> None:
        tracemalloc.start(25)  # 25 frames deep for accurate tracebacks

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        snapshot = tracemalloc.take_snapshot()
        snapshot = snapshot.filter_traces([
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
            tracemalloc.Filter(False, tracemalloc.__file__),
        ])
        self._snapshots.append(snapshot)

        current, peak = tracemalloc.get_traced_memory()
        top_stats = snapshot.statistics("lineno")

        return MemorySnapshot(
            label=label,
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            top_allocations=[
                {
                    "file": str(stat.traceback),
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
                for stat in top_stats[: self._top_n]
            ],
        )

    def compare_snapshots(self, label_before: str = "before", label_after: str = "after") -> list[dict]:
        """Compare last two snapshots to find memory growth."""
        if len(self._snapshots) < 2:
            raise ValueError("Need at least 2 snapshots to compare")

        old = self._snapshots[-2]
        new = self._snapshots[-1]
        top_diffs = new.compare_to(old, "lineno")

        return [
            {
                "file": str(diff.traceback),
                "size_diff_mb": diff.size_diff / 1024 / 1024,
                "count_diff": diff.count_diff,
            }
            for diff in top_diffs[: self._top_n]
            if diff.size_diff > 0
        ]

    def stop(self) -> None:
        tracemalloc.stop()


# Usage in agent/worker lifecycle:
# tracker = MemoryTracker()
# tracker.start()
# tracker.take_snapshot("before_batch")
# ... process batch ...
# tracker.take_snapshot("after_batch")
# leaks = tracker.compare_snapshots()
# if leaks and leaks[0]["size_diff_mb"] > 50:
#     logger.warning("Memory growth detected", extra={"leaks": leaks})
```

### Example 3: Async event loop monitoring

```python
# src/infrastructure/diagnostics/event_loop_monitor.py
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class EventLoopMetrics:
    lag_ms: float
    max_lag_ms: float
    avg_lag_ms: float
    blocked_count: int  # Times lag exceeded threshold


class EventLoopMonitor:
    """Detect event loop blocking in async Python services.

    A blocked event loop means a coroutine is doing CPU-bound or
    synchronous I/O work, starving other tasks.
    """

    def __init__(
        self,
        check_interval: float = 0.5,
        block_threshold_ms: float = 100.0,
        history_size: int = 100,
    ):
        self._check_interval = check_interval
        self._block_threshold_ms = block_threshold_ms
        self._history: deque[float] = deque(maxlen=history_size)
        self._blocked_count = 0
        self._running = False

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            await asyncio.sleep(self._check_interval)
            elapsed = (time.monotonic() - t0) * 1000  # ms
            lag = elapsed - (self._check_interval * 1000)

            self._history.append(lag)

            if lag > self._block_threshold_ms:
                self._blocked_count += 1
                logger.warning(
                    "event_loop_blocked",
                    lag_ms=round(lag, 2),
                    threshold_ms=self._block_threshold_ms,
                    blocked_count=self._blocked_count,
                )

    def get_metrics(self) -> EventLoopMetrics:
        if not self._history:
            return EventLoopMetrics(0, 0, 0, 0)

        return EventLoopMetrics(
            lag_ms=round(self._history[-1], 2),
            max_lag_ms=round(max(self._history), 2),
            avg_lag_ms=round(sum(self._history) / len(self._history), 2),
            blocked_count=self._blocked_count,
        )

    def stop(self) -> None:
        self._running = False


# Usage:
# monitor = EventLoopMonitor(block_threshold_ms=50.0)
# await monitor.start()
# ... later ...
# metrics = monitor.get_metrics()
# if metrics.blocked_count > 0:
#     logger.error("Event loop was blocked", extra=asdict(metrics))
```

### Example 4: LLM-specific performance tracking

```python
# src/infrastructure/diagnostics/llm_perf_tracker.py
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class LLMCallMetrics:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    time_to_first_token_ms: float | None  # For streaming
    tokens_per_second: float
    cost_usd: float


@dataclass
class LLMPerformanceSummary:
    total_calls: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_tokens_per_second: float
    avg_ttft_ms: float | None  # Streaming only


class LLMPerfTracker:
    """Track LLM call performance for optimization decisions.

    Captures latency, throughput, TTFT, and cost per call.
    Use to identify: slow models, inefficient prompts, batching opportunities.
    """

    def __init__(self):
        self._calls: list[LLMCallMetrics] = []

    @asynccontextmanager
    async def track_call(self, model: str):
        """Context manager to track a single LLM call."""
        ctx = {"start_time": time.monotonic(), "first_token_time": None}

        def on_first_token():
            if ctx["first_token_time"] is None:
                ctx["first_token_time"] = time.monotonic()

        ctx["on_first_token"] = on_first_token

        yield ctx

        elapsed_ms = (time.monotonic() - ctx["start_time"]) * 1000
        ttft_ms = None
        if ctx["first_token_time"]:
            ttft_ms = (ctx["first_token_time"] - ctx["start_time"]) * 1000

        input_tokens = ctx.get("input_tokens", 0)
        output_tokens = ctx.get("output_tokens", 0)

        tps = (output_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        metrics = LLMCallMetrics(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(elapsed_ms, 2),
            time_to_first_token_ms=round(ttft_ms, 2) if ttft_ms else None,
            tokens_per_second=round(tps, 2),
            cost_usd=ctx.get("cost_usd", 0.0),
        )

        self._calls.append(metrics)

        logger.info(
            "llm_call_completed",
            model=metrics.model,
            latency_ms=metrics.latency_ms,
            tokens=metrics.output_tokens,
            tps=metrics.tokens_per_second,
            ttft_ms=metrics.time_to_first_token_ms,
        )

    def get_summary(self) -> LLMPerformanceSummary:
        if not self._calls:
            return LLMPerformanceSummary(0, 0, 0.0, 0.0, 0.0, 0.0, None)

        latencies = sorted(c.latency_ms for c in self._calls)
        p95_idx = int(len(latencies) * 0.95)
        ttft_values = [c.time_to_first_token_ms for c in self._calls if c.time_to_first_token_ms]

        return LLMPerformanceSummary(
            total_calls=len(self._calls),
            total_tokens=sum(c.input_tokens + c.output_tokens for c in self._calls),
            total_cost_usd=sum(c.cost_usd for c in self._calls),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2),
            p95_latency_ms=round(latencies[p95_idx], 2),
            avg_tokens_per_second=round(
                sum(c.tokens_per_second for c in self._calls) / len(self._calls), 2
            ),
            avg_ttft_ms=round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else None,
        )


# Usage:
# tracker = LLMPerfTracker()
# async with tracker.track_call("gpt-4o") as ctx:
#     response = await llm.generate(prompt)
#     ctx["input_tokens"] = response.usage.prompt_tokens
#     ctx["output_tokens"] = response.usage.completion_tokens
#     ctx["cost_usd"] = calculate_cost(response.usage)
#
# summary = tracker.get_summary()
# if summary.avg_ttft_ms and summary.avg_ttft_ms > 500:
#     logger.warning("TTFT exceeds target", avg_ttft_ms=summary.avg_ttft_ms)
```

### Example 5: Load testing GenAI API con Locust

```python
# tests/load/locustfile.py
import json
import time

from locust import HttpUser, between, task


class GenAIUser(HttpUser):
    """Load test for GenAI API endpoints.

    Run with: uv run locust -f tests/load/locustfile.py --host=http://localhost:8000
    """

    wait_time = between(1, 3)

    @task(3)
    def chat_completion(self):
        """Test synchronous chat completion endpoint."""
        payload = {
            "messages": [{"role": "user", "content": "Summarize clean architecture in 2 sentences."}],
            "model": "gpt-4o-mini",
            "max_tokens": 100,
        }
        with self.client.post(
            "/api/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "choices" not in data:
                    response.failure("Missing 'choices' in response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def chat_streaming(self):
        """Test SSE streaming endpoint with TTFT measurement."""
        payload = {
            "messages": [{"role": "user", "content": "What is SOLID?"}],
            "model": "gpt-4o-mini",
            "max_tokens": 200,
            "stream": True,
        }
        start = time.monotonic()
        first_token_time = None

        with self.client.post(
            "/api/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")
                return

            for line in response.iter_lines():
                if line and first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft_ms = (first_token_time - start) * 1000
                    if ttft_ms > 2000:
                        response.failure(f"TTFT too slow: {ttft_ms:.0f}ms")
                        return

            response.success()

    @task(1)
    def health_check(self):
        """Baseline health endpoint — should always be < 50ms."""
        self.client.get("/health")

    @task(1)
    def embedding(self):
        """Test embedding generation endpoint."""
        payload = {
            "texts": ["Clean architecture separates concerns into layers."],
            "model": "text-embedding-3-small",
        }
        self.client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
        )
```

### Example 6: PostgreSQL query performance analysis

```sql
-- scripts/sql/performance_analysis.sql

-- 1. Enable pg_stat_statements (run once as superuser)
-- CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 2. Top 20 slowest queries by total time
SELECT
    LEFT(query, 100) AS query_preview,
    calls,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    ROUND(mean_exec_time::numeric, 2) AS mean_time_ms,
    ROUND(max_exec_time::numeric, 2) AS max_time_ms,
    ROUND(stddev_exec_time::numeric, 2) AS stddev_ms,
    rows,
    ROUND(
        100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0), 2
    ) AS cache_hit_pct
FROM pg_stat_statements
WHERE mean_exec_time > 50  -- queries averaging > 50ms
ORDER BY total_exec_time DESC
LIMIT 20;

-- 3. Unused indexes (candidates for removal)
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelid NOT IN (
      SELECT conindid FROM pg_constraint WHERE contype IN ('p', 'u')
  )
ORDER BY pg_relation_size(indexrelid) DESC;

-- 4. Tables needing VACUUM (high dead tuple ratio)
SELECT
    schemaname,
    relname AS table_name,
    n_live_tup,
    n_dead_tup,
    ROUND(n_dead_tup::numeric / NULLIF(n_live_tup, 0) * 100, 2) AS dead_pct,
    last_autovacuum,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;

-- 5. Lock contention (current)
SELECT
    pg_class.relname,
    pg_locks.mode,
    pg_locks.granted,
    COUNT(*) AS lock_count,
    pg_locks.pid
FROM pg_locks
JOIN pg_class ON pg_locks.relation = pg_class.oid
WHERE NOT pg_locks.granted
GROUP BY pg_class.relname, pg_locks.mode, pg_locks.granted, pg_locks.pid
ORDER BY lock_count DESC;

-- 6. Connection pool saturation
SELECT
    state,
    COUNT(*) AS connections,
    ROUND(AVG(EXTRACT(EPOCH FROM (now() - state_change)))::numeric, 2) AS avg_duration_sec
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY state
ORDER BY connections DESC;

-- 7. Sequential scans on large tables (missing indexes)
SELECT
    schemaname,
    relname AS table_name,
    seq_scan,
    seq_tup_read,
    idx_scan,
    ROUND(
        100.0 * idx_scan / NULLIF(seq_scan + idx_scan, 0), 2
    ) AS idx_usage_pct,
    pg_size_pretty(pg_total_relation_size(relid)) AS table_size
FROM pg_stat_user_tables
WHERE seq_scan > 100
  AND pg_total_relation_size(relid) > 10 * 1024 * 1024  -- > 10MB
ORDER BY seq_tup_read DESC;
```

### Example 7: pytest-benchmark para comparar implementaciones

```python
# tests/benchmarks/test_embedding_perf.py
"""Benchmark embedding strategies to choose optimal batch size.

Run with: uv run pytest tests/benchmarks/ --benchmark-only --benchmark-sort=mean
"""
import pytest


def chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


# Simulates embedding call latency based on batch size
def embed_batch(texts: list[str], batch_size: int) -> list[list[float]]:
    """Simulate batched embedding with realistic overhead."""
    results = []
    for batch in chunk_list(texts, batch_size):
        # Simulated: real implementation calls LLM provider
        results.extend([[0.1] * 1536 for _ in batch])
    return results


@pytest.fixture
def sample_texts() -> list[str]:
    return [f"Document number {i} about clean architecture." for i in range(200)]


@pytest.mark.benchmark(group="embedding-batch-size")
def test_batch_size_10(benchmark, sample_texts):
    benchmark(embed_batch, sample_texts, batch_size=10)


@pytest.mark.benchmark(group="embedding-batch-size")
def test_batch_size_50(benchmark, sample_texts):
    benchmark(embed_batch, sample_texts, batch_size=50)


@pytest.mark.benchmark(group="embedding-batch-size")
def test_batch_size_100(benchmark, sample_texts):
    benchmark(embed_batch, sample_texts, batch_size=100)
```

---

## Performance Optimization Strategies

### Python Application

| Estrategia | Cuándo usar | Herramienta |
|---|---|---|
| **Connection pooling** | Siempre en producción | SQLAlchemy `pool_size`, `httpx.AsyncClient` |
| **Response caching** | Endpoints idempotentes, datos poco variables | Redis con TTL, `@lru_cache` para in-process |
| **Async batch processing** | Múltiples LLM calls independientes | `asyncio.gather` con `Semaphore` |
| **Lazy loading** | Modelos ML grandes, configuraciones pesadas | `functools.cached_property`, factory pattern |
| **Object pooling** | Objetos costosos de crear (DB connections, HTTP clients) | Context manager con pool |
| **Generator/streaming** | Datasets grandes, respuestas LLM | `yield`, `StreamingResponse`, SSE |
| **ProcessPoolExecutor** | CPU-bound (parsing, tokenization, NLP) | `asyncio.loop.run_in_executor` |

### LLM-Specific Optimization

| Estrategia | Impacto | Complejidad |
|---|---|---|
| **Prompt compression** | Reduce tokens 30-50%, menor latencia y costo | Media |
| **Semantic caching** | Evita calls repetidas, ahorra ~40% de costo | Media |
| **Model routing** | Modelo barato para queries simples, potente para complejas | Alta |
| **Batch embedding** | 5-10x throughput vs llamadas individuales | Baja |
| **Streaming responses** | TTFT < 500ms vs esperar respuesta completa | Baja |
| **Parallel tool calls** | Reduce latencia de agentes con tools independientes | Media |
| **Max tokens tuning** | Reducir `max_tokens` reduce latencia proporcional | Baja |

### Database Optimization

| Estrategia | Señal de que lo necesitas |
|---|---|
| **Add index** | Sequential scans en tablas > 10MB, `idx_usage_pct < 95%` |
| **Remove unused index** | `idx_scan = 0` en producción por > 30 días |
| **VACUUM ANALYZE** | `dead_pct > 20%` o `last_autovacuum` > 7 días |
| **Connection pool tuning** | `idle` connections > 50% del pool o `active` = `pool_size` |
| **Query rewrite** | `mean_time_ms > 100` con `cache_hit_pct < 90%` |
| **Partitioning** | Tables > 100M rows con queries filtradas por fecha/tenant |
| **Read replicas** | Read/write ratio > 10:1 |

---

## Anti-Patterns to Avoid

### Premature Optimization
**Problem**: Optimizar sin profiling, basándose en intuición
**Solution**: Siempre medir primero. `py-spy` o `pyinstrument` toma 5 minutos de setup.

### Sync LLM Calls in Async Context
**Problem**: `requests.post()` dentro de endpoint `async def` — bloquea el event loop
**Solution**: Usar `httpx.AsyncClient` o `aiohttp`. Nunca `requests` en código async.

### Unbounded Concurrency
**Problem**: `asyncio.gather(*[llm_call() for _ in range(1000)])` — rate limit + OOM
**Solution**: `asyncio.Semaphore(max_concurrent)` para limitar concurrencia.

### N+1 Queries in Agent Loops
**Problem**: Agente que hace una query por cada item en un loop
**Solution**: Batch queries con `WHERE id IN (...)` o prefetch relationships.

### Caching Without Invalidation
**Problem**: Cache stale indefinidamente, datos incorrectos
**Solution**: TTL explícito, invalidación por eventos, cache-aside pattern.

### Loading Full Dataset in Memory
**Problem**: `df = pd.read_csv("10gb_file.csv")` — OOM en producción
**Solution**: Streaming con generators, chunked reading, o Polars lazy frames.

---

## Performance Checklist

### Application
- [ ] Profiling ejecutado en endpoints críticos (pyinstrument/py-spy)
- [ ] Event loop blocking verificado (< 50ms lag threshold)
- [ ] Connection pools configurados (DB, HTTP clients, Redis)
- [ ] Async para todas las I/O operations (LLM, DB, HTTP)
- [ ] CPU-bound work offloaded a ProcessPoolExecutor

### LLM / GenAI
- [ ] TTFT tracked y < 500ms para streaming
- [ ] Token throughput medido (tokens/sec)
- [ ] Semantic caching evaluado para queries frecuentes
- [ ] Batch embedding implementado (no 1-by-1)
- [ ] max_tokens ajustado por use case (no defaults)

### Database
- [ ] pg_stat_statements habilitado
- [ ] Queries > 50ms identificadas y optimizadas
- [ ] Unused indexes eliminados
- [ ] VACUUM/ANALYZE automatizado
- [ ] Connection pool saturation monitoreada

### Load Testing
- [ ] Load test con tráfico realista (Locust/k6)
- [ ] LLM latency simulada con delays reales (no mocks instantáneos)
- [ ] Rate limiting validado bajo carga
- [ ] Autoscaling verificado (responde en < 60s a spike)
- [ ] Resultados documentados (baseline, target, actual)

---

## Additional References

- **Python Profilers Comparison**: [pythonspeed.com/articles/beyond-cprofile/](https://pythonspeed.com/articles/beyond-cprofile/)
    - *Best for*: Elegir el profiler correcto para tu caso
- **AsyncIO Debug Mode**: [docs.python.org/3/library/asyncio-dev.html](https://docs.python.org/3/library/asyncio-dev.html)
    - *Best for*: Detectar coroutines bloqueantes y slow callbacks
- **FinOps for AI**: [finops.org](https://www.finops.org/)
    - *Best for*: Framework para gestión de costos cloud y LLM
- **USE Method (Brendan Gregg)**: [brendangregg.com/usemethod.html](https://www.brendangregg.com/usemethod.html)
    - *Best for*: Metodología sistemática de análisis de performance (Utilization, Saturation, Errors)
