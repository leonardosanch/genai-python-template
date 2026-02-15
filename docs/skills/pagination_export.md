---
name: Pagination & Data Export
description: Server-side pagination patterns, cursor-based pagination, and streaming CSV/Excel export.
---

# Skill: Pagination & Data Export

## Description

This skill covers server-side pagination and data export for Python backend applications. Use this when implementing paginated list endpoints, exporting data to CSV/Excel, generating reports, or handling large result sets efficiently.

## Executive Summary

**Critical pagination & export rules:**
- NEVER return unbounded result sets — every list endpoint MUST have pagination with configurable `page` + `limit`
- Use a generic `PaginatedResponse[T]` schema — consistent response format across ALL paginated endpoints
- Offset pagination for simple cases (< 100K rows); cursor/keyset pagination for large datasets
- NEVER use `fetchall()` + Python slicing for pagination — `LIMIT`/`OFFSET` at the database level
- CSV/Excel exports with > 1K rows MUST use `StreamingResponse` or Celery background task
- `COUNT(*)` can be expensive — cache it or use approximate count for tables > 1M rows

**Read full skill when:** Implementing list endpoints, adding pagination to existing APIs, exporting data to CSV/Excel, building report generation, or optimizing slow paginated queries.

---

## Versiones y Dependencias

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| fastapi | >= 0.100.0 | Query params, StreamingResponse |
| pydantic | >= 2.0.0 | Generics para PaginatedResponse[T] |
| openpyxl | >= 3.1.0 | ✅ Estable — generación de archivos Excel (.xlsx) |
| xlsxwriter | >= 3.2.0 | ✅ Alternativa más rápida que openpyxl para escritura |
| csv (stdlib) | — | Módulo estándar de Python |

> ⚠️ **openpyxl vs xlsxwriter**: `openpyxl` lee y escribe Excel. `xlsxwriter` solo escribe pero es ~2x más rápido y usa menos memoria. Para exports, preferir `xlsxwriter`.

---

## Deep Dive

## Core Concepts

1. **Offset Pagination**: `OFFSET n LIMIT m` — simple, stateless, soporta saltar a cualquier página. Degradación de performance en offsets grandes (PostgreSQL debe escanear y descartar las primeras N filas).

2. **Cursor/Keyset Pagination**: `WHERE id > last_id ORDER BY id LIMIT m` — performance constante independiente de la página. No soporta saltar a página arbitraria. Ideal para infinite scroll o datasets grandes.

3. **Standardized Response**: Toda respuesta paginada devuelve `{items, total, page, page_size, total_pages}`. Schema genérico reutilizable en todos los endpoints.

4. **Streaming Export**: Para archivos grandes, usar `StreamingResponse` que envía datos chunk por chunk. El servidor nunca carga el archivo completo en memoria.

5. **Background Export**: Para exports muy grandes (> 10K filas) o pesados (Excel con formato), usar Celery task. El usuario recibe una notificación cuando el archivo está listo.

---

## External Resources

### 📄 Pagination

- **REST API Pagination Best Practices**: [nordicapis.com/everything-you-need-to-know-about-api-pagination/](https://nordicapis.com/everything-you-need-to-know-about-api-pagination/)
    - *Best for*: Comparison of offset, cursor, and keyset pagination
- **PostgreSQL OFFSET Performance**: [use-the-index-luke.com/no-offset](https://use-the-index-luke.com/no-offset)
    - *Best for*: Why OFFSET is slow and how keyset pagination fixes it
- **SQLAlchemy Pagination**: [docs.sqlalchemy.org/en/20/orm/queryguide/query.html](https://docs.sqlalchemy.org/en/20/orm/queryguide/query.html)
    - *Best for*: `limit()`, `offset()`, `select()` with async

### 📊 Export & Reporting

- **openpyxl Documentation**: [openpyxl.readthedocs.io](https://openpyxl.readthedocs.io/)
    - *Best for*: Read/write Excel files, formatting, charts
- **xlsxwriter Documentation**: [xlsxwriter.readthedocs.io](https://xlsxwriter.readthedocs.io/)
    - *Best for*: Fast write-only Excel generation, streaming mode
- **FastAPI StreamingResponse**: [fastapi.tiangolo.com/advanced/custom-response/](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
    - *Best for*: Streaming large files without loading in memory

---

## Decision Trees

### Decision Tree 1: Qué tipo de paginación usar

```
¿Qué tipo de paginación necesitas?
├── Lista con navegación por página (1, 2, 3... última)
│   └── ¿Dataset < 100K filas?
│       ├── SÍ → Offset pagination (page + page_size)
│       │   └── Simple, stateless, soporta "ir a página X"
│       └── NO → Keyset pagination con opción de COUNT aproximado
│           └── Performance constante, pero sin "ir a página X"
├── Infinite scroll / "Cargar más"
│   └── Cursor pagination (after_id + limit)
│       ├── Performance constante O(1) vs O(n) de offset
│       ├── Solo "siguiente" y "anterior", no saltar páginas
│       └── Ideal para mobile y feeds
├── Paginación de búsqueda (FTS results)
│   └── Offset pagination (los resultados ya están rankeados)
│       └── FTS rara vez tiene > 100K resultados relevantes
└── API pública / third-party consumers
    └── Cursor pagination (Link headers, RFC 8288)
        └── Más resiliente a inserts/deletes concurrentes
```

### Decision Tree 2: Cómo exportar datos

```
¿Cómo exportar datos?
├── Pocos registros (< 1K filas)
│   └── CSV con StreamingResponse en el endpoint
│       └── Respuesta inmediata, ~100ms
├── Muchos registros (1K - 100K filas)
│   └── CSV con StreamingResponse + chunked DB queries
│       ├── Usar server-side cursor (fetchmany)
│       └── Nunca fetchall() → streaming por chunks de 500
├── Excel (cualquier tamaño)
│   └── ¿Necesita formato (colores, anchos, fórmulas)?
│       ├── SÍ → Celery task + xlsxwriter + notificación al usuario
│       └── NO → CSV (más rápido y ligero que Excel sin formato)
├── Reportes con gráficos
│   └── Celery task + xlsxwriter (charts nativos)
│       └── O generar PDF con WeasyPrint/ReportLab
└── Datasets masivos (> 100K filas)
    └── Celery task + archivo temporal en S3
        ├── Generar en background
        ├── Subir a S3
        ├── Enviar presigned URL al usuario
        └── Auto-cleanup con lifecycle policy
```

---

## Instructions for the Agent

1.  **PaginatedResponse genérico**: SIEMPRE usar el schema genérico `PaginatedResponse[T]`. Nunca crear un schema manual `CandidateListResponse(items, total, page...)` — el genérico garantiza consistencia.

2.  **Parámetros estándar**: Todo endpoint paginado recibe `page: int = 1` (≥1) y `limit: int = 25` (≥1, ≤100). Nunca aceptar `limit > 100` sin rate limiting especial.

3.  **COUNT**: Usar `SELECT COUNT(*)` con los mismos WHERE clauses que la query principal. Para tablas > 1M filas, considerar cache (Redis, 60s TTL) o PostgreSQL approximate count (`reltuples`).

4.  **Offset**: `OFFSET = (page - 1) * limit`. Siempre en la DB query, nunca en Python.

5.  **StreamingResponse para CSV**: Usar generator que hace `yield` por cada fila o chunk. Header `Content-Disposition: attachment; filename="export.csv"`.

6.  **Excel en background**: openpyxl/xlsxwriter NO son async. Ejecutar en Celery task. Guardar archivo resultante en FileStoragePort. Notificar al usuario vía WebSocket o email.

7.  **Clean Architecture**: La paginación es un concern de application/interface layer. El repository devuelve `(items, total_count)`. El schema `PaginatedResponse` vive en interfaces.

---

## Code Examples

### Example 1: Schema Genérico PaginatedResponse

```python
# src/interfaces/api/schemas/pagination.py
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Standard pagination query parameters."""

    page: int = Field(1, ge=1, description="Número de página")
    limit: int = Field(25, ge=1, le=100, description="Elementos por página")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated response — use for ALL list endpoints.

    Usage:
        PaginatedResponse[CandidateSchema]
    """

    items: list[T]
    total: int = Field(description="Total de registros")
    page: int = Field(description="Página actual")
    page_size: int = Field(description="Elementos por página")
    total_pages: int = Field(description="Total de páginas")

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        """Factory method para crear respuesta paginada."""
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=max(1, (total + page_size - 1) // page_size),
        )
```

### Example 2: Repository con Paginación

```python
# src/infrastructure/database/repositories/candidate_repository.py
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.candidate_model import CandidateModel


class CandidateRepository:
    """Repository with paginated queries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_all_paginated(
        self,
        offset: int,
        limit: int,
        status: str | None = None,
    ) -> tuple[list[CandidateModel], int]:
        """Return paginated results with total count.

        Returns: (items, total_count)
        """
        # Base query with filters
        base_query = select(CandidateModel)
        count_query = select(func.count(CandidateModel.id))

        if status:
            base_query = base_query.where(CandidateModel.status == status)
            count_query = count_query.where(CandidateModel.status == status)

        # Total count (same WHERE clauses)
        total = (await self._session.execute(count_query)).scalar() or 0

        # Paginated items
        stmt = (
            base_query
            .order_by(CandidateModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        items = list(result.scalars().all())

        return items, total
```

### Example 3: FastAPI Endpoint Paginado

```python
# src/interfaces/api/routes/candidate_routes.py
from fastapi import APIRouter, Depends, Query

from src.interfaces.api.schemas.pagination import PaginatedResponse, PaginationParams
from src.interfaces.api.schemas.candidate import CandidateSchema

router = APIRouter(prefix="/api/v1/candidates", tags=["candidates"])


@router.get("", response_model=PaginatedResponse[CandidateSchema])
async def list_candidates(
    page: int = Query(1, ge=1, description="Página"),
    limit: int = Query(25, ge=1, le=100, description="Elementos por página"),
    status: str | None = Query(None, description="Filtrar por estado"),
    repo: CandidateRepository = Depends(get_candidate_repository),
):
    """List candidates with pagination.

    Response format:
    ```json
    {
        "items": [...],
        "total": 150,
        "page": 1,
        "page_size": 25,
        "total_pages": 6
    }
    ```
    """
    params = PaginationParams(page=page, limit=limit)
    items, total = await repo.find_all_paginated(
        offset=params.offset,
        limit=params.limit,
        status=status,
    )

    return PaginatedResponse.create(
        items=[CandidateSchema.model_validate(item) for item in items],
        total=total,
        page=params.page,
        page_size=params.limit,
    )
```

### Example 4: Cursor/Keyset Pagination

```python
# src/interfaces/api/schemas/cursor_pagination.py
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class CursorPaginatedResponse(BaseModel, Generic[T]):
    """Cursor-based pagination for infinite scroll and large datasets."""

    items: list[T]
    next_cursor: str | None = Field(None, description="Cursor para la siguiente página")
    has_more: bool = Field(description="Si hay más resultados")


# Repository method for keyset pagination
async def find_after_cursor(
    self,
    cursor_id: str | None,
    limit: int = 25,
) -> tuple[list[CandidateModel], str | None]:
    """Keyset pagination — O(1) performance regardless of page depth.

    cursor_id: ID of the last item from previous page (None for first page)
    Returns: (items, next_cursor)
    """
    stmt = select(CandidateModel).order_by(CandidateModel.created_at.desc(), CandidateModel.id.desc())

    if cursor_id:
        # Fetch the cursor row to get its created_at
        cursor_row = await self._session.get(CandidateModel, cursor_id)
        if cursor_row:
            stmt = stmt.where(
                (CandidateModel.created_at < cursor_row.created_at)
                | (
                    (CandidateModel.created_at == cursor_row.created_at)
                    & (CandidateModel.id < cursor_id)
                )
            )

    # Fetch limit + 1 to check if there are more results
    stmt = stmt.limit(limit + 1)
    result = await self._session.execute(stmt)
    items = list(result.scalars().all())

    has_more = len(items) > limit
    if has_more:
        items = items[:limit]

    next_cursor = items[-1].id if items and has_more else None
    return items, next_cursor
```

### Example 5: CSV Export con StreamingResponse

```python
# src/interfaces/api/routes/export_routes.py
import csv
import io
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1/export", tags=["export"])

CHUNK_SIZE = 500  # Rows per DB fetch


@router.get("/candidates/csv")
async def export_candidates_csv(
    status: str | None = None,
    repo: CandidateRepository = Depends(get_candidate_repository),
):
    """Export candidates to CSV with streaming.

    - Streams response chunk by chunk (never loads all data in memory)
    - Fetches from DB in batches of 500 rows
    """

    async def generate_csv() -> AsyncGenerator[str, None]:
        # Header row
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Nombre", "Apellido", "Email", "Posición", "Estado", "Fecha Creación"])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        # Data rows in chunks
        offset = 0
        while True:
            items, _ = await repo.find_all_paginated(
                offset=offset,
                limit=CHUNK_SIZE,
                status=status,
            )

            if not items:
                break

            for candidate in items:
                writer.writerow([
                    candidate.id,
                    candidate.first_name,
                    candidate.last_name,
                    candidate.email,
                    candidate.position,
                    candidate.status,
                    candidate.created_at.isoformat(),
                ])

            yield output.getvalue()
            output.seek(0)
            output.truncate(0)
            offset += CHUNK_SIZE

            if len(items) < CHUNK_SIZE:
                break

    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="candidates_export.csv"',
        },
    )
```

### Example 6: Excel Export en Celery Task

```python
# src/infrastructure/tasks/export_tasks.py
import io
from datetime import datetime

from celery import shared_task
import structlog
import xlsxwriter

logger = structlog.get_logger()


@shared_task(
    bind=True,
    max_retries=2,
    time_limit=300,  # 5 min max
)
def export_candidates_excel_task(
    self,
    filters: dict | None = None,
    requested_by_user_id: str = "",
) -> str:
    """Generate Excel export in background.

    Returns: file_id of the exported file in storage.
    """
    import asyncio
    from src.infrastructure.container import Container

    async def _generate() -> str:
        container = Container()
        repo = container.candidate_repository()
        storage = container.file_storage()

        # Create Excel in memory
        buffer = io.BytesIO()
        workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
        worksheet = workbook.add_worksheet("Candidatos")

        # Styles
        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#2563EB",
            "font_color": "#FFFFFF",
            "border": 1,
        })
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm"})

        # Headers
        headers = ["ID", "Nombre", "Apellido", "Email", "Posición", "Estado", "Fecha"]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)

        # Data in chunks
        row = 1
        offset = 0
        chunk_size = 500
        while True:
            items, _ = await repo.find_all_paginated(offset=offset, limit=chunk_size)
            if not items:
                break

            for candidate in items:
                worksheet.write(row, 0, candidate.id)
                worksheet.write(row, 1, candidate.first_name)
                worksheet.write(row, 2, candidate.last_name)
                worksheet.write(row, 3, candidate.email)
                worksheet.write(row, 4, candidate.position or "")
                worksheet.write(row, 5, candidate.status or "")
                worksheet.write_datetime(row, 6, candidate.created_at, date_format)
                row += 1

            offset += chunk_size
            if len(items) < chunk_size:
                break

        # Auto-fit columns
        worksheet.set_column(0, 0, 36)  # ID (UUID)
        worksheet.set_column(1, 2, 20)  # Nombre, Apellido
        worksheet.set_column(3, 3, 30)  # Email
        worksheet.set_column(4, 5, 20)  # Posición, Estado
        worksheet.set_column(6, 6, 18)  # Fecha

        workbook.close()
        buffer.seek(0)

        # Save to file storage
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"candidatos_export_{timestamp}.xlsx"
        stored = await storage.save(
            file_content=buffer.getvalue(),
            original_name=filename,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        logger.info(
            "excel_export_completed",
            file_id=stored.file_id,
            rows=row - 1,
            requested_by=requested_by_user_id,
        )

        return stored.file_id

    return asyncio.run(_generate())
```

### Example 7: Endpoint para Solicitar Export

```python
# src/interfaces/api/routes/export_routes.py (additional endpoint)
from fastapi import APIRouter, Depends, BackgroundTasks

router = APIRouter(prefix="/api/v1/export", tags=["export"])


@router.post("/candidates/excel")
async def request_candidates_excel(
    current_user = Depends(get_current_user),
):
    """Request Excel export (generated in background).

    Returns immediately with a task ID.
    The user is notified when the file is ready.
    """
    from src.infrastructure.tasks.export_tasks import export_candidates_excel_task

    task = export_candidates_excel_task.delay(
        requested_by_user_id=str(current_user.id),
    )

    return {
        "task_id": task.id,
        "status": "processing",
        "message": "El archivo se está generando. Recibirás una notificación cuando esté listo.",
    }


@router.get("/status/{task_id}")
async def get_export_status(task_id: str):
    """Check export task status."""
    from celery.result import AsyncResult

    result = AsyncResult(task_id)

    if result.ready():
        if result.successful():
            file_id = result.result
            return {
                "status": "completed",
                "file_id": file_id,
                "download_url": f"/api/v1/files/{file_id}",
            }
        return {"status": "failed", "error": str(result.result)}

    return {"status": "processing"}
```

---

## Anti-Patterns to Avoid

### ❌ Fetching All Rows and Slicing in Python
**Problem**: Loads entire table into memory, then takes a slice
**Example**:
```python
# BAD: Fetches ALL candidates, then slices
all_candidates = await repo.find_all()  # 500K rows in memory!
page_items = all_candidates[offset:offset+limit]
```
**Solution**: DB-level pagination
```python
# GOOD: Only fetches the requested page
stmt = select(Candidate).offset(offset).limit(limit)
```

### ❌ No Total Count in Response
**Problem**: Frontend can't show page numbers or "X results found"
**Solution**: Always include `total` and `total_pages` in response

### ❌ Unbounded Limit
**Problem**: Client requests `limit=99999`, loading too much data
**Solution**: Cap `limit` at 100 via Pydantic validation
```python
limit: int = Query(25, ge=1, le=100)
```

### ❌ Building Full Excel in Memory for Large Datasets
**Problem**: 100K rows × 10 columns = ~200MB RAM per request
**Solution**: Celery task + xlsxwriter (streaming mode) + file storage

### ❌ CSV Without StreamingResponse
**Problem**: Server builds entire CSV string, then returns it — memory spike
**Example**:
```python
# BAD: Entire CSV in memory
csv_content = generate_csv_string(all_items)  # 50MB string
return Response(content=csv_content)
```
**Solution**: Generator + `StreamingResponse`
```python
# GOOD: Streaming — constant memory usage
return StreamingResponse(generate_csv_chunks(), media_type="text/csv")
```

### ❌ COUNT(*) on Every Request for Huge Tables
**Problem**: `COUNT(*)` scans the full table — slow on > 1M rows
**Solution**: Cache count in Redis (TTL 30-60s) or use approximate count
```sql
-- Approximate count (instant, ~95% accurate)
SELECT reltuples::bigint AS estimate
FROM pg_class WHERE relname = 'candidates';
```

---

## Pagination & Export Checklist

### Pagination
- [ ] `PaginatedResponse[T]` generic schema used for ALL list endpoints
- [ ] `page` (≥1) and `limit` (1-100) with Pydantic validation
- [ ] `LIMIT`/`OFFSET` at database level (never Python slicing)
- [ ] Response includes `items`, `total`, `page`, `page_size`, `total_pages`
- [ ] `ORDER BY` always present (deterministic ordering)
- [ ] Cursor pagination evaluated for datasets > 100K rows
- [ ] `COUNT(*)` uses same WHERE clauses as data query

### CSV Export
- [ ] `StreamingResponse` for all CSV exports
- [ ] Chunked DB queries (never `fetchall()`)
- [ ] `Content-Disposition: attachment` header
- [ ] Proper filename with timestamp
- [ ] UTF-8 BOM for Excel compatibility (`\ufeff` prefix)

### Excel Export
- [ ] Background task (Celery) for large exports
- [ ] xlsxwriter for write-only (faster than openpyxl)
- [ ] Time limit on task (max 5 minutes)
- [ ] File saved to storage (FileStoragePort), not returned inline
- [ ] User notification when export is ready
- [ ] Auto-cleanup of old export files

### Performance
- [ ] Indexes on columns used in ORDER BY
- [ ] COUNT cached for tables > 1M rows
- [ ] Monitoring: export duration, file sizes, concurrent exports
- [ ] Rate limiting on export endpoints (prevent abuse)

---

## Additional References

- [PostgreSQL OFFSET Performance](https://use-the-index-luke.com/no-offset)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [xlsxwriter Documentation](https://xlsxwriter.readthedocs.io/)
- [openpyxl Documentation](https://openpyxl.readthedocs.io/)
- [RFC 8288 — Web Linking (cursor pagination)](https://datatracker.ietf.org/doc/html/rfc8288)
- [REST API Pagination Patterns](https://nordicapis.com/everything-you-need-to-know-about-api-pagination/)
