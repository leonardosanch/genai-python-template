---
name: Full-Text Search
description: PostgreSQL full-text search with tsvector, GIN indexes, ranking, and SQLAlchemy integration.
---

# Skill: Full-Text Search

## Description

This skill covers full-text search implementation using PostgreSQL's native capabilities. Use this when building search features for candidates, products, articles, or any entity that requires keyword search, ranking, multi-field matching, or autocomplete.

## Executive Summary

**Critical full-text search rules:**
- Use PostgreSQL `tsvector` + `to_tsquery` for proper full-text search — `LIKE '%term%'` does NOT use indexes and degrades at scale
- ALWAYS create a GIN index on `tsvector` columns — without it, every query scans the full table
- Use a generated `search_vector` column updated via trigger — never compute tsvector at query time
- `ts_rank` for relevance ordering — without ranking, results are unsorted and useless
- `ILIKE` is acceptable ONLY for exact substring matching on small datasets (< 10K rows)
- Choose the correct text search configuration for your language (`spanish`, `english`, `simple`)

**Read full skill when:** Implementing search bars, multi-field keyword search, autocomplete, search ranking, or faceted search on PostgreSQL.

---

## Versiones y Dependencias

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| PostgreSQL | >= 12 | FTS estable desde v12; `websearch_to_tsquery` desde v11 |
| SQLAlchemy | >= 2.0.0 | Soporte completo para `tsvector`, `func.to_tsvector` |
| asyncpg | >= 0.28.0 | Driver async recomendado |

> ⚠️ **Configuración de idioma**: PostgreSQL incluye configuraciones para múltiples idiomas (`spanish`, `english`, `french`, etc.). La configuración afecta stemming, stop words, y normalización. Usar `simple` si el contenido es multi-idioma mezclado.

---

## Deep Dive

## Core Concepts

1. **tsvector**: Representación optimizada de un documento para búsqueda. Contiene lexemas normalizados (stems) con posiciones. Se almacena como columna en la tabla para evitar re-computar en cada query.

2. **tsquery**: Representación de la consulta del usuario. Soporta operadores `&` (AND), `|` (OR), `!` (NOT), `<->` (seguido por). `websearch_to_tsquery` acepta sintaxis natural del usuario.

3. **GIN Index**: Generalized Inverted Index — mapea cada lexema a las filas que lo contienen. Crítico para performance. Sin GIN, PostgreSQL hace sequential scan sobre toda la tabla.

4. **Ranking**: `ts_rank` calcula relevancia basada en frecuencia de términos y proximidad. `ts_rank_cd` penaliza distancia entre términos. Siempre ordenar por rank para resultados útiles.

5. **Text Search Configuration**: Define cómo un idioma procesa texto — stemming (`corriendo` → `corr`), stop words (`el`, `de`, `que`), y normalización. Elegir el idioma correcto mejora calidad de búsqueda.

---

## External Resources

### 🔍 PostgreSQL FTS Documentation

- **PostgreSQL Full-Text Search**: [postgresql.org/docs/current/textsearch.html](https://www.postgresql.org/docs/current/textsearch.html)
    - *Best for*: Referencia completa oficial — tsvector, tsquery, ranking, indexes
- **PostgreSQL FTS Tutorial**: [postgresql.org/docs/current/textsearch-intro.html](https://www.postgresql.org/docs/current/textsearch-intro.html)
    - *Best for*: Introducción paso a paso
- **PostgreSQL GIN Indexes**: [postgresql.org/docs/current/gin.html](https://www.postgresql.org/docs/current/gin.html)
    - *Best for*: Entender performance y tuning de GIN

### 🐍 Python & SQLAlchemy

- **SQLAlchemy PostgreSQL Dialects**: [docs.sqlalchemy.org/en/20/dialects/postgresql.html](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html)
    - *Best for*: Tipos PostgreSQL (`TSVECTOR`, `ARRAY`) en SQLAlchemy
- **Cosmic Python — Repository Pattern**: [cosmicpython.com](https://www.cosmicpython.com/)
    - *Best for*: Integrar search en repositories sin violar Clean Architecture

### 🔧 Alternatives & Scaling

- **pg_trgm (Trigram)**: [postgresql.org/docs/current/pgtrgm.html](https://www.postgresql.org/docs/current/pgtrgm.html)
    - *Best for*: Fuzzy search, typo tolerance, `SIMILAR TO`, `%` operator
- **Elasticsearch/OpenSearch**: [elastic.co/guide/en/elasticsearch/reference/current/index.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
    - *Best for*: Full-text search a escala (>10M docs), facets, aggregations
- **Meilisearch**: [meilisearch.com/docs](https://www.meilisearch.com/docs)
    - *Best for*: Search engine ligero, typo-tolerant, instant search

---

## Decision Trees

### Decision Tree 1: Qué mecanismo de búsqueda usar

```
¿Qué tipo de búsqueda necesitas?
├── Búsqueda por keywords en texto largo (descripción, biografía, skills)
│   └── PostgreSQL Full-Text Search (tsvector + tsquery)
│       ├── Stemming, stop words, ranking
│       ├── GIN index para performance
│       └── Cubre 90% de los casos para < 5M registros
├── Búsqueda exacta de substring (buscar "Juan" en nombre)
│   └── ¿Dataset < 10K registros?
│       ├── SÍ → ILIKE con índice trigram (pg_trgm)
│       └── NO → tsvector con config 'simple' (sin stemming)
├── Búsqueda fuzzy / typo-tolerant ("Juann" → "Juan")
│   └── pg_trgm extension
│       ├── CREATE EXTENSION pg_trgm;
│       ├── GIN index con gin_trgm_ops
│       └── Similarity threshold configurable
├── Autocomplete / prefix search ("Ju" → "Juan", "Julia")
│   └── pg_trgm + ILIKE 'Ju%' con GIN index
│       └── O tsvector con :* prefix operator
├── Búsqueda a escala (> 5M docs, facets, aggregations)
│   └── Elasticsearch / OpenSearch / Meilisearch
│       └── Sincronizar desde PostgreSQL (CDC o event-driven)
└── Múltiples campos (nombre + skills + puesto)
    └── tsvector concatenando múltiples campos con pesos (A, B, C, D)
```

### Decision Tree 2: Qué configuración de idioma usar

```
¿En qué idioma está el contenido?
├── Español → 'spanish'
│   └── Stemming: "corriendo" → "corr", stop words: "el", "de", "la"
├── Inglés → 'english'
│   └── Stemming: "running" → "run", stop words: "the", "is", "at"
├── Multi-idioma mezclado → 'simple'
│   └── No stemming, no stop words, solo normalización lowercase
│       └── Pierde calidad de búsqueda pero no rompe con mezcla de idiomas
├── Nombres propios (personas, empresas) → 'simple'
│   └── Stemming no aplica a nombres propios
└── No estoy seguro → 'simple' como fallback seguro
```

---

## Instructions for the Agent

1.  **tsvector como columna generada**: SIEMPRE crear una columna `search_vector` de tipo `TSVECTOR` actualizada por trigger o `GENERATED ALWAYS AS`. Nunca computar `to_tsvector()` en cada query — destruye performance.

2.  **GIN index obligatorio**: Toda columna `tsvector` DEBE tener un GIN index. Sin él, FTS no tiene beneficio sobre sequential scan.

3.  **Pesos para multi-campo**: Usar `setweight()` con categorías A (más relevante) a D (menos relevante). Ejemplo: nombre=A, puesto=B, skills=C, descripción=D.

4.  **`websearch_to_tsquery`**: Preferir sobre `to_tsquery` para input del usuario. Acepta sintaxis natural (`python developer -junior`) en vez de requerir operadores (`python & developer & !junior`).

5.  **Ranking siempre**: `ts_rank` o `ts_rank_cd` para ordenar por relevancia. Nunca devolver resultados FTS sin ordenar.

6.  **Clean Architecture**: El search vive en un repository del domain layer como un método (`search(query: str, filters: SearchFilters) -> list[Entity]`). La implementación FTS de PostgreSQL vive en el adapter de infrastructure.

7.  **Fallback para búsquedas cortas**: Para queries de 1-2 caracteres, usar `ILIKE` con trigram index. `tsvector` no funciona bien con tokens muy cortos.

8.  **Migraciones**: Crear la columna `search_vector`, el trigger, y el GIN index vía Alembic migration. Nunca crear manualmente.

---

## Code Examples

### Example 1: Modelo SQLAlchemy con tsvector

```python
# src/infrastructure/database/models/candidate_model.py
from sqlalchemy import Column, String, Text, Index, text, func, event
from sqlalchemy.dialects.postgresql import TSVECTOR

from src.infrastructure.database.models.base import Base


class CandidateModel(Base):
    """SQLAlchemy model with full-text search support."""

    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    position = Column(String(200))
    skills = Column(Text)  # Comma-separated or JSON
    bio = Column(Text)

    # Generated tsvector column combining multiple fields with weights
    search_vector = Column(
        TSVECTOR,
        nullable=True,
    )

    __table_args__ = (
        # GIN index on the search vector — CRITICAL for performance
        Index(
            "ix_candidates_search_vector",
            search_vector,
            postgresql_using="gin",
        ),
        # Standard indexes for filtered queries
        Index("ix_candidates_email", email),
        Index("ix_candidates_position", position),
    )
```

### Example 2: Alembic Migration para FTS

```python
# alembic/versions/xxxx_add_fulltext_search.py
"""Add full-text search to candidates table."""
from alembic import op


def upgrade() -> None:
    # 1. Add tsvector column
    op.execute("""
        ALTER TABLE candidates
        ADD COLUMN IF NOT EXISTS search_vector tsvector;
    """)

    # 2. Create GIN index
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_candidates_search_vector
        ON candidates USING GIN (search_vector);
    """)

    # 3. Create trigger function to auto-update search_vector
    op.execute("""
        CREATE OR REPLACE FUNCTION candidates_search_vector_update()
        RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('spanish', coalesce(NEW.first_name, '')), 'A') ||
                setweight(to_tsvector('spanish', coalesce(NEW.last_name, '')), 'A') ||
                setweight(to_tsvector('spanish', coalesce(NEW.position, '')), 'B') ||
                setweight(to_tsvector('spanish', coalesce(NEW.skills, '')), 'C') ||
                setweight(to_tsvector('spanish', coalesce(NEW.bio, '')), 'D');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # 4. Create trigger
    op.execute("""
        CREATE TRIGGER candidates_search_vector_trigger
        BEFORE INSERT OR UPDATE OF first_name, last_name, position, skills, bio
        ON candidates
        FOR EACH ROW
        EXECUTE FUNCTION candidates_search_vector_update();
    """)

    # 5. Populate existing rows
    op.execute("""
        UPDATE candidates SET search_vector =
            setweight(to_tsvector('spanish', coalesce(first_name, '')), 'A') ||
            setweight(to_tsvector('spanish', coalesce(last_name, '')), 'A') ||
            setweight(to_tsvector('spanish', coalesce(position, '')), 'B') ||
            setweight(to_tsvector('spanish', coalesce(skills, '')), 'C') ||
            setweight(to_tsvector('spanish', coalesce(bio, '')), 'D');
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS candidates_search_vector_trigger ON candidates;")
    op.execute("DROP FUNCTION IF EXISTS candidates_search_vector_update;")
    op.execute("DROP INDEX IF EXISTS ix_candidates_search_vector;")
    op.execute("ALTER TABLE candidates DROP COLUMN IF EXISTS search_vector;")
```

### Example 3: Port en Domain Layer

```python
# src/domain/ports/search_port.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchFilters:
    """Filters that can be combined with text search."""

    position: str | None = None
    skills: list[str] | None = None
    status: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Single search result with relevance score."""

    entity_id: str
    rank: float  # 0.0 to 1.0, higher = more relevant
    headline: str  # Highlighted snippet with <b> tags


class SearchableRepositoryPort(ABC):
    """Port for repositories that support full-text search."""

    @abstractmethod
    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> tuple[list[SearchResult], int]:
        """Search entities by text query.

        Returns: (results, total_count)
        """
```

### Example 4: Repository Adapter con FTS

```python
# src/infrastructure/database/repositories/candidate_search_repository.py
from sqlalchemy import func, select, text, case
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.ports.search_port import SearchableRepositoryPort, SearchFilters, SearchResult


class PostgresCandidateSearchRepository(SearchableRepositoryPort):
    """Full-text search implementation using PostgreSQL tsvector."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> tuple[list[SearchResult], int]:
        query = query.strip()
        if not query:
            return [], 0

        # Use websearch_to_tsquery for natural user input
        # "python developer -junior" → 'python' & 'developer' & !'junior'
        tsquery = func.websearch_to_tsquery("spanish", query)

        # Base query with ranking
        rank_expr = func.ts_rank_cd(
            CandidateModel.search_vector,
            tsquery,
        )

        # Headline: highlighted snippet for display
        headline_expr = func.ts_headline(
            "spanish",
            func.concat_ws(
                " | ",
                CandidateModel.first_name,
                CandidateModel.last_name,
                CandidateModel.position,
                CandidateModel.skills,
            ),
            tsquery,
            text("'StartSel=<b>, StopSel=</b>, MaxWords=50, MinWords=20'"),
        )

        stmt = (
            select(
                CandidateModel.id,
                rank_expr.label("rank"),
                headline_expr.label("headline"),
            )
            .where(CandidateModel.search_vector.op("@@")(tsquery))
        )

        # Apply additional filters
        if filters:
            if filters.position:
                stmt = stmt.where(CandidateModel.position.ilike(f"%{filters.position}%"))
            if filters.status:
                stmt = stmt.where(CandidateModel.status == filters.status)

        # Count total before pagination
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._session.execute(count_stmt)).scalar() or 0

        # Apply pagination and ordering
        stmt = (
            stmt
            .order_by(rank_expr.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        results = [
            SearchResult(
                entity_id=row.id,
                rank=float(row.rank),
                headline=row.headline,
            )
            for row in rows
        ]

        return results, total
```

### Example 5: Fuzzy Search con pg_trgm

```python
# src/infrastructure/database/repositories/fuzzy_search.py
"""Fuzzy search for short queries and typo tolerance.

Requires: CREATE EXTENSION pg_trgm;
"""
from sqlalchemy import func, select, text, Index
from sqlalchemy.ext.asyncio import AsyncSession


# Add trigram index (in Alembic migration)
# CREATE INDEX ix_candidates_name_trgm
#     ON candidates USING GIN (
#         (first_name || ' ' || last_name) gin_trgm_ops
#     );


class FuzzySearchMixin:
    """Mixin for repositories that need fuzzy matching."""

    async def fuzzy_search_by_name(
        self,
        session: AsyncSession,
        query: str,
        similarity_threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Search by name with typo tolerance (pg_trgm).

        Example: "Juann Perez" matches "Juan Pérez" with similarity 0.7
        """
        full_name = func.concat_ws(" ", CandidateModel.first_name, CandidateModel.last_name)
        similarity = func.similarity(full_name, query)

        stmt = (
            select(
                CandidateModel.id,
                CandidateModel.first_name,
                CandidateModel.last_name,
                similarity.label("score"),
            )
            .where(similarity > similarity_threshold)
            .order_by(similarity.desc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        return [
            {
                "id": row.id,
                "name": f"{row.first_name} {row.last_name}",
                "score": float(row.score),
            }
            for row in result
        ]
```

### Example 6: FastAPI Search Endpoint

```python
# src/interfaces/api/routes/search_routes.py
from fastapi import APIRouter, Depends, Query

from src.domain.ports.search_port import SearchableRepositoryPort, SearchFilters

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.get("/candidates")
async def search_candidates(
    q: str = Query(..., min_length=1, max_length=200, description="Texto de búsqueda"),
    position: str | None = Query(None, description="Filtrar por posición"),
    status: str | None = Query(None, description="Filtrar por estado"),
    page: int = Query(1, ge=1, description="Número de página"),
    limit: int = Query(25, ge=1, le=100, description="Resultados por página"),
    search_repo: SearchableRepositoryPort = Depends(get_search_repository),
):
    """Full-text search on candidates.

    Supports natural query syntax:
    - `python developer` → matches both terms
    - `"python developer"` → matches exact phrase
    - `python -junior` → matches python, excludes junior
    """
    filters = SearchFilters(position=position, status=status)
    results, total = await search_repo.search(
        query=q,
        filters=filters,
        page=page,
        page_size=limit,
    )

    return {
        "items": [
            {
                "id": r.entity_id,
                "rank": round(r.rank, 4),
                "headline": r.headline,
            }
            for r in results
        ],
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit,
    }
```

---

## Anti-Patterns to Avoid

### ❌ LIKE '%term%' on Large Tables
**Problem**: No index usage, sequential scan on every query, O(n) performance
**Example**:
```python
# BAD: Sequential scan on 500K rows
stmt = select(Candidate).where(Candidate.skills.like(f"%{query}%"))
```
**Solution**: `tsvector` with GIN index
```python
# GOOD: GIN index lookup, O(log n)
tsquery = func.websearch_to_tsquery("spanish", query)
stmt = select(Candidate).where(Candidate.search_vector.op("@@")(tsquery))
```

### ❌ Computing tsvector at Query Time
**Problem**: `to_tsvector()` called on every row, every query — O(n) and CPU-intensive
**Example**:
```sql
-- BAD: Computes tsvector for every row on every search
SELECT * FROM candidates
WHERE to_tsvector('spanish', first_name || ' ' || skills) @@ to_tsquery('python');
```
**Solution**: Pre-computed column with trigger
```sql
-- GOOD: Pre-computed, indexed column
SELECT * FROM candidates
WHERE search_vector @@ websearch_to_tsquery('spanish', 'python');
```

### ❌ No Ranking in Results
**Problem**: Results returned in arbitrary order — user sees irrelevant results first
**Solution**: Always use `ts_rank` or `ts_rank_cd` with `ORDER BY rank DESC`

### ❌ Using to_tsquery with Raw User Input
**Problem**: `to_tsquery` requires operator syntax — raw input like `python developer` fails
**Solution**: Use `websearch_to_tsquery` (accepts natural language) or `plainto_tsquery` (AND all terms)

### ❌ Wrong Language Configuration
**Problem**: Using `english` config for Spanish content — stemming produces wrong results, stop words not removed
**Solution**: Match the configuration to the content language. Use `simple` for mixed languages or proper nouns.

---

## Full-Text Search Checklist

### Schema
- [ ] `search_vector` column of type `TSVECTOR` on searchable tables
- [ ] Trigger function to auto-update `search_vector` on INSERT/UPDATE
- [ ] GIN index on `search_vector` column
- [ ] `pg_trgm` extension enabled if fuzzy search needed
- [ ] Trigram GIN index on name/title columns if autocomplete needed
- [ ] Correct language configuration (`spanish`, `english`, `simple`)

### Query
- [ ] `websearch_to_tsquery` for user-facing search (natural syntax)
- [ ] `ts_rank` or `ts_rank_cd` for relevance ordering
- [ ] `ts_headline` for highlighted snippets in results
- [ ] `setweight` with A/B/C/D for multi-field priority
- [ ] Fallback to `ILIKE` for queries < 2 characters

### Performance
- [ ] `EXPLAIN ANALYZE` confirms GIN index usage (no Seq Scan)
- [ ] Pagination with `LIMIT`/`OFFSET` (or keyset for large datasets)
- [ ] `COUNT(*)` cached or approximated for large tables
- [ ] Monitoring: average search latency, slow query log

### Architecture
- [ ] Search method defined in domain port (`SearchableRepositoryPort`)
- [ ] FTS implementation lives in infrastructure adapter
- [ ] SQLAlchemy `func.*` used instead of raw SQL where possible
- [ ] Alembic migration for trigger, column, and index creation

### Testing
- [ ] Tests with real PostgreSQL (docker) for FTS queries
- [ ] Tests verify ranking order (more relevant results first)
- [ ] Tests for edge cases: empty query, special characters, very long query
- [ ] Tests for multi-word queries, phrase matching, negation

---

## Additional References

- [PostgreSQL Full-Text Search Documentation](https://www.postgresql.org/docs/current/textsearch.html)
- [PostgreSQL pg_trgm Extension](https://www.postgresql.org/docs/current/pgtrgm.html)
- [SQLAlchemy PostgreSQL Dialects](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html)
- [Full-Text Search in PostgreSQL — Practical Guide](https://www.postgresql.org/docs/current/textsearch-intro.html)
- [Crunchy Data — PostgreSQL FTS Tutorial](https://www.crunchydata.com/blog/postgres-full-text-search-a-search-engine-in-a-database)
- [GIN vs GiST Indexes](https://www.postgresql.org/docs/current/textsearch-indexes.html)
