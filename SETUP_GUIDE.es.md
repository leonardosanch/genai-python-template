# Guía de Configuración — Claude Code + GenAI Template

Esta guía explica cómo configurar Claude Code con las reglas de ingeniería
y skills especializados incluidos en este template.

Hay dos escenarios:
1. **Crear un proyecto nuevo desde cero** usando este template
2. **Trabajar en un proyecto existente** (ej: un repo de la empresa que clonaste)

Ambos escenarios están cubiertos a continuación.

---

## Prerrequisitos

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) instalado
- [uv](https://docs.astral.sh/uv/) instalado (gestor de paquetes Python)
- Git instalado

---

## Escenario 1: Crear un Proyecto Nuevo

Quieres iniciar un proyecto nuevo (ej: `sistema_pos`) usando la arquitectura,
reglas y skills del template.

### Paso 1 — Clonar el template (una vez)

```bash
git clone <repo-url> ~/templates/genai-python-template
```

### Paso 2 — Crear tu proyecto

```bash
cd ~/templates/genai-python-template
./scripts/new-project.sh ~/proyectos/sistema_pos
```

Esto copia:
- `CLAUDE.md` con reglas del proyecto y referencias a skills (rutas relativas)
- `docs/skills/` — 19 archivos de skills especializados
- `.claude/` — comandos, hooks, configuración
- `.github/workflows/` — configuración CI/CD
- `deploy/` — scaffold de Terraform
- `pyproject.toml`, `.env.example`, `.pre-commit-config.yaml`
- Estructura limpia de `src/` (layout de Clean Architecture)
- Directorios vacíos `tests/unit/` y `tests/integration/`

NO copia código de ejemplo. Empiezas con un proyecto limpio.

Si quieres el código de ejemplo como referencia:

```bash
./scripts/new-project.sh ~/proyectos/sistema_pos --with-examples
```

### Paso 3 — Empezar a trabajar

```bash
cd ~/proyectos/sistema_pos
uv sync
claude
```

Claude Code lee `./CLAUDE.md` y tiene acceso a los 19 skills vía
rutas relativas dentro de `docs/skills/`.

**Esta configuración es totalmente portable** — cualquiera que clone tu proyecto
obtiene las mismas reglas y skills automáticamente.

---

## Escenario 2: Trabajar en un Proyecto Existente de la Empresa

Te asignan trabajar en un proyecto que ya existe (ej: `inventory-api` de tu empresa).
Este proyecto tiene su propio repo y NO incluye las reglas o skills del template.

### Paso 1 — Clonar el template (una vez)

```bash
git clone <repo-url> ~/templates/genai-python-template
```

### Paso 2 — Instalar configuración global

```bash
cd ~/templates/genai-python-template
./scripts/setup-global.sh
```

Esto instala `~/.claude/CLAUDE.md` con:
- Todas las reglas de ingeniería (SOLID, Clean Architecture, seguridad, etc.)
- Referencias a los 19 skills usando rutas absolutas a tu directorio del template

Solo ejecutas esto **una vez por máquina**.

### Paso 3 — Clonar el proyecto de la empresa

```bash
git clone git@company.com/team/inventory-api.git ~/projects/inventory-api
```

### Paso 4 — Empezar a trabajar

```bash
cd ~/projects/inventory-api
claude
```

Claude Code carga automáticamente `~/.claude/CLAUDE.md` (global) en cualquier
directorio. Obtienes todas las reglas de ingeniería y skills sin copiar nada
al proyecto de la empresa.

### Cómo funciona

```
Claude Code carga (en orden):

1. ~/.claude/CLAUDE.md          ← Reglas globales + 19 skills (rutas absolutas)
2. ~/projects/inventory-api/CLAUDE.md  ← Reglas del proyecto (si existe)

Ambos se fusionan. El archivo del proyecto se suma al global, no lo reemplaza.
```

### Importante

- NO muevas ni renombres el directorio del template después de ejecutar `setup-global.sh`.
  La configuración global lo referencia con rutas absolutas.
- Si mueves el template, ejecuta `setup-global.sh` nuevamente desde la nueva ubicación.

---

## Escenario 2b: Agregar Contexto del Proyecto (Recomendado)

Cuando trabajas en un proyecto específico, Claude Code conoce las reglas de ingeniería
(de la configuración global) pero NO sabe nada sobre **ese proyecto**: qué hace,
qué stack usa, cuáles son las reglas de negocio.

Puedes darle ese contexto creando un `CLAUDE.md` dentro del proyecto.
Esto es **opcional pero altamente recomendado** — hace que Claude Code sea
significativamente más efectivo porque entiende el dominio de negocio, no solo la tecnología.

### Qué incluir en el CLAUDE.md del proyecto

| Sección | Qué incluir | Ejemplo |
|---------|-------------|---------|
| **Contexto** | Qué es el proyecto, quién lo usa | "Backend del sistema de nómina de la empresa" |
| **Stack** | Tecnologías y versiones | "FastAPI + PostgreSQL + Redis + Celery" |
| **Reglas de negocio** | Restricciones del dominio que Claude debe respetar | "Valores monetarios siempre Decimal, nunca float" |
| **Historias de usuario** | Stories del sprint actual o backlog | "HU-042: Generar reportes de nómina en PDF" |
| **Decisiones del equipo** | Decisiones arquitectónicas ya tomadas | "Auth: JWT con refresh tokens" |
| **Convenciones** | Nombres, estructura, patrones específicos del proyecto | "Todos los endpoints bajo /api/v1/inventory/" |

### Ejemplo completo

```bash
cd ~/projects/inventory-api
```

Crear `CLAUDE.md`:

```markdown
# Inventory API — Gestión de Almacén

## Contexto
Backend del sistema de inventario de la empresa. Maneja productos, niveles de stock,
órdenes de compra y operaciones de almacén.
Usado por personal de almacén y compras (herramienta interna, no pública).

## Stack
- FastAPI + PostgreSQL 16 + Redis 7
- Celery para trabajos en background (generación de reportes, importaciones masivas)
- Alembic para migraciones
- Auth: JWT con refresh tokens (integración con SSO de la empresa)

## Reglas de Negocio
- Todos los valores monetarios usan Decimal con 2 decimales, nunca float
- Los niveles de stock no pueden ser negativos
- Auditoría requerida para todas las operaciones de escritura (quién, cuándo, qué cambió)
- Multi-almacén: toda consulta debe filtrar por warehouse_id
- Alertas de reorden cuando el stock cae por debajo del umbral mínimo

## Historias de Usuario (Sprint Actual)
- US-101: Como gerente, quiero generar reportes de inventario en PDF
- US-102: Como personal de almacén, quiero escanear códigos de barras para actualizar stock
- US-103: Como compras, quiero crear órdenes de compra desde items con bajo stock
- US-104: Como admin, quiero importar productos masivamente desde CSV

## Decisiones del Equipo
- Read replica para consultas de reportes (nunca queries pesados en primary)
- Celery para toda generación de PDF/CSV (async, nunca en request path)
- Todas las respuestas API incluyen paginación (máx 100 items por página)
- Códigos de error siguen estándar de la empresa: formato INV-XXX

## Convenciones
- Endpoints: /api/v1/inventory/...
- Nombres de ramas: feature/US-XXX-descripcion-corta
- Mensajes de commit: "US-XXX: descripción"
```

### Cómo Claude Code usa esto

Cuando escribes:

```
claude "implementar US-101"
```

Claude Code ya sabe:
- **Qué** es US-101 (generar reportes de inventario en PDF)
- **Cómo** hacerlo (Celery para async, read replica para queries)
- **Restricciones** (Decimal para dinero, auditoría, multi-almacén)
- **Estándares de calidad** (de los 19 skills globales)

No necesitas explicar nada de esto en el prompt.

### Mantenerlo actualizado

Actualiza el `CLAUDE.md` a medida que el proyecto evoluciona:
- ¿Nuevo sprint? Actualiza la sección de historias de usuario
- ¿Nueva decisión del equipo? Agrégala a decisiones del equipo
- ¿Nueva regla de negocio? Agrégala a reglas de negocio

Piénsalo como **el cerebro del proyecto** — mientras más contexto tenga,
mejor se desempeña Claude Code.

### Tres capas trabajando juntas

```
┌─────────────────────────────────────────────────┐
│  Prompt del CLI                                 │
│  "implementar HU-042"                           │
│  → La tarea específica que quieres hacer AHORA  │
├─────────────────────────────────────────────────┤
│  CLAUDE.md del Proyecto                         │
│  Stack, reglas de negocio, historias de usuario │
│  → QUÉ es el proyecto y QUÉ construir           │
├─────────────────────────────────────────────────┤
│  ~/.claude/CLAUDE.md Global                     │
│  SOLID, Clean Architecture, 19 skills           │
│  → CÓMO construirlo con calidad                 │
└─────────────────────────────────────────────────┘
```

---

## Combinar Ambos Escenarios

Puedes usar ambos enfoques simultáneamente:

| Proyecto | Origen | Reglas cargadas |
|----------|--------|-----------------|
| `sistema_pos` | Creado desde template | `./CLAUDE.md` (portable, con rutas relativas a skills) |
| `inventory-api` | Repo de la empresa | `~/.claude/CLAUDE.md` (global, con rutas absolutas a skills) |
| Cualquier otro dir | Cualquiera | `~/.claude/CLAUDE.md` (global) |

La configuración global es tu red de seguridad — cada proyecto en tu máquina
obtiene las reglas de ingeniería y skills, incluso si no tiene `CLAUDE.md`.

---

## Referencia Rápida

### Comandos

```bash
# Instalar configuración global (una vez por máquina)
cd ~/templates/genai-python-template
./scripts/setup-global.sh

# Crear nuevo proyecto desde template
./scripts/new-project.sh ~/proyectos/mi-proyecto

# Crear nuevo proyecto con código de ejemplo
./scripts/new-project.sh ~/proyectos/mi-proyecto --with-examples

# Re-instalar configuración global (después de mover template o actualizar skills)
./scripts/setup-global.sh --force
```

### Ubicaciones de Archivos

| Archivo | Propósito |
|---------|-----------|
| `~/.claude/CLAUDE.md` | Reglas globales para todos los proyectos (instalado por `setup-global.sh`) |
| `<proyecto>/CLAUDE.md` | Reglas específicas del proyecto (creado por `new-project.sh` o manualmente) |
| `<template>/docs/skills/*.md` | 19 archivos de skills especializados |
| `<template>/scripts/setup-global.sh` | Instala configuración global |
| `<template>/scripts/new-project.sh` | Crea nuevo proyecto desde template |

### Skills Incluidos (19)

| Skill | Archivo |
|-------|---------|
| Software Architecture | `software_architecture.md` |
| Security Engineering | `security.md` |
| GenAI & RAG | `genai_rag.md` |
| Multi-Agent Systems | `multi_agent_systems.md` |
| Data & ML Engineering | `data_ml_engineering.md` |
| API & Streaming | `api_streaming.md` |
| Cloud & Infrastructure | `cloud_infrastructure.md` |
| Testing & Quality | `testing_quality.md` |
| Observability & Monitoring | `observability_monitoring.md` |
| Context Engineering | `context_engineering.md` |
| Prompt Engineering | `prompt_engineering.md` |
| Multi-Tenancy | `multi_tenancy.md` |
| Hallucination Detection | `hallucination_detection.md` |
| Automation & Scripting | `automation.md` |
| Analytics & Reporting | `analytics.md` |
| Databases & Storage | `databases.md` |
| Event-Driven Systems | `event_driven_systems.md` |
| Model Context Protocol | `mcp.md` |
| AI Governance | `governance.md` |

### Validación de Skills

Antes de desarrollo importante, valida que los skills estén actualizados:

```bash
# Revisar el proceso de validación
cat docs/SKILL_VALIDATION_PROMPT.md
```

Ver [SKILL_VALIDATION_PROMPT.md](docs/SKILL_VALIDATION_PROMPT.md) para el proceso completo.

---

## Solución de Problemas

### Claude Code no parece seguir las reglas
- Verifica que `~/.claude/CLAUDE.md` existe: `cat ~/.claude/CLAUDE.md | head -5`
- Verifica que los skills son accesibles: `ls ~/templates/genai-python-template/docs/skills/`
- Si moviste el template, ejecuta `./scripts/setup-global.sh --force`

### Los skills no están siendo consultados
- Claude Code lee skills bajo demanda cuando la tarea es relevante
- Puedes pedirlo explícitamente: "Consulta el skill de Security antes de revisar este código"

### Conflicto entre reglas globales y del proyecto
- Las reglas del proyecto tienen prioridad cuando hay contradicción directa
- En la práctica, global = principios universales, proyecto = contexto específico
- Se complementan entre sí, los conflictos son raros
