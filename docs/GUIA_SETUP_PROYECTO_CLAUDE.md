# Guia de Setup de Proyecto con Claude Code

**Proposito:** Este documento explica como esta configurado el sistema de CLAUDE.md,
skills e historias de usuario para que Claude Code genere codigo de alta calidad
sin inventar ni alucinar. Compartir este archivo al inicio de cada proyecto nuevo.

---

## 1. Sistema de Dos Niveles de CLAUDE.md

Claude Code lee archivos `CLAUDE.md` automaticamente. Tenemos **dos niveles** que se
complementan:

### Nivel 1: Global (`~/.claude/CLAUDE.md`)

**Ubicacion:** `~/.claude/CLAUDE.md`
**Se carga:** SIEMPRE, en CUALQUIER directorio, en CUALQUIER proyecto.
**Contiene:** Reglas de CALIDAD y HOW (como escribir codigo).

Incluye:
- Rol: Senior/Staff Python Backend Engineer
- Principios SOLID y Clean Architecture
- Patrones de diseno (Strategy, Factory, Repository, etc.)
- Reglas de seguridad (OWASP, nunca hardcodear secrets)
- Stack preferido (FastAPI, SQLAlchemy, Pydantic, etc.)
- Reglas de async, refactoring, testing
- Dependencias via `uv`
- Quality gates (ruff, mypy, coverage >80%)

**Referencias a Skills globales** (archivos en `~/templates/genai-python-template/docs/skills/`):
- `software_architecture.md` — Clean Architecture, SOLID, C4 Model
- `security.md` — OWASP Top 10, API security, LLM safety
- `genai_rag.md` — RAG, embeddings, chunking
- `multi_agent_systems.md` — LangGraph, CrewAI, coordinacion
- `data_ml_engineering.md` — ETL, MLflow, data quality
- `api_streaming.md` — SSE, WebSockets, async patterns
- `cloud_infrastructure.md` — Docker, K8s, CI/CD
- `testing_quality.md` — pytest, Hypothesis, coverage
- `observability_monitoring.md` — OpenTelemetry, Prometheus
- `databases.md` — SQLAlchemy 2.0, Alembic, asyncpg
- `event_driven_systems.md` — Celery, Kafka, outbox/saga
- `prompt_engineering.md` — Templates, CoT, few-shot
- `context_engineering.md` — Semantic blueprints, dual-RAG
- `hallucination_detection.md` — Self-correction, confidence
- `multi_tenancy.md` — RBAC, tenant isolation
- `mcp.md` — MCP servers/clients
- `governance.md` — GDPR, EU AI Act, audit trails
- `automation.md` — CLI tools, Typer, scheduling
- `analytics.md` — Dashboards, cost tracking

Claude lee estos skills automaticamente cuando son relevantes a la tarea.
**No necesitas mencionarlos** — el global CLAUDE.md ya los referencia.

### Nivel 2: Proyecto (`<directorio-proyecto>/CLAUDE.md`)

**Ubicacion:** Raiz del proyecto (junto a `pyproject.toml` o `requirements.txt`)
**Se carga:** Solo cuando Claude Code esta DENTRO de ese directorio.
**Contiene:** CONTEXTO del proyecto y WHAT (que construir).

Incluye:
- Descripcion del proyecto
- Stack tecnologico especifico
- Estructura de carpetas del backend
- Roles y permisos del sistema
- Modelo de datos (tablas, relaciones)
- **Referencias a las historias de usuario** (la parte mas importante)
- Reglas especificas del proyecto (convenciones, seguridad, API, tests)
- Orden de implementacion de modulos

### Como se combinan

```
Claude Code abre un proyecto
    │
    ├── Lee ~/.claude/CLAUDE.md (global)
    │   └── Sabe COMO escribir codigo de calidad
    │       (SOLID, patterns, security, async, tests...)
    │
    └── Lee <proyecto>/CLAUDE.md (proyecto)
        └── Sabe QUE construir
            (endpoints, schemas, validaciones, tablas...)
            │
            └── Ve referencia a docs/stories/M1_AUTH.md
                └── Lee la historia ANTES de codificar
                    └── Implementa EXACTAMENTE lo especificado
```

**Resultado:** Claude no inventa endpoints, no alucina validaciones, no asume
schemas. Todo esta especificado en las historias y sigue las reglas de calidad
del global.

---

## 2. Estructura de Archivos por Proyecto

Cada proyecto nuevo debe tener esta estructura de documentacion:

```
mi-proyecto/
├── CLAUDE.md                        # Contexto del proyecto (nivel 2)
├── SRS_*.md                         # Documento de requerimientos (opcional)
├── docs/
│   └── stories/                     # Historias de usuario por modulo
│       ├── M1_NOMBRE_MODULO.md
│       ├── M2_NOMBRE_MODULO.md
│       ├── M3_NOMBRE_MODULO.md
│       └── ...
├── backend/                         # Codigo (se genera despues)
│   ├── app/
│   ├── tests/
│   └── ...
└── .env.example
```

---

## 3. Formato del CLAUDE.md del Proyecto

El CLAUDE.md del proyecto debe contener estas secciones:

```markdown
# NombreProyecto — Descripcion corta

## Descripcion del Proyecto
Parrafo breve de que hace el sistema.

## Stack Tecnologico
### Backend
- Framework, ORM, Auth, Jobs, etc.
### Base de Datos
- BD principal, cache, storage
### Integraciones
- APIs externas

## Estructura del Backend
Arbol de carpetas con descripcion de cada capa.

## Roles del Sistema
Tabla con roles y claves en BD.

## Modelo de Datos (N tablas)
Tabla con nombre y descripcion de cada tabla.
Referencia al SRS para esquemas SQL completos.

## Historias de Usuario por Modulo (Backend)
**IMPORTANTE:** Antes de codificar cualquier modulo, DEBES leer la historia.

### [M1: Nombre](docs/stories/M1_NOMBRE.md)
- Bullet points resumiendo lo que cubre

### [M2: Nombre](docs/stories/M2_NOMBRE.md)
- Bullet points...

(repetir por cada modulo)

## Reglas Especificas del Proyecto
### Convenciones de Codigo
### Seguridad
### API
### Tests
### Orden de Implementacion
```

**Regla clave:** La seccion "Historias de Usuario" es la que conecta el CLAUDE.md
con los archivos de stories. Claude ve el link, lee el archivo, y codifica
exactamente lo que dice.

---

## 4. Formato de las Historias de Usuario

Cada archivo de historia (`docs/stories/MX_NOMBRE.md`) sigue el formato INVEST
con criterios DADO/CUANDO/ENTONCES:

```markdown
# MX: Nombre del Modulo — Historias de Usuario Backend

**Proyecto:** NombreProyecto
**Modulo:** MX - Nombre
**Tablas:** `tabla1`, `tabla2`
**Depende de:** M1 (Auth), MN (otro)

---

## HU-MX-01: Titulo de la Historia

**Como** [rol]
**Quiero** [accion]
**Para** [beneficio]

### Endpoint

POST /api/v1/recurso

**Request:**
(JSON de ejemplo)

**Response 2XX:**
(JSON de ejemplo)

### Schema Pydantic

**NombreCreate:**
- `campo: tipo` (requerido/opcional, validaciones)

**NombreResponse:**
- `campo: tipo`

### Criterios de Aceptacion

1. **DADO** condicion **CUANDO** accion **ENTONCES** resultado esperado
2. **DADO** error X **ENTONCES** codigo HTTP + mensaje en espanol
3. (todos los casos de exito y error)

### Autorizacion (si aplica)
Tabla de roles y que puede hacer cada uno.

---

(repetir por cada historia del modulo)

## Tabla Resumen de Endpoints
Tabla con metodo, path, auth, descripcion.

## Orden de Implementacion
Secuencia recomendada dentro del modulo.

## Tests Esperados
Tabla con archivo de test y casos minimos.
```

### Elementos criticos de cada historia:

| Elemento | Por que es necesario | Claude lo usa para... |
|----------|---------------------|----------------------|
| Endpoint exacto | Evita que invente rutas | Crear el router con la ruta correcta |
| Request/Response JSON | Define el contrato | Generar schemas Pydantic exactos |
| Schemas Pydantic | Tipos y validaciones | Crear los schemas sin adivinar campos |
| Criterios DADO/ENTONCES | Todos los casos | Implementar validaciones y escribir tests |
| Mensajes de error | En espanol, exactos | Usar esos textos literales en HTTPException |
| Autorizacion por rol | Quien puede que | Implementar decoradores/dependencies |
| Efectos secundarios | Emails, notificaciones | No olvidar integraciones |

---

## 5. Workflow para Proyecto Nuevo

### Paso 1: Definir requerimientos
- Tener un SRS, documento de diseno, o al menos wireframes/flujos claros
- Identificar modulos, tablas, roles, integraciones

### Paso 2: Crear estructura de documentacion

```bash
mkdir -p mi-proyecto/docs/stories
```

### Paso 3: Compartir contexto a Claude Code
Abrir sesion y proporcionar:
1. Este archivo (`GUIA_SETUP_PROYECTO_CLAUDE.md`) para que entienda el sistema
2. El SRS o documento de requerimientos
3. Wireframes/imagenes si hay

### Paso 4: Pedirle a Claude que genere
1. **Primero** el `CLAUDE.md` del proyecto (contexto general)
2. **Despues** las historias por modulo en `docs/stories/`

### Paso 5: Implementar
- Claude lee el CLAUDE.md, ve las referencias
- Le pides "implementa M1" → lee `docs/stories/M1_AUTH.md` → codifica
- Avanza modulo por modulo en orden

---

## 6. Datos para Compartir al Inicio de Sesion Nueva

Al abrir una sesion nueva de Claude Code, comparte:

1. **Ruta del global CLAUDE.md:** `~/.claude/CLAUDE.md` (ya se carga solo)
2. **Ruta de skills globales:** `~/templates/genai-python-template/docs/skills/`
3. **Esta guia:** para que Claude entienda el sistema de dos niveles
4. **El SRS o requerimientos** del proyecto nuevo
5. **Instruccion clara:**
   ```
   Revisa esta guia para entender como trabajo con CLAUDE.md y stories.
   El global ya esta instalado en ~/.claude/CLAUDE.md con skills de calidad.
   Necesito que crees:
   1. El CLAUDE.md del proyecto en la raiz
   2. Las historias de usuario en docs/stories/ (una por modulo)
   Solo backend, zero codigo, formato INVEST con DADO/CUANDO/ENTONCES.
   ```

---

## 7. Ejemplo Real: TalentFinder

Proyecto completado con este sistema:

```
/home/leo/nivelics /Reclutamiento_backend/
├── CLAUDE.md                          # 222 lineas, contexto completo
├── SRS_TalentFinder_*.md              # ~3400 lineas, requerimientos
└── docs/stories/
    ├── M1_AUTH.md                     # 10 historias, 7 endpoints
    ├── M2_DASHBOARD.md                # 4 historias, 3 endpoints
    ├── M3_CANDIDATOS.md               # 10 historias, 10 endpoints
    ├── M4_PROCESOS.md                 # 14 historias, 14 endpoints
    ├── M5_ENTREVISTAS.md              # 6 historias, 5 endpoints
    ├── M6_NOTIFICACIONES.md           # 7 historias, 4 endpoints + WS
    └── M7_IA_LINKEDIN.md             # 7 historias, 4 endpoints + Celery
```

**Total:** 58 historias, ~45 endpoints, todo especificado antes de escribir
una sola linea de codigo.

---

## 8. Reglas Importantes

1. **Nunca codificar sin historia.** Claude debe leer la historia antes de implementar.
2. **Orden de modulos.** Respetar dependencias (M1 antes que M2, etc.).
3. **Mensajes en espanol.** Todos los mensajes de error de la API en espanol.
4. **Global no se toca.** El `~/.claude/CLAUDE.md` solo se modifica con `setup-global.sh`.
5. **Historias son contratos.** Si hay que cambiar algo, se actualiza la historia primero, despues el codigo.
6. **Un archivo por modulo.** Cada modulo tiene su propio archivo de historias.
7. **FastAPI route ordering.** Rutas estaticas ANTES de parametrizadas (`/unread-count` antes de `/{id}`).
8. **Schemas Pydantic en cada historia.** Nunca dejar que Claude invente campos.
