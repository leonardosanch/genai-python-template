# Architecture Decision Records (ADRs)

Registro de decisiones arquitectónicas significativas del proyecto.

Formato: [MADR](https://adr.github.io/madr/) simplificado.

---

## ADR-001: Clean Architecture como estilo arquitectónico

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
Los sistemas GenAI tienden a acoplarse fuertemente a proveedores de LLM, frameworks y servicios cloud. Necesitamos una arquitectura que permita cambiar proveedores sin afectar la lógica de negocio.

**Decisión:**
Adoptar Clean Architecture con 4 capas: domain, application, infrastructure, interfaces.

**Consecuencias:**
- (+) Dominio testeable sin dependencias externas
- (+) Proveedores de LLM intercambiables
- (+) Separación clara de responsabilidades
- (-) Más archivos y boilerplate inicial
- (-) Curva de aprendizaje para desarrolladores no familiarizados

---

## ADR-002: uv como gestor de dependencias y entornos virtuales

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
pip + venv es lento y carece de resolución de dependencias robusta. Poetry tiene overhead innecesario. uv es un drop-in replacement significativamente más rápido.

**Decisión:**
Usar uv para instalación de dependencias, gestión de virtualenvs y ejecución de scripts.

**Alternativas consideradas:**
- pip + venv: Lento, sin lock file nativo
- Poetry: Más lento que uv, resolución de dependencias menos eficiente
- PDM: Menos adopción en la comunidad

**Consecuencias:**
- (+) Instalaciones 10-100x más rápidas
- (+) Lock file nativo (`uv.lock`)
- (+) Compatible con `pyproject.toml` estándar
- (-) Herramienta relativamente nueva

---

## ADR-003: Abstracción de proveedores LLM detrás de ports

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
El proyecto debe soportar múltiples proveedores de LLM (OpenAI, Anthropic, Google, modelos locales) sin vendor lock-in.

**Decisión:**
Definir interfaces abstractas (ports) en el dominio. Cada proveedor tiene un adaptador en infrastructure.

**Consecuencias:**
- (+) Cambio de proveedor sin afectar lógica de negocio
- (+) Testing con mocks/stubs trivial
- (+) Soporte multi-modelo simultáneo
- (-) Cada proveedor nuevo requiere un adaptador

---

## ADR-004: Pydantic como estándar para structured output

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
Los LLMs producen texto no estructurado por defecto. Necesitamos validación y tipado de outputs.

**Decisión:**
Usar Pydantic models como schemas de salida. Integrar con Instructor para function calling y structured output.

**Consecuencias:**
- (+) Validación automática de outputs
- (+) Type safety en toda la cadena
- (+) Compatible con OpenAI function calling, Anthropic tool use
- (-) Overhead de definir schemas para cada output

---

## ADR-005: OpenTelemetry para observabilidad

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
Los sistemas GenAI requieren observabilidad específica: latencia de LLM, tokens consumidos, costos, trazas de agentes.

**Decisión:**
Adoptar OpenTelemetry como estándar de observabilidad. Structured logging con formato JSON.

**Consecuencias:**
- (+) Vendor-neutral (Datadog, Jaeger, Grafana)
- (+) Distributed tracing nativo
- (+) Métricas custom para LLM
- (-) Configuración inicial no trivial

---

## ADR-006: ruff + mypy para calidad de código

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
Necesitamos linting rápido y type checking estricto.

**Decisión:**
ruff para linting y formatting (reemplaza flake8, black, isort). mypy en modo strict para type checking.

**Consecuencias:**
- (+) ruff es 10-100x más rápido que alternativas
- (+) mypy strict previene errores en runtime
- (+) Una sola herramienta para lint + format
- (-) mypy strict puede ser verbose con libraries sin stubs

---

## ADR-007: pytest como framework de testing

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
Necesitamos un framework de testing flexible que soporte tests unitarios, de integración, y evaluación de LLMs.

**Decisión:**
pytest con plugins: pytest-asyncio, pytest-cov, pytest-mock.

**Consecuencias:**
- (+) Ecosistema de plugins extenso
- (+) Fixtures para setup/teardown
- (+) Soporte nativo para async
- (-) Ninguna significativa

---

## ADR-008: LangGraph para orquestación de agentes

**Estado:** Aceptada
**Fecha:** 2025-01

**Contexto:**
Los flujos multi-agente requieren orquestación con estado, condicionales y ciclos.

**Decisión:**
LangGraph como framework principal para orquestación de agentes. Máquinas de estado sobre flujos implícitos.

**Alternativas consideradas:**
- CrewAI: Más opinado, menos control fino
- AutoGen: API más compleja
- Custom: Mayor esfuerzo de mantenimiento

**Consecuencias:**
- (+) Grafos de estado explícitos y debuggeables
- (+) Checkpointing nativo
- (+) Integración con LangChain ecosystem
- (-) Acoplamiento a LangChain

---

## Plantilla para Nuevos ADRs

```markdown
## ADR-XXX: Título

**Estado:** Propuesta | Aceptada | Deprecada | Reemplazada
**Fecha:** YYYY-MM

**Contexto:**
¿Qué problema estamos resolviendo?

**Decisión:**
¿Qué decidimos?

**Alternativas consideradas:**
- Opción A: Razón de descarte
- Opción B: Razón de descarte

**Consecuencias:**
- (+) Positivas
- (-) Negativas
```
