# Prompt de Validación de Skills para Claude Code

Este prompt debe ser ejecutado por una IA para validar que todos los skills sean consistentes con el CLAUDE.md base y no introduzcan riesgos de alucinación.

---

## PROMPT DE VALIDACIÓN

```
Actúa como un **Senior AI Engineer** especializado en prevención de alucinaciones en sistemas LLM. Tu misión es revisar skills de Claude Code (.md) para verificar que sean consistentes con el CLAUDE.md base y no introduzcan riesgos de alucinación cuando Claude genere código Python backend para sistemas GenAI.

## CONTEXTO: CLAUDE.md Base (Resumen de Políticas Clave)

El proyecto usa este archivo base con reglas estrictas anti-alucinación:

### 🔴 Reglas Absolutas del Base
1. **Nunca inventar APIs, métodos o funciones** - Si no hay 100% certeza, declarar "Verificar en documentación: [link]"
2. **Versionado explícito obligatorio** - Especificar versiones mínimas, marcar APIs inestables con `⚠️`
3. **Honestidad epistémica** - "No estoy seguro de [X], verificar en [fuente]"

### 📋 Librerías Marcadas como Inestables (Requieren Verificación)
| Librería | Estabilidad | Acción Requerida |
|----------|-------------|------------------|
| langchain | ⚠️ Cambia frecuentemente | Verificar imports |
| langchain_experimental | ❌ Muy inestable | Verificar existencia de clases |
| langgraph | ⚠️ Estabilizándose | Verificar changelog |
| deepeval | ⚠️ API cambia | Verificar docs antes de usar |
| guardrails-ai | ⚠️ Inestable | Verificar docs actuales |
| crewai | ⚠️ En desarrollo | Verificar versión |

### 🏗️ Arquitectura Base (Clean Architecture)
- `domain`: pura lógica de negocio, sin frameworks
- `application`: casos de uso y orquestación
- `infrastructure`: sistemas externos (LLMs, DBs, cloud, MCP)
- `interfaces`: APIs, CLI, controllers
- **Nunca** filtrar infraestructura al dominio

### ⚡ Async Obligatorio
- Todas las llamadas LLM deben ser async (`await`)
- Usar `asyncio.gather` para paralelismo
- Usar `asyncio.Semaphore` para rate limiting
- **Nunca** bloquear el event loop con I/O síncrono

---

## INSTRUCCIONES DE REVISIÓN

Para cada skill proporcionado, evalúa:

### 1. CONSISTENCIA CON CLAUDE.md BASE
- [ ] ¿Respeta la estructura de Clean Architecture (domain/application/infra/interfaces)?
- [ ] ¿Mantiene el principio "domain never depends on infrastructure"?
- [ ] ¿Usa async/await consistentemente para operaciones I/O?
- [ ] ¿Sigue las convenciones de nombres y patrones del base?

### 2. CUMPLIMIENTO DE POLÍTICAS ANTI-ALUCINACIÓN
- [ ] **¿Prohíbe explícitamente inventar APIs?** Debe tener equivalente a: "Si no estás 100% seguro → declarar incertidumbre"
- [ ] **¿Requiere versionado explícito?** Debe especificar versiones mínimas de dependencias
- [ ] **¿Marca APIs inestables?** Debe usar `⚠️` o `❌` según tabla del base
- [ ] **¿Promueve honestidad epistémica?** Debe forzar a Claude a admitir cuando no sabe

### 3. MANEJO DE LIBRERÍAS INESTABLES
Para cada librería inestable mencionada en el skill:
- [ ] ¿Verifica que los imports existan antes de usarlos?
- [ ] ¿Incluye bloques try/except para ImportError con mensajes descriptivos?
- [ ] ¿Referencias a documentación oficial actualizada?
- [ ] ¿Advertencias sobre breaking changes conocidos?

### 4. PREVENCIÓN DE ALUCINACIONES ESPECÍFICAS EN CÓDIGO

#### A. Python / FastAPI
- [ ] ¿Evita inventar parámetros en funciones de librerías estándar?
- [ ] ¿Verifica que los decorators de FastAPI existan (@app.get, @app.post)?
- [ ] ¿No asume comportamientos de SQLAlchemy 2.0 vs 1.x sin verificar?

#### B. LLM / GenAI Frameworks
- [ ] ¿No inventa métodos en LangChain/LangGraph (ej: `chain.run()` vs `chain.invoke()`)?
- [ ] ¿Verifica que los nodos de LangGraph tengan la signature correcta?
- [ ] ¿No asume que CrewAI/AutoGen tienen APIs que cambiaron recientemente?

#### C. Pydantic / Structured Output
- [ ] ¿Usa `model_validate` vs `parse_obj` según versión de Pydantic v2?
- [ ] ¿Verifica que los tipos de Instructor sean compatibles con la versión?

### 5. PATRONES DE SEGURIDAD ANTI-HALLUCINACIÓN

Verifica que el skill incluya:

```python
# Patrón de respuesta segura (del base)
try:
    from libreria_inestable.modulo import Clase
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False

if not FEATURE_AVAILABLE:
    raise RuntimeError("Requiere X. Verificar versión compatible.")
```

- [ ] ¿Incluye este patrón o equivalente para dependencias inestables?
- [ ] ¿Nunca hardcodea precios de LLM, dimensiones de embeddings, URLs de APIs?
- [ ] ¿Usa variables de entorno para toda configuración sensible?

### 6. CALIDAD DE EJEMPLOS DE CÓDIGO
- [ ] ¿Los ejemplos son verificables? (pueden ejecutarse copiando y pegando)
- [ ] ¿Incluyen imports explícitos al inicio?
- [ ] ¿Especifican versiones de dependencias en comentarios?
- [ ] ¿Evitan "..." o "# implementar aquí" en código crítico?

---

## FORMATO DE RESPUESTA

Para cada skill revisado, entrega:

### 📊 SCORECARD DE CONFIABILIDAD

| Criterio | Score (1-10) | Evidencia |
|----------|--------------|-----------|
| Consistencia con Base | X/10 | [Cita específica del skill] |
| Cumplimiento Anti-Alucinación | X/10 | [Cita específica del skill] |
| Manejo de Librerías Inestables | X/10 | [Cita específica del skill] |
| Calidad de Ejemplos | X/10 | [Cita específica del skill] |
| **PROMEDIO GENERAL** | **X/10** | |

### 🚨 HALLAZGOS CRÍTICOS (Bloqueantes)

Lista de problemas que deben corregirse antes de usar el skill:

```
[CRÍTICO] [Descripción del problema] → [Línea o sección específica]
Riesgo: [Qué podría alucinar Claude]
Fix sugerido: [Texto exacto a cambiar]
```

### ⚠️ ADVERTENCIAS (No bloqueantes pero riesgosas)

```
[ADVERTENCIA] [Descripción] → [Ubicación]
Mitigación sugerida: [Acción recomendada]
```

### 🔧 RECOMENDACIONES DE MEJORA

Cambios opcionales para robustez adicional:
- [Sugerencia específica con ejemplo de redacción]

### ✅ CHECKLIST DE VALIDACIÓN FINAL

Antes de aprobar el skill, verificar:
- [ ] Todas las librerías inestables tienen warnings
- [ ] Los ejemplos de código incluyen versiones de dependencias
- [ ] Hay al menos un patrón de "honestidad epistémica" explícito
- [ ] El skill no contradice ninguna regla del CLAUDE.md base
- [ ] Async/await se usa correctamente en operaciones I/O

---

## EJEMPLO DE OUTPUT ESPERADO

**Skill revisado:** `docs/skills/multi_agent_systems.md`
**Score General:** 7.5/10

**Hallazgo Crítico:**
LangGraph menciona `StateGraph.add_node()` sin verificar versión. En 0.1.x vs 0.2.x cambia la API de compilación.
**Fix:** Agregar "⚠️ Verificar versión de LangGraph. En 0.2.x usar `graph.compile()` vs `graph.run()`"

**Advertencia:**
CrewAI no tiene disclaimer de inestabilidad pese a estar en la tabla del base.

**Recomendación:**
Agregar bloque de verificación de imports para langgraph al inicio de ejemplos.

---

## CONTEXTO ADICIONAL

El usuario aplica rigor similar al que usa para evaluar sesgos en análisis económicos. Valora:
- **Transparencia** cuando la IA no sabe algo
- **Precisión técnica** sobre velocidad de respuesta
- **Verificabilidad** de todo código generado
```

---

## LISTA DE SKILLS A VALIDAR

Ejecutar el prompt anterior para cada uno de estos archivos:

1. `docs/skills/software_architecture.md`
2. `docs/skills/security.md`
3. `docs/skills/genai_rag.md`
4. `docs/skills/multi_agent_systems.md`
5. `docs/skills/data_ml_engineering.md`
6. `docs/skills/api_streaming.md`
7. `docs/skills/cloud_infrastructure.md`
8. `docs/skills/testing_quality.md`
9. `docs/skills/observability_monitoring.md`
10. `docs/skills/databases.md`
11. `docs/skills/event_driven_systems.md`
12. `docs/skills/mcp.md`
13. `docs/skills/governance.md`
14. `docs/skills/context_engineering.md`
15. `docs/skills/prompt_engineering.md`
16. `docs/skills/hallucination_detection.md`
17. `docs/skills/multi_tenancy.md`
18. `docs/skills/automation.md`
19. `docs/skills/analytics.md`

---

## ENTREGABLES ESPERADOS

Al finalizar la validación de todos los skills:

### 1. Tabla Resumen de Scores

| Skill | Score | Críticos | Advertencias | Estado |
|-------|-------|----------|--------------|--------|
| software_architecture.md | X/10 | N | N | ✅/⚠️/❌ |
| security.md | X/10 | N | N | ✅/⚠️/❌ |
| ... | ... | ... | ... | ... |

### 2. Lista Consolidada de Fixes

Todos los hallazgos críticos agrupados por prioridad:
- **P0 (Bloqueante):** [Lista]
- **P1 (Importante):** [Lista]
- **P2 (Mejora):** [Lista]

### 3. Prompts de Remediación

Para cada hallazgo crítico, generar un prompt específico que otra IA pueda ejecutar para corregirlo.

Formato:
```
### Fix para [Skill]: [Descripción corta]

Lee el archivo `docs/skills/[skill].md` y realiza el siguiente cambio:

**Ubicación:** [Sección o línea]
**Cambio:** [Instrucción específica]
**Texto actual:** [Si aplica]
**Texto nuevo:** [Texto exacto a insertar]

No modifiques ningún otro contenido del archivo.
```
