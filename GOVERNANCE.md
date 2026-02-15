# AI Governance

## Responsible AI

### Principios

1. **Transparencia**: El usuario sabe que interactúa con AI
2. **Fairness**: Sin sesgos discriminatorios en outputs
3. **Accountability**: Decisiones de AI son auditables
4. **Privacy**: Datos personales protegidos por diseño
5. **Safety**: El sistema no genera contenido dañino
6. **Human oversight**: Humano en el loop para decisiones críticas

---

## Compliance

### Regulaciones Relevantes

| Regulación | Ámbito | Requisitos clave |
|------------|--------|-----------------|
| EU AI Act | Europa | Clasificación de riesgo, transparencia, auditoría |
| GDPR | Europa | Protección de datos, derecho al olvido, consentimiento |
| CCPA/CPRA | California | Privacy rights, opt-out |
| SOC 2 | Global | Controles de seguridad, disponibilidad, confidencialidad |

### Clasificación de Riesgo (EU AI Act)

- **Riesgo inaceptable**: Sistemas prohibidos (social scoring, manipulación)
- **Riesgo alto**: Requieren conformidad estricta (healthcare, legal, HR)
- **Riesgo limitado**: Obligaciones de transparencia
- **Riesgo mínimo**: Sin obligaciones específicas

---

## Data Privacy

### Principios

- **Data minimization**: Solo recoger datos necesarios
- **Purpose limitation**: Datos usados solo para el fin declarado
- **Storage limitation**: Retención con tiempo definido
- **PII handling**: Detectar, enmascarar o eliminar PII

```python
# PII Detection
import presidio_analyzer

analyzer = presidio_analyzer.AnalyzerEngine()

def detect_pii(text: str) -> list[str]:
    results = analyzer.analyze(text=text, language="en")
    return [r.entity_type for r in results]

# PII Masking en logs
def mask_pii(text: str) -> str:
    """Mask PII before logging."""
    anonymizer = presidio_anonymizer.AnonymizerEngine()
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text
```

### Data Flow

```
User Input → PII Detection → Processing → PII Masking → Logging
                                ↓
                            LLM Call (sin PII cuando posible)
                                ↓
                            Response → PII Check → User
```

---

## Model Cards

Documentación estandarizada de modelos usados en el sistema.

```yaml
# model_cards/gpt-4o.yaml
model:
  name: GPT-4o
  provider: OpenAI
  version: "2024-08-06"
  type: Large Language Model

intended_use:
  primary: "Document summarization and Q&A"
  out_of_scope: "Medical diagnosis, legal advice"

limitations:
  - "May hallucinate facts not present in context"
  - "Knowledge cutoff: April 2024"
  - "May exhibit biases present in training data"

evaluation:
  metrics:
    faithfulness: 0.89
    answer_relevancy: 0.92
    hallucination_rate: 0.08
  dataset: "internal_eval_v2"
  date: "2025-01"

ethical_considerations:
  - "Outputs must be validated before presenting as facts"
  - "Not suitable for high-stakes decisions without human review"
  - "PII filtering must be active in production"
```

---

## Audit Trails

### Qué registrar

| Evento | Datos | Retención |
|--------|-------|-----------|
| LLM invocation | Model, tokens, latency, cost | 90 días |
| Tool execution | Tool name, input hash, output hash | 90 días |
| User interaction | Session ID, query hash, timestamp | 30 días |
| Agent decision | Decision type, reasoning, outcome | 90 días |
| Access to data | User, resource, action, timestamp | 1 año |
| Configuration change | Who, what, when, previous value | 1 año |

### Implementación

```python
import structlog

audit_logger = structlog.get_logger("audit")

async def log_llm_call(
    model: str,
    prompt_hash: str,
    tokens: int,
    cost: float,
    user_id: str,
) -> None:
    audit_logger.info(
        "llm_invocation",
        model=model,
        prompt_hash=prompt_hash,  # Hash, no el prompt completo
        tokens=tokens,
        cost_usd=cost,
        user_id=user_id,
        timestamp=datetime.utcnow().isoformat(),
    )
```

**Reglas de audit:**
- Nunca loggear contenido raw de prompts con datos sensibles
- Usar hashes para referencia sin exposición
- Logs inmutables (append-only)
- Acceso a audit logs restringido

---

## Consent Management

Tracking de qué datos el usuario autorizó y para qué propósito.

En sistemas GenAI, los datos del usuario pueden fluir a: contexto del modelo, índices RAG,
datasets de fine-tuning, logs y analytics. Cada uso requiere consentimiento explícito y granular.

### Propósitos de Procesamiento

| Propósito | Descripción | Requiere consentimiento explícito |
|-----------|-------------|----------------------------------|
| `model_context` | Datos enviados al LLM como contexto | Sí |
| `rag_indexing` | Datos indexados en vector store | Sí |
| `fine_tuning` | Datos usados para entrenar modelo | Sí (consentimiento separado) |
| `analytics` | Analytics agregados de uso | Depende (interés legítimo posible) |
| `feedback_training` | Feedback del usuario para mejorar modelo | Sí |
| `logging` | Datos retenidos en logs operacionales | Depende (interés legítimo posible) |

### Reglas

- Verificar consentimiento antes de procesar datos para cada propósito
- Default a denegado si no existe registro de consentimiento
- Consentimiento es retirable en cualquier momento
- Registros de consentimiento exportables para auditoría regulatoria

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#consent-management)

---

## Data Subject Rights (GDPR / CCPA)

Sistemas GenAI deben soportar el ciclo completo de derechos del titular de datos:

| Derecho | GDPR Artículo | Implementación |
|---------|---------------|----------------|
| Acceso | Art. 15 | Exportar todos los datos del usuario de TODOS los stores |
| Eliminación | Art. 17 | Borrar de: DB, vector store, cache, logs |
| Portabilidad | Art. 20 | Exportar en formato machine-readable (JSON) |
| Rectificación | Art. 16 | Corregir datos inexactos |
| Objeción | Art. 21 | Detener procesamiento para propósito específico |

**Crítico**: La eliminación debe cubrir TODOS los data stores:
- Base de datos relacional
- Vector store (embeddings del usuario)
- Cache semántico
- Logs (dentro de política de retención)
- Datasets del LLM provider (si se envió para fine-tuning)

Responder a solicitudes dentro de **30 días calendario** (GDPR).

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#data-subject-rights-gdpr--ccpa)

---

## Data Residency

Las reglas de residencia dictan **dónde** los datos pueden ser almacenados y procesados.

- Las llamadas a APIs de LLM envían datos a proveedores externos, potencialmente cross-border
- Usar un router que mapee políticas de tenant a endpoints en regiones permitidas
- Si no existe endpoint compliant: **fallar ruidosamente**, nunca caer silenciosamente a región no permitida
- Documentar qué proveedores tienen endpoints en qué regiones

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#data-residency-and-sovereignty)

---

## Compliance por Industria

### Healthcare (HIPAA)

- BAA (Business Associate Agreement) obligatorio con LLM provider
- Regla de mínimo necesario: solo enviar PHI necesario para la tarea
- De-identificar antes de enviar al LLM cuando sea posible
- Auditar cada acceso a PHI
- Encriptar at rest y in transit

### Financial (PCI-DSS / SOX)

- Nunca enviar datos de tarjeta al LLM (PCI-DSS)
- Segregación de funciones: AI no puede recomendar Y aprobar (SOX)
- Retención de audit trails: 7 años mínimo (SOX)
- Validar que no se filtren números de cuenta en outputs

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#industry-specific-compliance)

---

## Data Retention

| Categoría | Retención | Estrategia |
|-----------|-----------|------------|
| LLM invocations | 90 días | Hard delete |
| User queries | 30 días | Anonymize |
| Semantic cache | 7 días | Hard delete |
| User embeddings | 365 días | Hard delete |
| Audit trails | 7 años | Legal hold |

- Ejecutar limpieza automatizada con schedule
- Respetar legal holds (no borrar aunque pase el periodo)
- Documentar resultados de cada ejecución

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#data-retention-and-lifecycle)

---

## Content Safety

- Filtrar TODOS los outputs del LLM antes de retornar al usuario
- Detectar: API keys, SSNs, ecos de prompt injection, lenguaje tóxico
- Sanitizar o rechazar outputs inseguros
- Para alto riesgo: usar LLM-as-judge como segunda capa

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#content-safety-and-output-filtering)

---

## Explainability

- Proveer **source attribution** para respuestas RAG (document ID, relevance score)
- Incluir **confidence score** en cada respuesta
- Agregar **caveats** cuando el knowledge base puede estar desactualizado
- Mantener **reasoning trace** para decisiones de agentes

Ver código completo en: [docs/skills/governance.md](docs/skills/governance.md#explainability-for-genai)

---

## Checklist de Governance

### Privacy & PII
- [ ] PII detection y masking activos en boundaries
- [ ] Raw PII nunca aparece en logs, traces o errores
- [ ] Data retention policies definidas y automatizadas
- [ ] Data subject requests (acceso, eliminación) manejados en TODOS los stores
- [ ] Consent management con tracking por propósito

### Compliance
- [ ] Clasificación de riesgo EU AI Act documentada
- [ ] GDPR/CCPA data processing agreements en lugar
- [ ] Compliance por industria validado (HIPAA/PCI-DSS/SOX si aplica)
- [ ] Data residency policies enforced con routing por región
- [ ] Registros de consentimiento exportables para auditoría

### Transparencia & Explicabilidad
- [ ] Model cards para todos los componentes AI
- [ ] Uso de AI divulgado a usuarios
- [ ] Versiones de prompts tracked y vinculados a deployments
- [ ] Source attribution en respuestas RAG
- [ ] Confidence scores incluidos en respuestas

### Fairness & Bias
- [ ] Bias evaluation antes de cada deployment
- [ ] Atributos protegidos identificados y testeados
- [ ] Métricas de fairness con umbrales pass/fail
- [ ] Re-evaluación periódica (mínimo trimestral)

### Content Safety
- [ ] Output safety guard activo en boundaries
- [ ] Lista de patrones bloqueados mantenida
- [ ] Sanitización aplicada antes de retornar outputs flaggeados

### Operacional
- [ ] Audit trail captura todas las decisiones AI con correlation IDs
- [ ] Mecanismo de feedback desplegado en todas las interfaces AI
- [ ] Human-in-the-loop para decisiones de alto impacto
- [ ] Governance dashboard con métricas clave
- [ ] Incident response plan para fallos de AI
- [ ] Training del equipo en responsible AI

Ver también: [SECURITY.md](SECURITY.md), [OBSERVABILITY.md](OBSERVABILITY.md)
