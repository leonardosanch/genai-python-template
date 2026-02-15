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

## Checklist de Governance

- [ ] Model cards documentados para cada modelo en uso
- [ ] PII detection y masking activos
- [ ] Audit trails configurados
- [ ] Data retention policies definidas
- [ ] Human-in-the-loop para decisiones críticas
- [ ] Bias testing periódico
- [ ] Transparencia: usuarios saben que interactúan con AI
- [ ] Clasificación de riesgo del sistema documentada
- [ ] Incident response plan para fallos de AI
- [ ] Training del equipo en responsible AI

Ver también: [SECURITY.md](SECURITY.md), [OBSERVABILITY.md](OBSERVABILITY.md)
