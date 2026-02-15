# Observabilidad

## Stack

| Componente | Herramienta | Propósito |
|------------|------------|-----------|
| Tracing | OpenTelemetry | Distributed traces, spans de LLM |
| Logging | structlog | Structured logging JSON |
| Métricas | OpenTelemetry Metrics | Counters, histograms, gauges |
| Dashboards | Grafana | Visualización |
| Alerting | Grafana / PagerDuty | Alertas |
| LLM-specific | LangSmith / Langfuse | Trazas de LLM, prompt debugging |

---

## OpenTelemetry

### Setup

```python
# src/infrastructure/observability/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

def setup_tracing(service_name: str) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
```

### Tracing de LLM Calls

```python
tracer = trace.get_tracer(__name__)

async def generate(self, prompt: str, **kwargs) -> str:
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("llm.model", self._model)
        span.set_attribute("llm.prompt_length", len(prompt))

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        span.set_attribute("llm.response_length", len(response.choices[0].message.content))
        span.set_attribute("llm.tokens.prompt", response.usage.prompt_tokens)
        span.set_attribute("llm.tokens.completion", response.usage.completion_tokens)
        span.set_attribute("llm.tokens.total", response.usage.total_tokens)

        return response.choices[0].message.content
```

---

## Structured Logging

```python
# src/infrastructure/observability/logging.py
import structlog

def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )

logger = structlog.get_logger()

# Uso
logger.info("llm_call_completed",
    model="gpt-4o",
    tokens_used=150,
    latency_ms=320,
    cost_usd=0.0045,
)
```

**Reglas de logging:**
- Formato JSON siempre
- Nunca loggear secrets, PII, o prompts con datos sensibles
- Incluir correlation ID en toda request
- Niveles: DEBUG (desarrollo), INFO (producción), WARNING, ERROR

---

## Métricas LLM

### Métricas Clave

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `llm.requests.total` | Counter | Total de requests al LLM |
| `llm.requests.errors` | Counter | Requests fallidas |
| `llm.tokens.used` | Counter | Tokens consumidos (prompt + completion) |
| `llm.latency` | Histogram | Latencia de requests |
| `llm.cost.usd` | Counter | Costo acumulado |
| `rag.retrieval.latency` | Histogram | Latencia de retrieval |
| `rag.retrieval.documents` | Histogram | Documentos recuperados |
| `agent.steps.total` | Counter | Pasos de agente |

### Implementación

```python
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

llm_request_counter = meter.create_counter(
    "llm.requests.total",
    description="Total LLM requests",
)
llm_latency_histogram = meter.create_histogram(
    "llm.latency",
    description="LLM request latency in ms",
    unit="ms",
)
llm_token_counter = meter.create_counter(
    "llm.tokens.used",
    description="Total tokens consumed",
)
```

---

## Dashboards

### Dashboard LLM Recomendado

- **Requests/min** por modelo y estado (success/error)
- **Latencia P50/P95/P99** por modelo
- **Tokens/hora** (prompt vs completion)
- **Costo acumulado** por modelo y por use case
- **Error rate** por modelo y tipo de error
- **Token budget utilización** (% del presupuesto consumido)

### Dashboard de Agentes

- **Steps por task** (distribución)
- **Tool invocations** por tool y resultado
- **Agent completion rate**
- **Agent execution time** (distribución)

---

## Alerting

| Alerta | Condición | Severidad |
|--------|-----------|-----------|
| LLM Error Rate Alto | >5% errores en 5min | Critical |
| Latencia LLM Elevada | P95 > 10s por 5min | Warning |
| Token Budget Excedido | >90% del budget diario | Critical |
| Agent Loop Detectado | >50 steps en un task | Warning |
| API Rate Limit | >80% del rate limit | Warning |

---

## Integración con el Proyecto

```
src/infrastructure/observability/
├── __init__.py
├── logging.py       # structlog setup
├── tracing.py       # OpenTelemetry tracing
├── metrics.py       # OpenTelemetry metrics
└── middleware.py    # FastAPI middleware para tracing
```

Ver también: [TOOLS.md](TOOLS.md), [DEPLOYMENT.md](DEPLOYMENT.md)
