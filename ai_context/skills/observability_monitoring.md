# Skill: Observability & Monitoring

## Description
This skill covers production observability, monitoring, and debugging for GenAI applications. Use this when instrumenting code, setting up metrics, creating dashboards, or troubleshooting production issues.

## Core Concepts

1. **Three Pillars**: Logs (what happened), Metrics (how much), Traces (where/when).
2. **OpenTelemetry**: Vendor-neutral standard for instrumentation.
3. **LLM-Specific Metrics**: Tokens, cost, latency, quality scores.
4. **Distributed Tracing**: Track requests across multi-agent systems.

---

## External Resources

### 📊 Observability Frameworks

#### OpenTelemetry
- **OpenTelemetry Documentation**: [opentelemetry.io/docs/](https://opentelemetry.io/docs/)
    - *Best for*: Vendor-neutral instrumentation standard
- **OpenTelemetry Python**: [opentelemetry-python.readthedocs.io](https://opentelemetry-python.readthedocs.io/)
    - *Best for*: Python instrumentation, auto-instrumentation
- **OpenTelemetry Semantic Conventions**: [opentelemetry.io/docs/specs/semconv/](https://opentelemetry.io/docs/specs/semconv/)
    - *Best for*: Standardized attribute naming

#### LLM Observability
- **LangSmith**: [docs.smith.langchain.com](https://docs.smith.langchain.com/)
    - *Best for*: LLM tracing, debugging, evaluation
- **Phoenix** (Arize AI): [docs.arize.com/phoenix](https://docs.arize.com/phoenix)
    - *Best for*: LLM observability, embeddings visualization
- **Weights & Biases**: [docs.wandb.ai](https://docs.wandb.ai/)
    - *Best for*: Experiment tracking, model monitoring
- **Helicone**: [docs.helicone.ai](https://docs.helicone.ai/)
    - *Best for*: LLM request logging, cost tracking
- **LangFuse**: [langfuse.com/docs](https://langfuse.com/docs)
    - *Best for*: Open-source LLM observability

---

### 📈 Metrics & Monitoring

#### Metrics Collection
- **Prometheus**: [prometheus.io/docs/](https://prometheus.io/docs/)
    - *Best for*: Time-series metrics, alerting
- **Prometheus Python Client**: [github.com/prometheus/client_python](https://github.com/prometheus/client_python)
    - *Best for*: Exposing custom metrics
- **StatsD**: [github.com/statsd/statsd](https://github.com/statsd/statsd)
    - *Best for*: Application metrics aggregation

#### Dashboards
- **Grafana**: [grafana.com/docs/](https://grafana.com/docs/)
    - *Best for*: Metrics visualization, alerting
- **Grafana Loki**: [grafana.com/docs/loki/](https://grafana.com/docs/loki/)
    - *Best for*: Log aggregation (like Prometheus for logs)
- **Datadog**: [docs.datadoghq.com](https://docs.datadoghq.com/)
    - *Best for*: Full-stack observability (SaaS)

---

### 🔍 Logging

#### Logging Libraries
- **structlog**: [www.structlog.org](https://www.structlog.org/)
    - *Best for*: Structured logging in Python
- **Python logging**: [docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)
    - *Best for*: Standard library logging
- **loguru**: [loguru.readthedocs.io](https://loguru.readthedocs.io/)
    - *Best for*: Simplified logging with better defaults

#### Log Aggregation
- **ELK Stack** (Elasticsearch, Logstash, Kibana): [elastic.co/elastic-stack](https://www.elastic.co/elastic-stack)
    - *Best for*: Centralized logging, search, visualization
- **Grafana Loki**: [grafana.com/oss/loki/](https://grafana.com/oss/loki/)
    - *Best for*: Cost-effective log aggregation
- **Fluentd**: [docs.fluentd.org](https://docs.fluentd.org/)
    - *Best for*: Log collection and forwarding

---

### 🕵️ Distributed Tracing

#### Tracing Backends
- **Jaeger**: [jaegertracing.io/docs/](https://www.jaegertracing.io/docs/)
    - *Best for*: Distributed tracing, open-source
- **Zipkin**: [zipkin.io](https://zipkin.io/)
    - *Best for*: Distributed tracing, simple setup
- **Tempo** (Grafana): [grafana.com/docs/tempo/](https://grafana.com/docs/tempo/)
    - *Best for*: Cost-effective tracing storage
- **Honeycomb**: [docs.honeycomb.io](https://docs.honeycomb.io/)
    - *Best for*: High-cardinality observability (SaaS)

---

### 🚨 Alerting & Incident Response

#### Alerting
- **Prometheus Alertmanager**: [prometheus.io/docs/alerting/](https://prometheus.io/docs/alerting/latest/alertmanager/)
    - *Best for*: Alert routing, grouping, silencing
- **PagerDuty**: [developer.pagerduty.com](https://developer.pagerduty.com/)
    - *Best for*: Incident management, on-call scheduling
- **Opsgenie**: [support.atlassian.com/opsgenie/](https://support.atlassian.com/opsgenie/)
    - *Best for*: Alert management, escalation

#### Error Tracking
- **Sentry**: [docs.sentry.io](https://docs.sentry.io/)
    - *Best for*: Error tracking, performance monitoring
- **Rollbar**: [docs.rollbar.com](https://docs.rollbar.com/)
    - *Best for*: Real-time error tracking

---

### 📚 Books & Guides

#### Observability Fundamentals
- **Observability Engineering** (Charity Majors, Liz Fong-Jones, George Miranda)
    - *Best for*: Modern observability practices
- **Distributed Systems Observability** (Cindy Sridharan)
    - [distributed-systems-observability-ebook.humio.com](https://distributed-systems-observability-ebook.humio.com/)
    - *Best for*: Observability in microservices
- **Site Reliability Engineering** (Google)
    - [sre.google/books/](https://sre.google/books/)
    - *Best for*: SLIs, SLOs, SLAs, error budgets

#### Best Practices
- **The Twelve-Factor App - Logs**: [12factor.net/logs](https://12factor.net/logs)
    - *Best for*: Treating logs as event streams
- **OpenTelemetry Best Practices**: [opentelemetry.io/docs/concepts/](https://opentelemetry.io/docs/concepts/)
    - *Best for*: Instrumentation patterns
- **DORA 2024 State of DevOps Report**: [dora.dev/research/2024/](https://dora.dev/research/2024/)
    - *Best for*: Benchmarks on software delivery performance, AI impact on productivity, and platform engineering insights

---

## Code Examples

### Example 1: OpenTelemetry Instrumentation for LLM Calls

```python
# src/observability/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to Jaeger/Tempo
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument HTTP clients
HTTPXClientInstrumentor().instrument()

# Instrument LLM calls
async def call_llm(prompt: str, model: str = "gpt-4") -> str:
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_length", len(prompt))
        
        start_time = time.time()
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start_time
        
        span.set_attribute("llm.tokens_used", response.usage.total_tokens)
        span.set_attribute("llm.latency_ms", latency * 1000)
        span.set_attribute("llm.cost_usd", calculate_cost(response.usage, model))
        
        return response.choices[0].message.content
```

### Example 2: Custom Prometheus Metrics for LLM

```python
# src/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['model', 'type']  # type: prompt, completion
)

llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

llm_cost_usd = Counter(
    'llm_cost_usd_total',
    'Total LLM cost in USD',
    ['model']
)

active_llm_requests = Gauge(
    'llm_active_requests',
    'Currently active LLM requests',
    ['model']
)

# Instrument function
async def call_llm_with_metrics(prompt: str, model: str = "gpt-4") -> str:
    active_llm_requests.labels(model=model).inc()
    
    try:
        start_time = time.time()
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start_time
        
        # Record metrics
        llm_requests_total.labels(model=model, status="success").inc()
        llm_tokens_total.labels(model=model, type="prompt").inc(
            response.usage.prompt_tokens
        )
        llm_tokens_total.labels(model=model, type="completion").inc(
            response.usage.completion_tokens
        )
        llm_latency_seconds.labels(model=model).observe(latency)
        
        cost = calculate_cost(response.usage, model)
        llm_cost_usd.labels(model=model).inc(cost)
        
        return response.choices[0].message.content
        
    except Exception as e:
        llm_requests_total.labels(model=model, status="error").inc()
        raise
    finally:
        active_llm_requests.labels(model=model).dec()

# Start metrics server
start_http_server(8000)  # Metrics available at http://localhost:8000/metrics
```

### Example 3: Structured Logging with Context

```python
# src/observability/logging.py
import structlog
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Usage in application
async def process_query(query: str, user_id: str):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    user_id_var.set(user_id)
    
    logger.info(
        "query_received",
        query_length=len(query),
        user_id=user_id,
        request_id=request_id
    )
    
    try:
        result = await rag_pipeline.query(query)
        logger.info(
            "query_completed",
            sources_count=len(result.sources),
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms
        )
        return result
    except Exception as e:
        logger.error(
            "query_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise
```

### Example 4: Multi-Agent Tracing

```python
# src/agents/supervisor.py
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def supervisor_agent(task: str) -> str:
    with tracer.start_as_current_span("agent.supervisor") as span:
        span.set_attribute("task", task)
        span.set_attribute("agent.type", "supervisor")
        
        # Route to worker
        worker = select_worker(task)
        span.set_attribute("worker.selected", worker.name)
        
        with tracer.start_as_current_span(f"agent.worker.{worker.name}") as worker_span:
            worker_span.set_attribute("agent.type", "worker")
            result = await worker.execute(task)
            worker_span.set_attribute("result.length", len(result))
        
        span.set_attribute("status", "completed")
        return result
```

---

## Anti-Patterns to Avoid

### ❌ Logging Secrets
**Problem**: Sensitive data in logs  
**Example**:
```python
# BAD: Logging API keys
logger.info(f"Calling API with key: {api_key}")
```
**Solution**: Mask sensitive data
```python
# GOOD: Masked secrets
logger.info(f"Calling API with key: {api_key[:4]}***")
```

### ❌ High-Cardinality Metrics
**Problem**: Too many unique label combinations, memory explosion  
**Example**:
```python
# BAD: User ID as label (millions of unique values)
requests_total.labels(user_id=user_id).inc()
```
**Solution**: Use low-cardinality labels
```python
# GOOD: User tier instead of ID
requests_total.labels(user_tier=get_user_tier(user_id)).inc()
```

### ❌ Synchronous Logging in Async Code
**Problem**: Blocking event loop  
**Solution**: Use async logging or background tasks

### ❌ No Sampling for High-Volume Traces
**Problem**: Trace storage costs explode  
**Solution**: Use probabilistic sampling (e.g., 1% of requests)

---

## Observability Checklist

### Pre-Production Checklist
- [ ] OpenTelemetry instrumentation configured
- [ ] Structured logging implemented (JSON format)
- [ ] Custom metrics defined for LLM calls
- [ ] Dashboards created in Grafana
- [ ] Alerts configured for critical metrics
- [ ] Error tracking enabled (Sentry)
- [ ] Log aggregation configured (Loki/ELK)
- [ ] Distributed tracing backend deployed (Jaeger/Tempo)

### LLM Metrics Checklist
- [ ] Token usage tracked per model
- [ ] Cost tracked per request
- [ ] Latency measured (p50, p95, p99)
- [ ] Error rate monitored
- [ ] Quality scores logged (faithfulness, relevancy)
- [ ] Rate limit violations tracked

### Logging Checklist
- [ ] All logs are structured (JSON)
- [ ] Request ID in all log entries
- [ ] No secrets in logs
- [ ] Log levels used correctly (DEBUG, INFO, WARNING, ERROR)
- [ ] Correlation IDs for distributed requests

### Dashboard Checklist
- [ ] LLM request rate (requests/sec)
- [ ] LLM latency (p50, p95, p99)
- [ ] Token usage over time
- [ ] Cost per hour/day
- [ ] Error rate by model
- [ ] Active requests gauge

---

## Instructions for the Agent

1. **Instrumentation**:
   - Use OpenTelemetry for all instrumentation
   - Instrument LLM calls with custom spans
   - Add semantic attributes (model, tokens, cost)
   - Use auto-instrumentation for HTTP/DB calls

2. **Metrics**:
   - Expose Prometheus metrics on `/metrics` endpoint
   - Track: request rate, latency, errors, tokens, cost
   - Use Histograms for latency (not Gauges)
   - Keep label cardinality low (< 100 unique values)

3. **Logging**:
   - Use structured logging (structlog or JSON)
   - Include request_id in all logs
   - Never log secrets or PII
   - Log to stdout (12-factor app)
   - Use appropriate log levels

4. **Tracing**:
   - Create spans for each agent in multi-agent systems
   - Add attributes for debugging (model, tokens, cost)
   - Use trace context propagation
   - Sample traces in production (1-10%)

5. **Dashboards**:
   - Create Grafana dashboards for LLM metrics
   - Include SLI/SLO tracking
   - Set up alerts for SLO violations
   - Monitor cost trends

6. **Alerting**:
   - Alert on error rate > 5%
   - Alert on latency p95 > threshold
   - Alert on cost spike (> 2x baseline)
   - Alert on rate limit violations

7. **Cost Tracking**:
   - Calculate cost per request
   - Track cost by model, user tier, feature
   - Set up cost budgets and alerts
   - Optimize high-cost queries
