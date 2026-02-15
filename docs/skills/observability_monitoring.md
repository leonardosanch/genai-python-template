# Skill: Observability & Monitoring

## Description
This skill covers production observability, monitoring, and debugging for GenAI applications. It includes instrumentation, metrics collection, structured logging, distributed tracing, cost tracking, quality degradation detection, and incident response.

## Executive Summary

**Critical observability rules:**
- **Track LLM cost per request** — Log model, tokens (input+output), latency, and computed cost for EVERY call
- **Structured logging mandatory** — Use JSON with structlog; include `request_id` in ALL log entries for correlation
- **OpenTelemetry spans for ALL LLM calls** — Instrument with model, tokens, cost, and latency attributes
- **Monitor Quality & Drift** — Alert on quality degradation (e.g., faithfulness < threshold) and detect semantic drift
- **TTFT is critical** — For streaming responses, track Time to First Token (target: < 500ms)
- **Never log secrets or raw PII** — Mask sensitive data before logging prompts or outputs
- **Safe metrics collection** — Prometheus labels cardinality must be < 100; avoid user IDs in labels

**Read full skill when:** Instrumenting LLM applications, setting up dashboards, configuring alerting, tracking costs, or responding to production incidents.

---

## Versiones y Thresholds

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| opentelemetry-sdk | >= 1.20.0 | API estable desde 1.x |
| opentelemetry-api | >= 1.20.0 | Requerido junto con SDK |
| structlog | >= 23.0.0 | Procesadores async estables |
| prometheus-client | >= 0.19.0 | Multiprocess mode estable |
| loki-logger | >= 1.0.0 | Si se usa Loki |

### ⚠️ Advertencias de Plataformas de Observabilidad

| Herramienta | Estabilidad | Acción Requerida |
|-------------|-------------|------------------|
| LangSmith | ⚠️ API evoluciona | Verificar changelog antes de actualizar SDK |
| LangFuse | ⚠️ En desarrollo activo | Verificar breaking changes en releases |
| Phoenix (Arize) | ⚠️ API puede cambiar | Consultar docs para versión específica |

---

## Deep Dive

## Core Concepts

1. **The Three Pillars**: 
   - **Logs**: Structured record of events (what happened).
   - **Metrics**: Numerical time-series data (how much/how fast).
   - **Traces**: Request flow through distributed systems (where/when).

2. **LLM-Specific Observability**:
   - Traditional metrics (CPU/RAM) are insufficient for GenAI.
   - Must track internal LLM state: token usage, latency (TTFT, total), cost, and qualitative performance.

3. **Quality & Hallucination Monitoring**:
   - Automated evaluation of production samples using "LLM-as-judge" (RAGAS, DeepEval).
   - Detection of semantic drift compared to "golden" response distributions.

4. **Distributed Tracing in Agents**:
   - Multi-agent systems (LangGraph) require tracing across sequential or parallel node executions.

---

## External Resources

### 📊 Observability Frameworks
- **OpenTelemetry**: [opentelemetry.io](https://opentelemetry.io/) - Vendor-neutral instrumentation standard.
- **LangSmith**: [smith.langchain.com](https://smith.langchain.com/) - LLM debugging and evaluation platform.
- **LangFuse**: [langfuse.com](https://langfuse.com/) - Open-source alternative for LLM observability.
- **Phoenix**: [arize.com/phoenix](https://arize.com/phoenix) - Observability and embeddings visualization.

### 📈 Metrics & Dashboards
- **Prometheus**: [prometheus.io](https://prometheus.io/) - Standard for metrics and alerting.
- **Grafana**: [grafana.com](https://grafana.com/) - Multi-source dashboarding and visualization.
- **Loki**: [grafana.com/oss/loki](https://grafana.com/oss/loki/) - Cost-effective log aggregation.

---

## Code Examples

### Example 1: Full OpenTelemetry & Prometheus Instrumentation

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter (Jaeger, Tempo, etc.)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Prometheus Metrics
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['model', 'type', 'endpoint']
)
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
llm_ttft_seconds = Histogram(
    'llm_ttft_seconds',
    'Time to first token',
    ['model'],
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0]
)
llm_cost_usd = Counter(
    'llm_cost_usd_total',
    'Total LLM cost in USD',
    ['model', 'endpoint']
)
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'endpoint', 'status']
)
llm_active_requests = Gauge(
    'llm_active_requests',
    'Currently active LLM requests',
    ['model']
)

async def call_llm_monitored(prompt: str, model: str = "gpt-4", endpoint: str = "/generate"):
    """Fully instrumented LLM call with tracing and metrics."""
    with tracer.start_as_current_span("llm.generate") as span:
        # Span attributes
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.endpoint", endpoint)
        span.set_attribute("llm.prompt_length", len(prompt))
        
        llm_active_requests.labels(model=model).inc()
        start_time = time.time()
        ttft = None
        
        try:
            # Simulate LLM call
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            first_token = True
            full_response = ""
            
            async for chunk in response:
                if first_token:
                    ttft = time.time() - start_time
                    llm_ttft_seconds.labels(model=model).observe(ttft)
                    span.set_attribute("llm.ttft_ms", ttft * 1000)
                    first_token = False
                
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            latency = time.time() - start_time
            
            # Calculate cost (example pricing)
            cost = calculate_cost(model, response.usage.total_tokens)
            
            # Record metrics
            llm_tokens_total.labels(model=model, type="prompt", endpoint=endpoint).inc(
                response.usage.prompt_tokens
            )
            llm_tokens_total.labels(model=model, type="completion", endpoint=endpoint).inc(
                response.usage.completion_tokens
            )
            llm_latency_seconds.labels(model=model, endpoint=endpoint).observe(latency)
            llm_cost_usd.labels(model=model, endpoint=endpoint).inc(cost)
            llm_requests_total.labels(model=model, endpoint=endpoint, status="success").inc()
            
            # Span attributes
            span.set_attribute("llm.tokens.prompt", response.usage.prompt_tokens)
            span.set_attribute("llm.tokens.completion", response.usage.completion_tokens)
            span.set_attribute("llm.tokens.total", response.usage.total_tokens)
            span.set_attribute("llm.latency_ms", latency * 1000)
            span.set_attribute("llm.cost_usd", cost)
            span.set_status(trace.Status(trace.StatusCode.OK))
            
            return full_response
            
        except Exception as e:
            llm_requests_total.labels(model=model, endpoint=endpoint, status="error").inc()
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            llm_active_requests.labels(model=model).dec()


def calculate_cost(model: str, tokens: int) -> float:
    """Calculate cost based on model pricing."""
    pricing = {
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
    }
    # Simplified: assume 50/50 split
    rate = pricing.get(model, {"input": 0.01 / 1000, "output": 0.01 / 1000})
    return tokens * (rate["input"] + rate["output"]) / 2
```

### Example 2: Structured Logging with Context

```python
import structlog
from uuid import uuid4

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def process_user_query(query: str, user_id: str):
    """Process query with full observability context."""
    request_id = str(uuid4())
    log = logger.bind(
        user_id=user_id,
        request_id=request_id,
        query_length=len(query)
    )
    
    log.info("query_started")
    
    try:
        # Business logic with nested spans
        with tracer.start_as_current_span("process_query") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("request_id", request_id)
            
            result = await run_pipeline(query)
            
            log.info(
                "query_success",
                tokens=result.tokens,
                cost_usd=result.cost,
                latency_ms=result.latency_ms
            )
            return result
            
    except ValueError as e:
        log.warning("query_validation_failed", error=str(e))
        raise
    except Exception as e:
        log.error(
            "query_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise
```

---

## Grafana Dashboards

### LLM Performance Dashboard

```json
{
  "dashboard": {
    "title": "LLM Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])",
            "legendFormat": "{{model}} - {{status}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency by Model",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Time to First Token (TTFT)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_ttft_seconds_bucket[5m]))",
            "legendFormat": "P50 - {{model}}"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_ttft_seconds_bucket[5m]))",
            "legendFormat": "P95 - {{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Token Consumption",
        "targets": [
          {
            "expr": "rate(llm_tokens_total[5m])",
            "legendFormat": "{{model}} - {{type}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Cost per Hour",
        "targets": [
          {
            "expr": "rate(llm_cost_usd_total[1h]) * 3600",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(llm_requests_total{status='error'}[5m]) / rate(llm_requests_total[5m])",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph",
        "alert": {
          "conditions": [
            {
              "evaluator": {"params": [0.05], "type": "gt"},
              "query": {"params": ["A", "5m", "now"]}
            }
          ]
        }
      }
    ]
  }
}
```

### Key PromQL Queries

```promql
# Request rate per model
rate(llm_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))

# Error rate percentage
rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) * 100

# Cost per day projection
rate(llm_cost_usd_total[1h]) * 24

# Average tokens per request
rate(llm_tokens_total[5m]) / rate(llm_requests_total[5m])

# Active requests gauge
llm_active_requests
```

---

## Prometheus Alerting Rules

```yaml
# /etc/prometheus/rules/llm_alerts.yml
groups:
  - name: llm_performance
    interval: 30s
    rules:
      # High latency alert
      - alert: LLMHighLatency
        expr: |
          histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
          component: llm
        annotations:
          summary: "High LLM latency detected"
          description: "P95 latency for {{ $labels.model }} is {{ $value }}s (threshold: 5s)"
      
      # High error rate
      - alert: LLMHighErrorRate
        expr: |
          rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
          component: llm
        annotations:
          summary: "High LLM error rate"
          description: "Error rate for {{ $labels.model }} is {{ $value | humanizePercentage }} (threshold: 5%)"
      
      # Cost spike
      - alert: LLMCostSpike
        expr: |
          rate(llm_cost_usd_total[1h]) > 2 * avg_over_time(rate(llm_cost_usd_total[1h])[24h:1h])
        for: 10m
        labels:
          severity: warning
          component: llm
          team: finance
        annotations:
          summary: "LLM cost spike detected"
          description: "Hourly cost for {{ $labels.model }} is {{ $value | humanize }}x baseline"
      
      # Slow TTFT
      - alert: LLMSlowTTFT
        expr: |
          histogram_quantile(0.95, rate(llm_ttft_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
          component: llm
        annotations:
          summary: "Slow time to first token"
          description: "P95 TTFT for {{ $labels.model }} is {{ $value }}s (threshold: 1s)"
      
      # No requests (service down?)
      - alert: LLMNoRequests
        expr: |
          rate(llm_requests_total[5m]) == 0
        for: 5m
        labels:
          severity: critical
          component: llm
        annotations:
          summary: "No LLM requests detected"
          description: "Model {{ $labels.model }} has received no requests in 5 minutes"
```

---

## Distributed Tracing Patterns

### Multi-Agent Tracing

```python
from opentelemetry import trace, context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

tracer = trace.get_tracer(__name__)
propagator = TraceContextTextMapPropagator()

async def supervisor_agent(task: str):
    """Supervisor agent that delegates to workers."""
    with tracer.start_as_current_span("supervisor.delegate") as span:
        span.set_attribute("task", task)
        span.set_attribute("agent.role", "supervisor")
        
        # Propagate trace context to worker agents
        carrier = {}
        propagator.inject(carrier)
        
        # Delegate to workers
        results = await asyncio.gather(
            worker_agent_1(task, carrier),
            worker_agent_2(task, carrier),
        )
        
        span.set_attribute("workers.count", len(results))
        return results

async def worker_agent_1(task: str, carrier: dict):
    """Worker agent that continues the trace."""
    # Extract parent context
    ctx = propagator.extract(carrier)
    
    with tracer.start_as_current_span("worker1.execute", context=ctx) as span:
        span.set_attribute("agent.role", "worker1")
        span.set_attribute("task", task)
        
        # Nested LLM call
        result = await call_llm_monitored(f"Process: {task}", model="gpt-4")
        
        span.set_attribute("result.length", len(result))
        return result
```

### RAG Pipeline Tracing

```python
async def rag_query_traced(query: str):
    """RAG pipeline with full tracing."""
    with tracer.start_as_current_span("rag.query") as root_span:
        root_span.set_attribute("query", query)
        
        # Step 1: Embedding
        with tracer.start_as_current_span("rag.embed") as span:
            embedding = await embed_query(query)
            span.set_attribute("embedding.dim", len(embedding))
        
        # Step 2: Retrieval
        with tracer.start_as_current_span("rag.retrieve") as span:
            docs = await vector_store.search(embedding, top_k=5)
            span.set_attribute("docs.count", len(docs))
            span.set_attribute("docs.avg_score", sum(d.score for d in docs) / len(docs))
        
        # Step 3: Reranking
        with tracer.start_as_current_span("rag.rerank") as span:
            reranked = await reranker.rerank(query, docs)
            span.set_attribute("reranked.count", len(reranked))
        
        # Step 4: Generation
        with tracer.start_as_current_span("rag.generate") as span:
            context = "\n".join(d.content for d in reranked[:3])
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            
            answer = await call_llm_monitored(prompt, model="gpt-4")
            
            span.set_attribute("context.length", len(context))
            span.set_attribute("answer.length", len(answer))
        
        root_span.set_attribute("pipeline.success", True)
        return answer
```

---

## Incident Playbooks

### 🚨 High Latency (P95 > 5s)

**Symptoms:**
- Grafana alert: `LLMHighLatency` firing
- User complaints about slow responses
- P95 latency metric spiking

**Diagnosis Steps:**
1. **Check Provider Status**
   ```bash
   curl https://status.openai.com/api/v2/status.json
   ```
2. **Analyze Request Distribution**
   ```promql
   topk(10, rate(llm_latency_seconds_sum[5m]) / rate(llm_latency_seconds_count[5m]))
   ```
3. **Check Prompt Complexity**
   - Review recent prompt changes in Git
   - Check average prompt length: `avg(llm_prompt_length)`
4. **Verify Concurrency Limits**
   ```promql
   llm_active_requests > 50
   ```

**Mitigation:**
1. **Immediate**: Switch to faster model (gpt-3.5-turbo)
   ```python
   # Feature flag or config change
   DEFAULT_MODEL = "gpt-3.5-turbo"  # was gpt-4
   ```
2. **Short-term**: Enable semantic caching
   ```python
   cache_threshold = 0.95  # Cache queries with >95% similarity
   ```
3. **Long-term**: Implement request queuing with priority

---

### 💸 Cost Spike (> 2x baseline)

**Symptoms:**
- Grafana alert: `LLMCostSpike` firing
- Daily cost report shows anomaly
- Finance team escalation

**Diagnosis Steps:**
1. **Identify High-Usage Users/Endpoints**
   ```promql
   topk(10, sum by (user_id) (rate(llm_cost_usd_total[1h])))
   topk(10, sum by (endpoint) (rate(llm_cost_usd_total[1h])))
   ```
2. **Check for Retry Loops**
   ```bash
   # Analyze logs for repeated requests
   cat logs/app.log | jq 'select(.event=="llm_call") | .request_id' | sort | uniq -c | sort -rn | head
   ```
3. **Detect Prompt Injection Attacks**
   ```python
   # Check for abnormally long prompts
   SELECT AVG(LENGTH(prompt)) FROM llm_logs WHERE timestamp > NOW() - INTERVAL '1 hour';
   ```

**Mitigation:**
1. **Immediate**: Apply per-user cost caps
   ```python
   if user_daily_cost > USER_COST_LIMIT:
       raise RateLimitError("Daily cost limit exceeded")
   ```
2. **Short-term**: Aggressive caching (lower similarity threshold)
   ```python
   cache_threshold = 0.85  # was 0.95
   ```
3. **Long-term**: Implement tiered pricing with quotas

---

### 📉 Quality Drop (Faithfulness < 0.7)

**Symptoms:**
- RAGAS evaluation metrics declining
- User feedback scores dropping
- Increased hallucination reports

**Diagnosis Steps:**
1. **Check Recent Prompt Changes**
   ```bash
   git log --since="1 week ago" -- prompts/
   ```
2. **Analyze Retrieval Quality**
   ```promql
   avg(rag_context_precision) < 0.75
   ```
3. **Detect Input Distribution Shift**
   ```python
   # Compare query embeddings distribution
   from scipy.stats import ks_2samp
   stat, p_value = ks_2samp(current_embeddings, baseline_embeddings)
   if p_value < 0.05:
       print("Distribution shift detected")
   ```

**Mitigation:**
1. **Immediate**: Rollback to previous prompt version
   ```bash
   git revert HEAD~1 -- prompts/v2/
   ```
2. **Short-term**: Increase RAG top-k (retrieve more context)
   ```python
   top_k = 10  # was 5
   ```
3. **Long-term**: Re-index embeddings with updated model

---

### 🔥 Service Outage (Error Rate > 50%)

**Symptoms:**
- Multiple alerts firing
- `/health` endpoint failing
- No successful requests

**Diagnosis Steps:**
1. **Check Service Health**
   ```bash
   curl http://localhost:8000/health
   kubectl get pods -n production
   ```
2. **Review Recent Deployments**
   ```bash
   kubectl rollout history deployment/llm-api
   ```
3. **Check External Dependencies**
   - Database connectivity
   - Redis cache
   - Vector store
   - LLM provider API

**Mitigation:**
1. **Immediate**: Rollback deployment
   ```bash
   kubectl rollout undo deployment/llm-api
   ```
2. **Activate fallback**: Switch to backup LLM provider
3. **Communicate**: Update status page, notify users

---

## Instructions for the Agent

1. **Instrumentation**: ALWAYS include OpenTelemetry spans for core LLM logic. Every LLM call, RAG pipeline step, and agent action must be traced.

2. **Metrics**: Export `/metrics` endpoint with:
   - Token counts (prompt, completion, total) by model and endpoint
   - Latency histograms with buckets [0.1, 0.5, 1, 2, 5, 10, 30]
   - Cost counters in USD
   - Error rates by type
   - TTFT (Time to First Token) for streaming

3. **Logging**: Use JSON structured logs with structlog. NEVER log:
   - Raw API keys or secrets
   - Full prompts containing PII
   - User passwords or tokens
   - Always include: `request_id`, `user_id`, `timestamp`, `level`, `event`

4. **Alerts**: Define alerting thresholds:
   - Error rate > 5% for 3 minutes
   - P95 latency > 5s for 5 minutes
   - Cost spike > 2x baseline for 10 minutes
   - TTFT > 1s for 5 minutes

5. **Costing**: Implement `calculate_cost()` utility:
   - Use current pricing from provider docs
   - Track separately by model and endpoint
   - Update pricing monthly
   - Include in every span and log entry

6. **Dashboards**: Create Grafana dashboards for:
   - LLM Performance (latency, throughput, errors)
   - Cost Tracking (hourly, daily, by model)
   - Quality Metrics (faithfulness, relevancy)
   - System Health (CPU, memory, active requests)

7. **Incident Response**: Follow playbooks for:
   - High latency: Check provider → prompt complexity → concurrency
   - Cost spike: Identify users → check retries → apply caps
   - Quality drop: Check prompts → retrieval → distribution shift
   - Outage: Health check → rollback → fallback → communicate
