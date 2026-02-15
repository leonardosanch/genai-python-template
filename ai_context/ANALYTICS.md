# Analytics, BI & Reporting

## Python para Data-Driven Decisions

Desde análisis ad-hoc hasta dashboards en producción y reportes automáticos.

---

## Análisis de Datos

### Pandas / Polars

Ver [DATA_ENGINEERING.md](DATA_ENGINEERING.md) para comparativa detallada.

```python
import polars as pl

def analyze_llm_usage(logs_path: str) -> dict:
    """Analizar uso de LLM desde logs."""
    df = pl.read_parquet(logs_path)

    summary = {
        "total_requests": len(df),
        "total_tokens": df["tokens_used"].sum(),
        "total_cost": df["cost_usd"].sum(),
        "avg_latency_ms": df["latency_ms"].mean(),
        "p95_latency_ms": df["latency_ms"].quantile(0.95),
        "error_rate": df.filter(pl.col("status") == "error").height / len(df),
        "by_model": df.group_by("model").agg(
            pl.col("tokens_used").sum().alias("total_tokens"),
            pl.col("cost_usd").sum().alias("total_cost"),
            pl.len().alias("requests"),
        ).to_dicts(),
    }
    return summary
```

---

## Dashboards

### Plotly Dash

Dashboards interactivos en Python puro.

```python
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div([
    html.H1("GenAI System Dashboard"),

    dcc.Dropdown(
        id="model-filter",
        options=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
        value="gpt-4o",
    ),

    dcc.Graph(id="tokens-over-time"),
    dcc.Graph(id="cost-by-model"),
    dcc.Graph(id="latency-distribution"),
])

@callback(Output("tokens-over-time", "figure"), Input("model-filter", "value"))
def update_tokens_chart(model: str):
    df = load_metrics(model=model)
    return px.line(df, x="timestamp", y="tokens_used", title="Tokens Over Time")

@callback(Output("cost-by-model", "figure"), Input("model-filter", "value"))
def update_cost_chart(_):
    df = load_cost_summary()
    return px.pie(df, values="cost_usd", names="model", title="Cost by Model")

@callback(Output("latency-distribution", "figure"), Input("model-filter", "value"))
def update_latency_chart(model: str):
    df = load_metrics(model=model)
    return px.histogram(df, x="latency_ms", title="Latency Distribution", nbins=50)
```

### Streamlit

Prototipos rápidos de dashboards y demos interactivas.

```python
import streamlit as st

st.title("RAG System Explorer")

query = st.text_input("Ask a question")
model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
top_k = st.slider("Documents to retrieve", 1, 20, 5)

if st.button("Search"):
    with st.spinner("Searching..."):
        result = rag_pipeline.query(query, model=model, top_k=top_k)

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Sources")
    for source in result.sources:
        st.markdown(f"- {source}")

    st.metric("Confidence", f"{result.confidence:.0%}")
    st.metric("Tokens Used", result.tokens_used)
```

---

## Reportes Automáticos

```python
async def generate_weekly_report() -> Report:
    """Reporte semanal automático para stakeholders."""
    metrics = await gather_weekly_metrics()

    # Usar LLM para generar análisis narrativo
    analysis = await llm.generate_structured(
        prompt=f"""Analyze these weekly GenAI system metrics and provide insights:

        Total requests: {metrics.total_requests}
        Total cost: ${metrics.total_cost:.2f}
        Average latency: {metrics.avg_latency_ms:.0f}ms
        Error rate: {metrics.error_rate:.1%}
        Top queries: {metrics.top_queries}

        Provide: executive summary, trends, concerns, recommendations.""",
        schema=WeeklyAnalysis,
    )

    return Report(
        title=f"GenAI Weekly Report - {date.today().isoformat()}",
        metrics=metrics,
        analysis=analysis,
        charts=generate_charts(metrics),
    )
```

---

## Jupyter Notebooks

Para exploración y análisis ad-hoc. No para producción.

```python
# notebooks/analyze_rag_quality.ipynb

# Cell 1: Setup
import polars as pl
from src.application.use_cases.evaluate import run_evaluation

# Cell 2: Load data
eval_results = pl.read_json("evaluation_results.json")

# Cell 3: Analyze
print(f"Average faithfulness: {eval_results['faithfulness'].mean():.3f}")
print(f"Average relevancy: {eval_results['answer_relevancy'].mean():.3f}")

# Cell 4: Visualize
eval_results.select("faithfulness", "answer_relevancy").to_pandas().plot.hist(bins=20)
```

**Regla**: Notebooks para exploración. Si algo es útil, extraerlo a un script o módulo.

---

## Superset / Metabase Integration

Conectar dashboards de BI con métricas del sistema GenAI.

```sql
-- Vista SQL para Superset/Metabase
CREATE VIEW v_llm_daily_metrics AS
SELECT
    DATE(created_at) AS date,
    model,
    COUNT(*) AS total_requests,
    SUM(tokens_used) AS total_tokens,
    SUM(cost_usd) AS total_cost,
    AVG(latency_ms) AS avg_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency,
    COUNT(*) FILTER (WHERE status = 'error') * 100.0 / COUNT(*) AS error_rate
FROM llm_logs
GROUP BY DATE(created_at), model;
```

---

## Reglas

1. **Dash para dashboards de producción** — robusto, customizable
2. **Streamlit para demos y prototipos** — rápido, no para prod
3. **Notebooks para exploración** — nunca en producción
4. **Reportes automáticos** — scheduled, no manuales
5. **SQL views para BI tools** — Superset, Metabase, Grafana

Ver también: [OBSERVABILITY.md](OBSERVABILITY.md), [DATA_ENGINEERING.md](DATA_ENGINEERING.md)
