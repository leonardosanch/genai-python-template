---
name: Analytics & BI
description: Data analysis, dashboarding, and automated reporting for GenAI systems.
---

# Skill: Analytics & BI

## Description

Analytics and business intelligence provide visibility into the business value, cost efficiency,
and user behavior of GenAI applications. This skill covers data analysis pipelines, interactive
dashboards, LLM cost tracking, automated reporting, and privacy-aware data handling — ensuring
that operational data is transformed into actionable insights.

## Executive Summary

**Critical analytics rules:**
- NEVER run analytics on production OLTP databases — use read replicas, data warehouses, or DuckDB export
- LLM cost tracking is MANDATORY — log model, tokens, and computed cost per request for aggregation
- Dashboards MUST display data freshness timestamp — stale data without timestamps creates false confidence
- PII MUST be masked before analytics ingestion — apply pseudonymization/hashing at the boundary
- Jupyter ONLY for exploration — production analytics in proper Python modules with tests (never schedule notebooks)

**Read full skill when:** Building dashboards, tracking LLM costs, creating automated reports, implementing data pipelines, or ensuring privacy compliance in analytics.

---

## Versiones y Privacidad

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| polars | >= 0.20.0 | Engine OLAP recomendado |
| duckdb | >= 0.9.0 | SQL para Parquet |
| streamlit | >= 1.30.0 | Dashboards rápidos |

### PII Masking (Hashing)

```python
import hashlib

def mask_user_id(user_id: str, salt: str) -> str:
    # ❌ NUNCA procesar PII en crudo para analítica
    return hashlib.sha256((user_id + salt).encode()).hexdigest()
```

---

## Deep Dive

## Core Concepts

1. **OLTP/OLAP Separation** — Never run analytical queries against production transactional
   databases. Use read replicas, data warehouses (BigQuery, Snowflake, Redshift), or in-process
   OLAP engines (DuckDB) to isolate analytical workloads from user-facing traffic.

2. **LLM Cost as First-Class Metric** — Token usage, model selection, latency, and dollar cost
   per request must be tracked with the same rigor as uptime or error rates. Cost anomaly
   detection prevents runaway spend from prompt injection or misconfigured agents.

3. **Dashboards as Code** — All dashboards must be version-controlled, reproducible, and
   deployable from code. No manually configured Grafana panels or click-built Streamlit apps
   that exist only on one machine.

4. **Privacy by Design** — PII must be masked or removed before data enters analytics pipelines.
   Use pseudonymization, hashing, or differential privacy. Log aggregates, not individual
   user data.

5. **Data Freshness Indicators** — Every dashboard and report must display when the underlying
   data was last updated. Stale data without a timestamp is worse than no data — it creates
   false confidence.

6. **Jupyter for Exploration Only** — Notebooks are for ad-hoc analysis and prototyping.
   Production analytics must be in proper Python modules, tested and version-controlled.
   Never schedule a notebook as a production job.

## External Resources

### ⚡ Data Processing
- [Polars Documentation](https://docs.pola.rs/)
  *Best for*: High-performance DataFrame operations with lazy evaluation and multi-threading.
- [Pandas Documentation](https://pandas.pydata.org/docs/)
  *Best for*: Widely adopted DataFrame library; preferred for small datasets and ecosystem compatibility.
- [DuckDB Documentation](https://duckdb.org/docs/)
  *Best for*: In-process OLAP engine for analytical SQL on local files (Parquet, CSV, JSON).

### 📊 Dashboards & Visualization
- [Plotly Dash Documentation](https://dash.plotly.com/)
  *Best for*: Production-grade, customizable analytical dashboards in pure Python.
- [Streamlit Documentation](https://docs.streamlit.io/)
  *Best for*: Rapid prototyping of data apps and internal tools with minimal frontend code.

> ⚠️ **Streamlit Cache API**: Desde Streamlit 1.18+, usar `@st.cache_data` para datos y `@st.cache_resource` para conexiones. El decorador `@st.cache` está deprecado.

- [Altair Documentation](https://altair-viz.github.io/)
  *Best for*: Declarative statistical visualization based on Vega-Lite grammar.
- [Panel Documentation](https://panel.holoviz.org/)
  *Best for*: Flexible dashboarding that works with multiple plotting libraries.
- [Great Tables Documentation](https://posit-dev.github.io/great-tables/)
  *Best for*: Publication-quality tables for reports and dashboards.

### 🏢 BI Platforms
- [Apache Superset](https://superset.apache.org/)
  *Best for*: Open-source BI platform with SQL Lab, charting, and dashboard sharing.
- [Metabase](https://www.metabase.com/docs/)
  *Best for*: Simple, self-service BI for teams that need quick insights without SQL expertise.

### 📖 General
- [Jupyter Documentation](https://jupyter.org/documentation)
  *Best for*: Interactive exploration and ad-hoc analysis (not production).
- [WeasyPrint Documentation](https://doc.courtbouillon.org/weasyprint/)
  *Best for*: Generating PDF reports from HTML/CSS templates in Python.

## Instructions for the Agent

1. **Always separate OLTP from OLAP.** When writing analytical queries, target read replicas,
   data warehouses, or in-process engines like DuckDB. Never query production databases directly
   for reporting or dashboards.

2. **Treat LLM cost tracking as mandatory.** Every LLM call must log model name, token counts
   (input + output), latency, and computed cost. Provide aggregation utilities for daily/weekly
   cost reports per model, per use case, and per user.

3. **Mask PII before analytics.** Apply pseudonymization or hashing to user identifiers, email
   addresses, and any personal data before loading into analytics tables. Document which fields
   are masked and the masking strategy.

4. **Dashboards must be code.** Generate Streamlit or Dash apps from version-controlled Python
   modules. Include data freshness timestamps on every view. No manually configured dashboards.

5. **Jupyter is for exploration only.** When an analysis matures, extract it into a proper
   Python module with tests. Never deploy notebooks as scheduled jobs or production artifacts.

6. **Include data freshness on every dashboard.** Display the last-updated timestamp prominently.
   Add visual warnings (color changes, banners) when data is stale beyond a defined threshold.

7. **Prefer Polars over Pandas for new pipelines.** Polars provides better performance through
   lazy evaluation and multi-threading. Use Pandas only when a dependency requires it or for
   small ad-hoc analysis.

8. **Automate report generation.** Weekly/monthly reports should be generated by code, not
   manually. Use LLMs to produce narrative summaries from structured metrics when appropriate.

## Code Examples

### LLM Cost Tracking with Structured Logging and Aggregation

```python
"""LLM cost tracking: log every call, aggregate for reporting."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import structlog

logger = structlog.get_logger()

# Cost per 1K tokens by model (example rates)
MODEL_PRICING: dict[str, dict[str, Decimal]] = {
    "gpt-4o": {"input": Decimal("0.005"), "output": Decimal("0.015")},
    "gpt-4o-mini": {"input": Decimal("0.00015"), "output": Decimal("0.0006")},
    "claude-sonnet-4-20250514": {"input": Decimal("0.003"), "output": Decimal("0.015")},
}


@dataclass(frozen=True)
class LLMUsageRecord:
    """Single LLM call usage record."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    use_case: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def cost_usd(self) -> Decimal:
        pricing = MODEL_PRICING.get(self.model, {"input": Decimal("0"), "output": Decimal("0")})
        input_cost = (Decimal(self.input_tokens) / 1000) * pricing["input"]
        output_cost = (Decimal(self.output_tokens) / 1000) * pricing["output"]
        return input_cost + output_cost


def log_llm_usage(record: LLMUsageRecord) -> None:
    """Log LLM usage with structured fields for aggregation."""
    logger.info(
        "llm_usage",
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        latency_ms=round(record.latency_ms, 2),
        cost_usd=float(record.cost_usd),
        use_case=record.use_case,
        timestamp=record.timestamp.isoformat(),
    )


def aggregate_costs(records: list[LLMUsageRecord]) -> dict[str, dict]:
    """Aggregate costs by model and use case."""
    by_model: dict[str, dict] = {}

    for record in records:
        key = record.model
        if key not in by_model:
            by_model[key] = {
                "total_cost_usd": Decimal("0"),
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "call_count": 0,
                "avg_latency_ms": 0.0,
            }
        entry = by_model[key]
        entry["total_cost_usd"] += record.cost_usd
        entry["total_input_tokens"] += record.input_tokens
        entry["total_output_tokens"] += record.output_tokens
        entry["call_count"] += 1
        # Running average
        n = entry["call_count"]
        entry["avg_latency_ms"] = entry["avg_latency_ms"] + (record.latency_ms - entry["avg_latency_ms"]) / n

    return by_model
```

### Polars Data Analysis Pipeline

```python
"""Polars-based analysis pipeline for LLM usage logs."""

from pathlib import Path

import polars as pl


def load_usage_logs(path: Path) -> pl.LazyFrame:
    """Load LLM usage logs from Parquet with lazy evaluation."""
    return pl.scan_parquet(path)


def daily_cost_summary(logs: pl.LazyFrame) -> pl.DataFrame:
    """Compute daily cost summary grouped by model and use case."""
    return (
        logs.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date", "model", "use_case")
        .agg(
            pl.col("cost_usd").sum().alias("total_cost"),
            pl.col("input_tokens").sum().alias("total_input_tokens"),
            pl.col("output_tokens").sum().alias("total_output_tokens"),
            pl.col("latency_ms").mean().alias("avg_latency_ms"),
            pl.col("latency_ms").quantile(0.95).alias("p95_latency_ms"),
            pl.len().alias("call_count"),
        )
        .sort("date", "model")
        .collect()
    )


def detect_cost_anomalies(
    daily_summary: pl.DataFrame,
    std_threshold: float = 2.0,
) -> pl.DataFrame:
    """Flag days where cost exceeds mean + N standard deviations per model."""
    return (
        daily_summary.with_columns(
            pl.col("total_cost").mean().over("model").alias("mean_cost"),
            pl.col("total_cost").std().over("model").alias("std_cost"),
        )
        .with_columns(
            (pl.col("total_cost") > pl.col("mean_cost") + std_threshold * pl.col("std_cost"))
            .alias("is_anomaly")
        )
        .filter(pl.col("is_anomaly"))
    )


def top_cost_drivers(logs: pl.LazyFrame, top_n: int = 10) -> pl.DataFrame:
    """Identify the top N use cases by total cost."""
    return (
        logs.group_by("use_case")
        .agg(
            pl.col("cost_usd").sum().alias("total_cost"),
            pl.len().alias("call_count"),
            pl.col("cost_usd").mean().alias("avg_cost_per_call"),
        )
        .sort("total_cost", descending=True)
        .head(top_n)
        .collect()
    )
```

### Streamlit Dashboard for GenAI Metrics

```python
"""Streamlit dashboard for GenAI cost and performance metrics."""

from datetime import datetime, timezone
from pathlib import Path

import plotly.express as px
import streamlit as st

# Must be first Streamlit call
st.set_page_config(page_title="GenAI Analytics", layout="wide")

DATA_PATH = Path("data/usage_logs.parquet")
STALE_THRESHOLD_MINUTES = 60


def _check_data_freshness(path: Path) -> tuple[datetime, bool]:
    """Check when data was last modified and if it is stale."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - mtime).total_seconds() / 60
    return mtime, age_minutes > STALE_THRESHOLD_MINUTES


def main() -> None:
    st.title("GenAI Analytics Dashboard")

    # Data freshness indicator
    if DATA_PATH.exists():
        last_updated, is_stale = _check_data_freshness(DATA_PATH)
        color = "red" if is_stale else "green"
        st.markdown(
            f"Data last updated: **:{color}[{last_updated.strftime('%Y-%m-%d %H:%M UTC')}]**"
        )
        if is_stale:
            st.warning("Data is stale. Pipeline may need attention.")
    else:
        st.error("No data file found.")
        return

    import polars as pl

    df = pl.read_parquet(DATA_PATH).to_pandas()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cost (USD)", f"${df['cost_usd'].sum():.2f}")
    col2.metric("Total Calls", f"{len(df):,}")
    col3.metric("Avg Latency (ms)", f"{df['latency_ms'].mean():.0f}")
    col4.metric("Unique Models", df["model"].nunique())

    # Cost over time
    st.subheader("Daily Cost by Model")
    daily = df.groupby([df["timestamp"].dt.date, "model"])["cost_usd"].sum().reset_index()
    daily.columns = ["date", "model", "cost"]
    fig = px.bar(daily, x="date", y="cost", color="model", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    # Latency distribution
    st.subheader("Latency Distribution by Model")
    fig_lat = px.box(df, x="model", y="latency_ms", color="model")
    st.plotly_chart(fig_lat, use_container_width=True)

    # Top use cases
    st.subheader("Top Cost Drivers by Use Case")
    top = df.groupby("use_case")["cost_usd"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top)


if __name__ == "__main__":
    main()
```

### Automated PDF Report with LLM-Generated Analysis

```python
"""Automated report generation with LLM narrative summary."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class ReportMetrics:
    """Aggregated metrics for a reporting period."""

    period_start: datetime
    period_end: datetime
    total_cost_usd: float
    total_calls: int
    avg_latency_ms: float
    cost_by_model: dict[str, float]
    top_use_cases: list[tuple[str, float]]
    anomaly_count: int


async def generate_narrative(
    metrics: ReportMetrics,
    llm_client,  # Abstract LLM port
) -> str:
    """Use LLM to generate a narrative summary of the metrics."""
    prompt = f"""Analyze the following GenAI system metrics for the period
{metrics.period_start.date()} to {metrics.period_end.date()}:

- Total cost: ${metrics.total_cost_usd:.2f}
- Total API calls: {metrics.total_calls:,}
- Average latency: {metrics.avg_latency_ms:.0f}ms
- Cost by model: {metrics.cost_by_model}
- Top use cases by cost: {metrics.top_use_cases[:5]}
- Cost anomalies detected: {metrics.anomaly_count}

Provide a 3-paragraph executive summary highlighting:
1. Cost trends and efficiency
2. Performance observations
3. Recommended actions"""

    return await llm_client.generate(prompt)


def render_html_report(metrics: ReportMetrics, narrative: str) -> str:
    """Render metrics and narrative into an HTML report."""
    model_rows = "".join(
        f"<tr><td>{model}</td><td>${cost:.2f}</td></tr>"
        for model, cost in metrics.cost_by_model.items()
    )

    return f"""<!DOCTYPE html>
<html>
<head><title>GenAI Report — {metrics.period_start.date()}</title></head>
<body>
  <h1>GenAI Weekly Report</h1>
  <p>Period: {metrics.period_start.date()} to {metrics.period_end.date()}</p>
  <p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

  <h2>Key Metrics</h2>
  <table border="1" cellpadding="8">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Cost</td><td>${metrics.total_cost_usd:.2f}</td></tr>
    <tr><td>Total Calls</td><td>{metrics.total_calls:,}</td></tr>
    <tr><td>Avg Latency</td><td>{metrics.avg_latency_ms:.0f}ms</td></tr>
    <tr><td>Anomalies</td><td>{metrics.anomaly_count}</td></tr>
  </table>

  <h2>Cost by Model</h2>
  <table border="1" cellpadding="8">
    <tr><th>Model</th><th>Cost (USD)</th></tr>
    {model_rows}
  </table>

  <h2>Analysis</h2>
  <div>{narrative}</div>
</body>
</html>"""


async def generate_weekly_report(
    metrics: ReportMetrics,
    llm_client,
    output_dir: Path,
) -> Path:
    """Generate a full weekly report as HTML (convert to PDF via WeasyPrint)."""
    log = logger.bind(
        period_start=metrics.period_start.isoformat(),
        period_end=metrics.period_end.isoformat(),
    )
    log.info("report_generation_started")

    narrative = await generate_narrative(metrics, llm_client)
    html = render_html_report(metrics, narrative)

    output_path = output_dir / f"report_{metrics.period_start.date()}.html"
    output_path.write_text(html)

    log.info("report_generated", output=str(output_path))
    return output_path
```

## Anti-Patterns to Avoid

### ❌ Running Analytics on the Production Database

**Problem:** Heavy analytical queries (aggregations, joins, full scans) compete with
user-facing transactions, causing latency spikes and potential outages.

**Example:**
```python
# Querying production DB directly for a dashboard
df = pd.read_sql("SELECT * FROM llm_logs WHERE date > '2024-01-01'", production_engine)
```

**Solution:** Use a read replica, data warehouse, or export to Parquet and query with DuckDB.
Schedule ETL jobs to populate analytics tables separately from OLTP.

### ❌ Jupyter Notebooks in Production

**Problem:** Notebooks lack version control discipline, are hard to test, have hidden state
from out-of-order execution, and cannot be reliably scheduled or monitored.

**Example:**
```bash
# Anti-pattern: scheduling a notebook as a cron job
0 6 * * * jupyter nbconvert --execute /reports/weekly.ipynb
```

**Solution:** Extract mature analysis into Python modules with proper imports, tests, and
CLI entry points. Use notebooks only for exploration and prototyping.

### ❌ No Cost Tracking for LLM Calls

**Problem:** Without per-call cost logging, spend is invisible until the monthly bill arrives.
Runaway costs from misconfigured agents, prompt injection, or retry storms go undetected.

**Example:**
```python
async def ask_llm(prompt: str) -> str:
    response = await client.chat(prompt)
    return response.text  # no token count, no cost, no logging
```

**Solution:** Log model, token counts, latency, and computed cost on every call. Aggregate
daily. Set budget alerts. Build dashboards that show cost trends and anomalies.

### ❌ Dashboards Without Data Freshness Indicators

**Problem:** Stakeholders make decisions based on dashboards without knowing the data is
hours or days old. This creates false confidence and incorrect conclusions.

**Example:**
```python
st.title("Real-time Metrics")  # But data is from yesterday's batch job
st.metric("Total Revenue", calculate_revenue())  # No timestamp shown
```

**Solution:** Display last-updated timestamp on every dashboard view. Add visual warnings
(color change, banner) when data age exceeds a configured threshold. Include the data
pipeline status.

## Analytics Checklist

### Data Pipeline
- [ ] OLTP and OLAP workloads are fully separated
- [ ] ETL jobs are idempotent and resumable
- [ ] Data freshness is tracked and monitored
- [ ] Schema changes are versioned and backward-compatible
- [ ] Parquet or columnar format used for analytical storage

### Dashboard Quality
- [ ] All dashboards are version-controlled as code
- [ ] Data freshness timestamp displayed on every view
- [ ] Stale data warnings with configurable thresholds
- [ ] KPIs defined and agreed upon with stakeholders
- [ ] Dashboards load in under 3 seconds for typical queries

### Cost Management
- [ ] Every LLM call logs model, tokens, latency, and cost
- [ ] Daily cost aggregation by model and use case
- [ ] Cost anomaly detection with alerting
- [ ] Budget thresholds with automated warnings
- [ ] Monthly cost reports generated automatically

### Privacy & Compliance
- [ ] PII masked or pseudonymized before analytics ingestion
- [ ] Data retention policies defined and enforced
- [ ] Access controls on analytics dashboards and data
- [ ] Audit trail for who accessed what data
- [ ] No raw user content stored in analytics tables

## Additional References

- [Polars User Guide](https://docs.pola.rs/user-guide/)
- [DuckDB — Python API](https://duckdb.org/docs/api/python/overview)
- [Streamlit — Deploy Your App](https://docs.streamlit.io/deploy)
- [WeasyPrint — PDF Generation](https://doc.courtbouillon.org/weasyprint/stable/)
- [OpenTelemetry Metrics for LLM Observability](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
