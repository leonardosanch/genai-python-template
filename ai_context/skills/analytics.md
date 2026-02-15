---
name: Analytics & BI
description: Data analysis, dashboarding, and automated reporting for GenAI systems.
---

# Analytics & BI

## Overview
Observing the *technical* health of a system is not enough. Analytics focuses on the *business* value and behavior of the GenAI application (costs, user patterns, content quality).

## Key Technologies

### Data Analysis
- **Pandas**: Standard for small-to-medium datasets.
- **Polars**: High-performance, multithreaded alternative for larger datasets.
- **DuckDB**: In-process OLAP database for fast analytical queries.

### Visualization & Dashboards
- **Plotly Dash**: Production-grade, highly customizable dashboards in Python.
- **Streamlit**: Rapid prototyping for data apps and internal tools.
- **Superset / Metabase**: Full-featured BI platforms connecting to your SQL warehouse.

## Implementation Patterns

### 1. Cost & Usage Tracking
Log token usage, model names, and latency for every request to build cost allocation reports.

```sql
SELECT 
    model, 
    SUM(input_tokens) as input, 
    SUM(output_tokens) as output,
    SUM(cost_usd) as total_cost
FROM llm_logs
GROUP BY model;
```

### 2. Automated Reporting
Generate weekly insights using the LLM itself to analyze metrics.
*Flow: Query Metrics -> Formatting -> LLM Analysis -> PDF/Email Report.*

## Best Practices
1.  **Separate OLTP and OLAP**: Don't run heavy analytic queries on your main production database. Use a read replica or a data warehouse.
2.  **Privacy First**: Mask PII before loading data into analytics systems.
3.  **Snapshotting**: Periodically snapshot operational data for historical trend analysis.

## External Resources
- [Polars Documentation](https://pola.rs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Dash](https://dash.plotly.com/)
