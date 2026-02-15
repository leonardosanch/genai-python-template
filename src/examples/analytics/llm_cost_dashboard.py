"""
LLM Cost Tracking Dashboard.

Demonstrates:
- Token usage tracking per model/endpoint
- Cost calculation with provider pricing
- Daily/weekly/monthly aggregations
- Plotly Dash dashboard
- Cost anomaly detection

Run: uv run python -m src.examples.analytics.llm_cost_dashboard

Dashboard available at: http://localhost:8050
"""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Optional: Plotly Dash for dashboard
try:
    import dash  # type: ignore
    import plotly.graph_objs as go  # type: ignore
    from dash import dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


# Model pricing (per 1K tokens)
MODEL_PRICING = {
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}


class DailyCost(BaseModel):
    """Daily cost summary."""

    date: str
    total_cost: float
    total_requests: int
    total_tokens: int


class CostAnomaly(BaseModel):
    """Detected cost anomaly."""

    timestamp: datetime
    cost: float
    expected_cost: float
    deviation_pct: float


class LLMCostTracker:
    """Track LLM usage and costs."""

    def __init__(self, db_path: Path = Path("llm_costs.db")):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                user_id TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        endpoint: str,
        user_id: str | None = None,
    ) -> None:
        """Log LLM usage."""
        if model not in MODEL_PRICING:
            raise ValueError(f"Unknown model: {model}")

        pricing = MODEL_PRICING[model]
        cost = (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO llm_usage (model, endpoint, input_tokens, output_tokens, cost, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model, endpoint, input_tokens, output_tokens, cost, user_id),
        )

        conn.commit()
        conn.close()

    async def get_daily_costs(self, days: int = 30) -> list[DailyCost]:
        """Get daily cost aggregations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                DATE(timestamp) as date,
                SUM(cost) as total_cost,
                COUNT(*) as total_requests,
                SUM(input_tokens + output_tokens) as total_tokens
            FROM llm_usage
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            (days,),
        )

        results = [
            DailyCost(
                date=row[0],
                total_cost=row[1],
                total_requests=row[2],
                total_tokens=row[3],
            )
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    async def get_costs_by_model(self) -> dict[str, float]:
        """Get costs grouped by model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT model, SUM(cost)
            FROM llm_usage
            GROUP BY model
            """
        )

        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return results

    async def detect_anomalies(self, threshold: float = 2.0) -> list[CostAnomaly]:
        """Detect cost anomalies."""
        daily_costs = await self.get_daily_costs(days=30)

        if len(daily_costs) < 7:
            return []  # Not enough data

        # Calculate average and std dev
        costs = [c.total_cost for c in daily_costs]
        avg_cost = sum(costs) / len(costs)

        anomalies = []
        for cost_record in daily_costs:
            if cost_record.total_cost > avg_cost * threshold:
                anomalies.append(
                    CostAnomaly(
                        timestamp=datetime.fromisoformat(cost_record.date),
                        cost=cost_record.total_cost,
                        expected_cost=avg_cost,
                        deviation_pct=((cost_record.total_cost - avg_cost) / avg_cost) * 100,
                    )
                )

        return anomalies


def run_dashboard() -> None:
    """Run Plotly Dash dashboard."""
    if not DASH_AVAILABLE:
        print("❌ Dash not installed. Install with: uv pip install dash plotly")
        return

    app = dash.Dash(__name__)
    tracker = LLMCostTracker()

    app.layout = html.Div(
        [
            html.H1("LLM Cost Dashboard"),
            dcc.Graph(id="daily-cost-chart"),
            dcc.Graph(id="model-breakdown"),
            dcc.Interval(id="interval", interval=60000),  # Update every minute
        ]
    )

    @app.callback(  # type: ignore
        dash.dependencies.Output("daily-cost-chart", "figure"),
        [dash.dependencies.Input("interval", "n_intervals")],
    )
    def update_daily_chart(n: int) -> Any:
        daily_costs = asyncio.run(tracker.get_daily_costs(days=30))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[c.date for c in daily_costs],
                y=[c.total_cost for c in daily_costs],
                mode="lines+markers",
                name="Daily Cost",
            )
        )

        fig.update_layout(title="Daily LLM Costs", xaxis_title="Date", yaxis_title="Cost ($)")
        return fig

    @app.callback(  # type: ignore
        dash.dependencies.Output("model-breakdown", "figure"),
        [dash.dependencies.Input("interval", "n_intervals")],
    )
    def update_model_breakdown(n: int) -> Any:
        costs_by_model = asyncio.run(tracker.get_costs_by_model())

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(costs_by_model.keys()),
                    values=list(costs_by_model.values()),
                )
            ]
        )

        fig.update_layout(title="Cost by Model")
        return fig

    app.run_server(debug=True, port=8050)


async def main() -> None:
    """Example usage."""
    tracker = LLMCostTracker()

    print("💰 LLM Cost Tracker Example\n")

    # Log some usage
    await tracker.log_usage("gpt-4-turbo", 1000, 500, "/chat/completions", "user123")
    await tracker.log_usage("gpt-3.5-turbo", 500, 200, "/chat/completions", "user456")

    # Get daily costs
    daily_costs = await tracker.get_daily_costs(days=7)
    print(f"Daily costs (last 7 days): {len(daily_costs)} records")

    # Get costs by model
    costs_by_model = await tracker.get_costs_by_model()
    print("\nCosts by model:")
    for model, cost in costs_by_model.items():
        print(f"  {model}: ${cost:.4f}")

    # Detect anomalies
    anomalies = await tracker.detect_anomalies()
    print(f"\nAnomalies detected: {len(anomalies)}")


if __name__ == "__main__":
    asyncio.run(main())
