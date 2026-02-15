# src/interfaces/api/routes/analytics_routes.py
"""Analytics API routes for LLM cost tracking."""

from typing import Any

from fastapi import APIRouter, Depends, Request

from src.domain.value_objects.role import Role
from src.interfaces.api.dependencies.rbac import require_roles

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/costs", dependencies=[Depends(require_roles(Role.ADMIN))])
async def get_costs(
    request: Request,
    days: int = 30,
    group_by: str = "model",
) -> dict[str, Any]:
    """Get LLM cost summary.

    Query params:
        days: Number of days to look back (default 30).
        group_by: Grouping field — model, use_case, or day.
    """
    container = request.app.state.container
    tracker = container.cost_tracker
    total = await tracker.get_total_cost(days=days)
    breakdown = await tracker.get_summary(days=days, group_by=group_by)
    return {
        "days": days,
        "group_by": group_by,
        "total_cost_usd": total,
        "breakdown": breakdown,
    }
