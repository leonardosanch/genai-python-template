"""Role-based access control dependency for FastAPI routes.

Usage in a route::

    from src.domain.value_objects.role import Role
    from src.interfaces.api.dependencies.rbac import require_roles

    @router.get("/admin-only", dependencies=[Depends(require_roles(Role.ADMIN))])
    async def admin_endpoint() -> dict: ...

    @router.post("/operate", dependencies=[Depends(require_roles(Role.OPERATOR, Role.ADMIN))])
    async def operator_endpoint() -> dict: ...

The dependency reads ``request.state.user`` (injected by JWTAuthMiddleware)
and checks the ``roles`` claim against the required roles.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request, status

from src.domain.value_objects.role import Role


def require_roles(
    *allowed: Role,
) -> Callable[[Request], Coroutine[Any, Any, dict[str, Any]]]:
    """Return a FastAPI dependency that enforces role-based access.

    Args:
        *allowed: One or more roles that are permitted to access the route.
                  The user must have **at least one** of the listed roles.

    Raises:
        HTTPException 401: If no authenticated user is found on the request.
        HTTPException 403: If the user lacks the required role(s).
    """

    async def _check(request: Request) -> dict[str, Any]:
        user: dict[str, Any] | None = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        user_roles: list[str] = user.get("roles", [])
        if not any(role.value in user_roles for role in allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[r.value for r in allowed]}",
            )

        return user

    return _check
