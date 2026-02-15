"""User roles for role-based access control (RBAC)."""

from enum import StrEnum


class Role(StrEnum):
    """Application roles included in JWT claims under the ``roles`` key.

    Roles follow a simple hierarchy for this template:
    - VIEWER: read-only access
    - OPERATOR: can trigger pipelines, queries, and streams
    - ADMIN: full access including configuration and analytics
    """

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
