"""Tests for Role value object."""

from src.domain.value_objects.role import Role


class TestRole:
    def test_values(self) -> None:
        assert Role.VIEWER == "viewer"
        assert Role.OPERATOR == "operator"
        assert Role.ADMIN == "admin"

    def test_from_string(self) -> None:
        assert Role("viewer") is Role.VIEWER
        assert Role("admin") is Role.ADMIN

    def test_all_members(self) -> None:
        assert set(Role) == {Role.VIEWER, Role.OPERATOR, Role.ADMIN}
