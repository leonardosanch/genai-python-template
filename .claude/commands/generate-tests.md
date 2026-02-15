---
description: Scaffold test file for a source file (mocks async, fixtures, layer-aware)
---

1. Read the source file provided by the user (e.g., `src/application/services/my_service.py`).
2. Read [docs/skills/testing_quality.md](file:///home/leo/templates/genai-python-template/docs/skills/testing_quality.md) for testing patterns.
3. Determine the architectural layer (`domain`, `application`, `infrastructure`, `interfaces`) from the file path.
4. Generate a test file at the corresponding path under `tests/unit/` mirroring the source structure.
5. Apply layer-specific conventions:
   - **domain**: Pure unit tests, no mocks, no I/O.
   - **application**: Mock infrastructure ports (repositories, LLM clients). Use `AsyncMock` for async dependencies.
   - **infrastructure**: Integration-style tests with fixtures for external systems (DB, Redis, APIs). Mark with `@pytest.mark.integration`.
   - **interfaces**: Test routes with `httpx.AsyncClient` and `app` fixture. Validate status codes, response schemas, and error cases.
6. Include:
   - `pytest` and `pytest-asyncio` imports as needed.
   - Fixtures for dependency injection (following the project's container pattern).
   - At least one happy-path test and one error/edge-case test per public method.
   - `AsyncMock` for all async dependencies.
   - Descriptive test names: `test_<method>_<scenario>_<expected_result>`.
7. Do NOT generate tests for private methods or trivial getters/setters.
8. Output the complete test file ready to run with `uv run pytest`.
