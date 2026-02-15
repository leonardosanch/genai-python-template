---
description: Validate Clean Architecture dependency rules across src/
---

1. Read [docs/skills/software_architecture.md](file:///home/leo/templates/genai-python-template/docs/skills/software_architecture.md) for architecture rules.
2. Scan all Python files under `src/` and check these dependency rules:

   **Allowed dependencies (inner -> outer is FORBIDDEN):**
   - `domain/` must NOT import from `application/`, `infrastructure/`, or `interfaces/`.
   - `application/` must NOT import from `infrastructure/` or `interfaces/`.
   - `interfaces/` must NOT import from `infrastructure/` directly (use DI).

   **Specifically check for:**
   - `domain/` importing `fastapi`, `sqlalchemy`, `redis`, `httpx`, `langchain`, or any framework.
   - `application/` importing concrete infrastructure classes (should use ports/interfaces).
   - `interfaces/` importing repository implementations directly.
   - Circular imports between layers.

3. For each violation found, report:
   - File path and line number.
   - The offending import.
   - Which dependency rule it breaks.
   - Suggested fix (e.g., "inject via port", "move to infrastructure layer").

4. Summarize results:
   - Total files scanned.
   - Number of violations per layer.
   - Overall architecture health: CLEAN / NEEDS ATTENTION / CRITICAL.

5. If no violations found, confirm the architecture is clean.
