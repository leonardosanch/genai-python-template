---
description: Scaffold FastAPI endpoint (route + DTO + use case) following Clean Architecture
---

1. Ask the user for: resource name, HTTP method(s), and brief description of the endpoint's purpose.
2. Read `docs/skills/api_streaming.md` for API patterns.
3. Read `docs/skills/software_architecture.md` for layer rules.
4. Create or update the following files:

   **a. DTO (request/response schemas)**
   - Path: `src/application/dtos/<resource>.py`
   - Pydantic `BaseModel` classes for request and response.
   - Strict validation, explicit field types, examples in `model_config`.

   **b. Use Case (application layer)**
   - Path: `src/application/use_cases/<resource>_use_case.py`
   - Single `execute` method (async).
   - Receives DTO, returns DTO. No framework imports.
   - Dependencies injected via constructor (ports/interfaces).

   **c. Route (interface layer)**
   - Path: `src/interfaces/api/routes/<resource>_routes.py`
   - FastAPI `APIRouter` with proper tags and prefix (`/api/v1/<resource>`).
   - Dependency injection via `Depends` wiring to the container.
   - Proper HTTP status codes, error handling with `HTTPException`.

   **d. Register route**
   - Update `src/interfaces/api/main.py` to include the new router.

5. Follow these rules:
   - Route layer ONLY handles HTTP concerns (parsing, status codes, headers).
   - Use case ONLY handles business orchestration.
   - No business logic in routes. No HTTP concerns in use cases.
   - All async.
6. Output all files ready to use.
