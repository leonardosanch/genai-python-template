---
description: Review test coverage and quality for the current file
---

1.  **Identify the corresponding test file**:
    -   If checking `src/foo/bar.py`, look for `tests/foo/test_bar.py`.
    -   If test file is missing, flag it immediately.

2.  **Analyze Test Coverage**:
    -   **Happy Path**: Are normal usage scenarios tested?
    -   **Edge Cases**: Are boundary conditions (empty lists, None, huge numbers) tested?
    -   **Error Handling**: Are raised exceptions verified?

3.  **Quality Check**:
    -   Are assertions specific? (Avoid `assert result` without checking value)
    -   Is mocking used appropriately? (Avoid external network calls)
    -   Are tests independent?

4.  **Consult Testing Standards**: Refer to `TESTING.md` for project conventions.

5.  **Report**:
    -   List missing test cases.
    -   Suggestion improvements for existing tests.
    -   If tests are missing entirely, propose a skeleton test file.
