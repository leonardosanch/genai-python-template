---
description: Perform a security review of the current file using OWASP Top 10 and best practices
---

1.  **Analyze the current file** for common security vulnerabilities:
    -   **Injection** (SQL, Command, Prompt)
    -   **Broken Authentication**
    -   **Sensitive Data Exposure** (Hardcoded secrets, PII logging)
    -   **Insecure Deserialization** (pickle, yaml)
    -   **Insufficient Logging/Monitoring**

2.  **Consult the Security Skill**: Read `docs/skills/security.md` for specific checklists.

3.  **Check for Python-specific issues**:
    -   Use of `eval()`, `exec()`, or `subprocess` without validation.
    -   Insecure temp file usage.
    -   Weak cryptography or random number generation (`random` vs `secrets`).

4.  **Report Findings**:
    -   List each potential vulnerability with line number.
    -   Assess severity (Critical, High, Medium, Low).
    -   Suggest a specific fix for each issue.

5.  **If no issues found**, explicitly state: "No obvious security vulnerabilities found based on OWASP Top 10 guidelines."
