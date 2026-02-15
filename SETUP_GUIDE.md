# Setup Guide — Claude Code + GenAI Template

This guide explains how to configure Claude Code with the engineering rules
and specialized skills included in this template.

There are two scenarios:
1. **Creating a new project from scratch** using this template
2. **Working on an existing project** (e.g., a company repo you cloned)

Both scenarios are covered below.

---

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed
- [uv](https://docs.astral.sh/uv/) installed (Python package manager)
- Git installed

---

## Scenario 1: Creating a New Project

You want to start a new project (e.g., `sistema_pos`) using the template's
architecture, rules, and skills.

### Step 1 — Clone the template (once)

```bash
git clone <repo-url> ~/templates/genai-python-template
```

### Step 2 — Create your project

```bash
cd ~/templates/genai-python-template
./scripts/new-project.sh ~/proyectos/sistema_pos
```

This copies:
- `CLAUDE.md` with project rules and skill references (relative paths)
- `docs/skills/` — 19 specialized skill files
- `.claude/` — commands, hooks, settings
- `.github/workflows/` — CI/CD configuration
- `deploy/` — Terraform scaffold
- `pyproject.toml`, `.env.example`, `.pre-commit-config.yaml`
- Clean `src/` directory structure (Clean Architecture layout)
- Empty `tests/unit/` and `tests/integration/` directories

It does NOT copy example code. You start with a clean project.

If you want the example code as reference:

```bash
./scripts/new-project.sh ~/proyectos/sistema_pos --with-examples
```

### Step 3 — Start working

```bash
cd ~/proyectos/sistema_pos
uv sync
claude
```

Claude Code reads `./CLAUDE.md` and has access to all 19 skills via
relative paths inside `docs/skills/`.

**This setup is fully portable** — anyone who clones your project gets
the same rules and skills automatically.

---

## Scenario 2: Working on an Existing Company Project

You are assigned to work on a project that already exists (e.g., `inventory-api`
from your company). This project has its own repo and does NOT include
the template's rules or skills.

### Step 1 — Clone the template (once)

```bash
git clone <repo-url> ~/templates/genai-python-template
```

### Step 2 — Install global configuration

```bash
cd ~/templates/genai-python-template
./scripts/setup-global.sh
```

This installs `~/.claude/CLAUDE.md` with:
- All engineering rules (SOLID, Clean Architecture, security, etc.)
- References to the 19 skills using absolute paths to your template directory

You only run this **once per machine**.

### Step 3 — Clone the company project

```bash
git clone git@company.com/team/inventory-api.git ~/projects/inventory-api
```

### Step 4 — Start working

```bash
cd ~/projects/inventory-api
claude
```

Claude Code automatically loads `~/.claude/CLAUDE.md` (global) in every
directory. You get all the engineering rules and skills without copying
anything into the company project.

### How it works

```
Claude Code loads (in order):

1. ~/.claude/CLAUDE.md          ← Global rules + 19 skills (absolute paths)
2. ~/projects/inventory-api/CLAUDE.md  ← Project rules (if it has one)

Both are merged. The project file adds to the global, it does not replace it.
```

### Important

- Do **NOT** move or rename the template directory after running `setup-global.sh`.
  The global config references it with absolute paths.
- If you move the template, run `setup-global.sh` again from the new location.

---

## Scenario 2b: Adding Project Context (Recommended)

When you work on a specific project, Claude Code knows the engineering rules
(from the global config) but does NOT know anything about **that project**:
what it does, what stack it uses, what the business rules are.

You can give it that context by creating a `CLAUDE.md` inside the project.
This is **optional but highly recommended** — it makes Claude Code significantly
more effective because it understands the business domain, not just the tech.

### What to put in the project CLAUDE.md

| Section | What to include | Example |
|---------|----------------|---------|
| **Context** | What the project is, who uses it | "Backend for the company's payroll system" |
| **Stack** | Technologies and versions | "FastAPI + PostgreSQL + Redis + Celery" |
| **Business rules** | Domain constraints Claude must respect | "Monetary values always Decimal, never float" |
| **User stories** | Current sprint stories or backlog | "HU-042: Generate payroll reports in PDF" |
| **Team decisions** | Architecture choices already made | "Auth: JWT with refresh tokens" |
| **Conventions** | Naming, structure, patterns specific to this project | "All endpoints under /api/v1/inventory/" |

### Full example

```bash
cd ~/projects/inventory-api
```

Create `CLAUDE.md`:

```markdown
# Inventory API — Warehouse Management

## Context
Backend for the company's inventory system. Handles products, stock levels,
purchase orders, and warehouse operations.
Used by warehouse staff and procurement (internal tool, not public-facing).

## Stack
- FastAPI + PostgreSQL 16 + Redis 7
- Celery for background jobs (report generation, bulk imports)
- Alembic for migrations
- Auth: JWT with refresh tokens (company SSO integration)

## Business Rules
- All monetary values use Decimal with 2 decimal places, never float
- Stock levels cannot go negative
- Audit trail required for all write operations (who, when, what changed)
- Multi-warehouse: every query must filter by warehouse_id
- Reorder alerts when stock falls below minimum threshold

## User Stories (Current Sprint)
- US-101: As a manager, I want to generate inventory reports in PDF
- US-102: As warehouse staff, I want to scan barcodes to update stock
- US-103: As procurement, I want to create purchase orders from low-stock items
- US-104: As admin, I want to bulk import products from CSV

## Team Decisions
- Read replica for reporting queries (never heavy queries on primary)
- Celery for all PDF/CSV generation (async, never in request path)
- All API responses include pagination (max 100 items per page)
- Error codes follow company standard: INV-XXX format

## Conventions
- Endpoints: /api/v1/inventory/...
- Branch naming: feature/US-XXX-short-description
- Commit messages: "US-XXX: description"
```

### How Claude Code uses this

When you write:

```
claude "implement US-101"
```

Claude Code already knows:
- **What** US-101 is (generate inventory reports in PDF)
- **How** to do it (Celery for async, read replica for queries)
- **Constraints** (Decimal for money, audit trail, multi-warehouse)
- **Quality standards** (from the 19 global skills)

You don't need to explain any of this in the prompt.

### Keeping it updated

Update the `CLAUDE.md` as the project evolves:
- New sprint? Update the user stories section
- New team decision? Add it to team decisions
- New business rule? Add it to business rules

Think of it as **the project's brain** — the more context it has, the
better Claude Code performs.

### Three layers working together

```
┌─────────────────────────────────────────────────┐
│  CLI prompt                                     │
│  "implement HU-042"                             │
│  → The specific task you want done NOW          │
├─────────────────────────────────────────────────┤
│  Project CLAUDE.md                              │
│  Stack, business rules, user stories            │
│  → WHAT the project is and WHAT to build        │
├─────────────────────────────────────────────────┤
│  Global ~/.claude/CLAUDE.md                     │
│  SOLID, Clean Architecture, 19 skills           │
│  → HOW to build it with quality                 │
└─────────────────────────────────────────────────┘
```

---

## Combining Both Scenarios

You can use both approaches simultaneously:

| Project | Source | Rules loaded |
|---------|--------|-------------|
| `sistema_pos` | Created from template | `./CLAUDE.md` (portable, with relative skill paths) |
| `inventory-api` | Company repo | `~/.claude/CLAUDE.md` (global, with absolute skill paths) |
| Any other dir | Anything | `~/.claude/CLAUDE.md` (global) |

The global config is your safety net — every project on your machine
gets the engineering rules and skills, even if it has no `CLAUDE.md`.

---

## Quick Reference

### Commands

```bash
# Install global config (once per machine)
cd ~/templates/genai-python-template
./scripts/setup-global.sh

# Create new project from template
./scripts/new-project.sh ~/proyectos/my-project

# Create new project with example code
./scripts/new-project.sh ~/proyectos/my-project --with-examples

# Re-install global config (after moving template or updating skills)
./scripts/setup-global.sh --force
```

### File Locations

| File | Purpose |
|------|---------|
| `~/.claude/CLAUDE.md` | Global rules for all projects (installed by `setup-global.sh`) |
| `<project>/CLAUDE.md` | Project-specific rules (created by `new-project.sh` or manually) |
| `<template>/docs/skills/*.md` | 19 specialized skill files |
| `<template>/scripts/setup-global.sh` | Installs global configuration |
| `<template>/scripts/new-project.sh` | Creates new project from template |

### Skills Included (19)

| Skill | File |
|-------|------|
| Software Architecture | `software_architecture.md` |
| Security Engineering | `security.md` |
| GenAI & RAG | `genai_rag.md` |
| Multi-Agent Systems | `multi_agent_systems.md` |
| Data & ML Engineering | `data_ml_engineering.md` |
| API & Streaming | `api_streaming.md` |
| Cloud & Infrastructure | `cloud_infrastructure.md` |
| Testing & Quality | `testing_quality.md` |
| Observability & Monitoring | `observability_monitoring.md` |
| Context Engineering | `context_engineering.md` |
| Prompt Engineering | `prompt_engineering.md` |
| Multi-Tenancy | `multi_tenancy.md` |
| Hallucination Detection | `hallucination_detection.md` |
| Automation & Scripting | `automation.md` |
| Analytics & Reporting | `analytics.md` |
| Databases & Storage | `databases.md` |
| Event-Driven Systems | `event_driven_systems.md` |
| Model Context Protocol | `mcp.md` |
| AI Governance | `governance.md` |

### Skill Validation

Before important development work, validate that skills are up to date:

```bash
# Review the validation process
cat docs/SKILL_VALIDATION_PROMPT.md
```

See [SKILL_VALIDATION_PROMPT.md](docs/SKILL_VALIDATION_PROMPT.md) for the complete process.

---

## Troubleshooting

### Claude Code does not seem to follow the rules
- Verify `~/.claude/CLAUDE.md` exists: `cat ~/.claude/CLAUDE.md | head -5`
- Verify skills are accessible: `ls ~/templates/genai-python-template/docs/skills/`
- If you moved the template, run `./scripts/setup-global.sh --force`

### Skills are not being consulted
- Claude Code reads skills on demand when the task is relevant
- You can explicitly ask: "Consult the Security skill before reviewing this code"

### Conflict between global and project rules
- Project rules take priority when there is a direct contradiction
- In practice, global = universal principles, project = specific context
- They complement each other, conflicts are rare
