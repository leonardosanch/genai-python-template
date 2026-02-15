# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Performance Profiling skill (`docs/skills/performance_profiling.md`) — Python profiling (py-spy, scalene, memray), async event loop monitoring, LLM performance tracking, Locust load testing, PostgreSQL query analysis, pytest-benchmark examples
- Design Patterns skill (`docs/skills/design_patterns.md`) — GoF patterns in Python (Adapter, Strategy, Factory, Decorator, Proxy, Observer, Command, Chain of Responsibility), GenAI-specific pattern decision trees, 8 code examples, anti-patterns
- Disaster Recovery section in cloud infrastructure skill — RTO/RPO tiers, Route53 failover (Terraform), cross-region backup (K8s CronJob)
- Cloud Migration section — 7Rs framework decision tree, migration waves planning, validation checklist
- Hybrid Cloud section — connectivity options comparison, workload placement decision tree
- Landing Zone section — AWS account structure, Terraform modules (organization, guardrails, logging, networking)
- FinOps section — budget alerts (Terraform), tagging enforcement, cost optimization strategies
- New anti-patterns: no DR testing, big bang migration, no cost visibility, single-account
- New checklists: DR, FinOps, Landing Zone, infrastructure
- New external resources: FinOps Foundation, AWS DR Whitepaper, Landing Zone Accelerator, Migration Hub

### Changed
- Expanded cloud infrastructure skill from 425 to 898 lines
- Updated skill count from 15 to 26 in documentation
- Registered Performance Profiling in CLAUDE.md (table + documentation index + global CLAUDE.md)

### Fixed
- Fixed 10 broken file paths in slash commands (hardcoded `/home/leo/templates/...` → relative paths)

## [0.1.0] — Initial Release

### Added
- Clean Architecture project structure (domain, application, infrastructure, interfaces)
- OpenAI and LiteLLM adapters with unified LLMPort interface
- RAG pipeline with vector store integration
- Data engineering scaffolding (ETL, validation, data cleaning pipeline)
- Auth middleware (API key) and rate limiting middleware (token bucket)
- MCP client via stdio JSON-RPC
- S3 storage adapter via aiobotocore
- WebSocket chat endpoint
- CLI commands: generate, stream, etl, validate
- FastAPI application with health, ready, and pipeline endpoints
- 15 specialized skill documentation files
- CI/CD workflow with ruff, mypy, pytest
- Pre-commit hooks configuration
- Terraform deployment scaffold
- SonarQube configuration
- Setup scripts for global and per-project configuration
- Comprehensive setup guide (SETUP_GUIDE.md)
