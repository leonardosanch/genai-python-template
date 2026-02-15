# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
