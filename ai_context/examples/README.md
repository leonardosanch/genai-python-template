# GenAI Code Examples

Functional, executable examples demonstrating all patterns documented in this template.

## Quick Start

```bash
# Install dependencies
uv sync

# Run an example
uv run python -m src.examples.llm.basic_client

# Run with environment variables
export OPENAI_API_KEY="your-key"
uv run python -m src.examples.rag.naive_rag
```

## Examples by Category

### 🤖 LLM Basics
- **[basic_client.py](llm/basic_client.py)**: Multi-provider LLM client (OpenAI, Anthropic)
- **[structured_output.py](llm/structured_output.py)**: Pydantic-based structured extraction

### 📚 RAG (Retrieval-Augmented Generation)
- **[naive_rag.py](rag/naive_rag.py)**: Simple RAG implementation
- **[advanced_rag.py](rag/advanced_rag.py)**: Production RAG with reranking
- **[modular_rag.py](rag/modular_rag.py)**: Composable RAG pipeline
- **[graphrag_example.py](rag/graphrag_example.py)**: Knowledge graph + vectors
- **[grar_agent.py](rag/grar_agent.py)**: Graph reasoning with agents
- **[semantic_cache.py](rag/semantic_cache.py)**: Vector-based caching

### 🤝 Multi-Agent Systems
- **[single_agent.py](agents/single_agent.py)**: Basic agent with tools
- **[supervisor_pattern.py](agents/supervisor_pattern.py)**: Supervisor + workers
- **[sequential_pipeline.py](agents/sequential_pipeline.py)**: Fixed agent chain
- **[hierarchical_agents.py](agents/hierarchical_agents.py)**: Multi-level hierarchy
- **[debate_pattern.py](agents/debate_pattern.py)**: Collaborative consensus
- **[agentic_rag.py](agents/agentic_rag.py)**: Dynamic retrieval agent

### 🗄️ Vector Stores
- **[chromadb_example.py](vectorstores/chromadb_example.py)**: ChromaDB integration
- **[pgvector_example.py](vectorstores/pgvector_example.py)**: PostgreSQL pgvector

### 📝 Prompts & Templates
- **[template_system.py](prompts/template_system.py)**: Versioned prompt management

### 🌊 Streaming
- **[sse_example.py](streaming/sse_example.py)**: Server-Sent Events
- **[websocket_example.py](streaming/websocket_example.py)**: WebSocket chat

### 📊 Evaluation & Observability
- **[ragas_eval.py](evaluation/ragas_eval.py)**: RAG evaluation with RAGAS
- **[otel_setup.py](observability/otel_setup.py)**: OpenTelemetry tracing

### 🔌 Integrations
- **[rag_endpoints.py](api/rag_endpoints.py)**: FastAPI RAG endpoints
- **[rag_cli.py](cli/rag_cli.py)**: CLI tool with Typer
- **[background_jobs.py](tasks/background_jobs.py)**: Celery tasks
- **[rag_application.py](complete/rag_application.py)**: Complete end-to-end app

## Environment Variables

Most examples require API keys:

```bash
# Required for LLM examples
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional for specific examples
export COHERE_API_KEY="..."  # For reranking
export NEO4J_URI="bolt://localhost:7687"  # For GraphRAG
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

## Running Tests

```bash
# Run all example tests
uv run pytest tests/examples/

# Run specific example test
uv run pytest tests/examples/test_rag_examples.py -v
```

## Architecture

All examples follow Clean Architecture principles:
- **Domain**: Pure business logic (in `src/domain/`)
- **Application**: Use cases (in `src/application/`)
- **Infrastructure**: External systems (in `src/infrastructure/`)
- **Examples**: Demonstrations using the above layers

## Contributing

When adding new examples:
1. Follow the existing structure
2. Add comprehensive docstrings
3. Include usage examples in docstring
4. Add corresponding tests in `tests/examples/`
5. Update this README

## Support

For issues or questions:
- Check the main documentation in project root
- Review the specific pattern docs (RAG.md, AGENTS.md, etc.)
- Open an issue on GitHub
