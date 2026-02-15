# Examples Directory

This directory contains runnable code examples demonstrating best practices from the skills documentation.

## Available Examples

### 1. RAG Pipeline (`rag_pipeline.py`)
Complete RAG implementation with:
- Document loading and chunking
- Vector store integration (Pinecone)
- LLM integration (OpenAI)
- Evaluation with RAGAS

**Run**: `uv run python examples/rag_pipeline.py`

### 2. Multi-Agent Supervisor (`multi_agent_supervisor.py`)
LangGraph supervisor pattern with:
- Supervisor agent routing
- Specialized worker agents
- State checkpointing
- Error handling

**Run**: `uv run python examples/multi_agent_supervisor.py`

### 3. FastAPI Streaming (`fastapi_streaming.py`)
SSE streaming for LLM responses:
- Server-Sent Events implementation
- Error handling
- OpenTelemetry instrumentation

**Run**: `uv run uvicorn examples.fastapi_streaming:app --reload`

### 4. Docker GenAI App (`docker_genai_app/`)
Production-ready containerization:
- Multi-stage Dockerfile
- docker-compose.yml with dependencies
- Kubernetes manifests

**Build**: `cd docker_genai_app && docker-compose up`

## Prerequisites

```bash
# Install dependencies
uv sync

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Running Examples

All examples are self-contained and can be run independently. Make sure to set the required environment variables in `.env`:

```bash
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_ENVIRONMENT=your_env_here
```

## Testing Examples

```bash
# Run tests for examples
uv run pytest tests/examples/
```
