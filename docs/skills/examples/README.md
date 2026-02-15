# Skill Examples Index

This directory contains pointers to full examples in `src/examples/`.

## Mapping: Skill → Example

| Skill | Example | Path | Description |
|-------|---------|------|-------------|
| **hallucination_detection.md** | DeepEval Hallucination Detection | [src/examples/evaluation/deepeval_hallucination.py](file:///home/leo/genai-python-template/src/examples/evaluation/deepeval_hallucination.py) | Faithfulness, hallucination, and relevancy metrics |
| **hallucination_detection.md** | RAGAS Evaluation | [src/examples/evaluation/ragas_evaluation.py](file:///home/leo/genai-python-template/src/examples/evaluation/ragas_evaluation.py) | End-to-end RAG evaluation with all RAGAS metrics |
| **hallucination_detection.md** | LLM-as-Judge | [src/examples/evaluation/llm_as_judge.py](file:///home/leo/genai-python-template/src/examples/evaluation/llm_as_judge.py) | Pairwise comparison and rubric-based evaluation |
| **hallucination_detection.md** | Regression Testing | [src/examples/evaluation/regression_testing.py](file:///home/leo/genai-python-template/src/examples/evaluation/regression_testing.py) | Golden dataset management and drift detection |
| **prompt_engineering.md** | Prompt Templates | [src/examples/supporting/prompt_templates.py](file:///home/leo/genai-python-template/src/examples/supporting/prompt_templates.py) | Jinja2 templates with Pydantic validation |
| **context_engineering.md** | Semantic Blueprint | [src/examples/context_engineering/semantic_blueprint.py](file:///home/leo/genai-python-template/src/examples/context_engineering/semantic_blueprint.py) | Priority-based context selection with token budgets |
| **context_engineering.md** | Context Compression | [src/examples/context_engineering/context_compression.py](file:///home/leo/genai-python-template/src/examples/context_engineering/context_compression.py) | LLM-based and extractive summarization |
| **context_engineering.md** | Memory Management | [src/examples/context_engineering/memory_management.py](file:///home/leo/genai-python-template/src/examples/context_engineering/memory_management.py) | Sliding window, summary, and entity memory |
| **context_engineering.md** | Context Router | [src/examples/context_engineering/context_router.py](file:///home/leo/genai-python-template/src/examples/context_engineering/context_router.py) | Intent-based routing and caching |
| **observability_monitoring.md** | Cost Dashboard | [src/examples/analytics/llm_cost_dashboard.py](file:///home/leo/genai-python-template/src/examples/analytics/llm_cost_dashboard.py) | Real-time cost tracking with Plotly Dash |
| **genai_rag.md** | Naive RAG | [src/examples/rag/naive_rag.py](file:///home/leo/genai-python-template/src/examples/rag/naive_rag.py) | Basic RAG implementation |
| **genai_rag.md** | Advanced RAG | [src/examples/rag/advanced_rag.py](file:///home/leo/genai-python-template/src/examples/rag/advanced_rag.py) | RAG with reranking and hybrid search |
| **genai_rag.md** | Modular RAG | [src/examples/rag/modular_rag.py](file:///home/leo/genai-python-template/src/examples/rag/modular_rag.py) | Modular RAG architecture |
| **genai_rag.md** | GraphRAG | [src/examples/rag/graph_rag.py](file:///home/leo/genai-python-template/src/examples/rag/graph_rag.py) | Knowledge graph-based RAG |
| **genai_rag.md** | GRaR Agent | [src/examples/rag/grar_agent.py](file:///home/leo/genai-python-template/src/examples/rag/grar_agent.py) | Graph-Retrieval-Augmented Reasoning |
| **genai_rag.md** | Semantic Cache | [src/examples/rag/semantic_cache.py](file:///home/leo/genai-python-template/src/examples/rag/semantic_cache.py) | Semantic caching for cost optimization |
| **multi_agent_systems.md** | Single Agent | [src/examples/agents/single_agent.py](file:///home/leo/genai-python-template/src/examples/agents/single_agent.py) | Basic agent with tool use |
| **multi_agent_systems.md** | Supervisor Pattern | [src/examples/agents/supervisor_pattern.py](file:///home/leo/genai-python-template/src/examples/agents/supervisor_pattern.py) | Supervisor routing to worker agents |
| **multi_agent_systems.md** | Hierarchical Agents | [src/examples/agents/hierarchical_agents.py](file:///home/leo/genai-python-template/src/examples/agents/hierarchical_agents.py) | Multi-level agent hierarchy |
| **multi_agent_systems.md** | Sequential Pipeline | [src/examples/agents/sequential_pipeline.py](file:///home/leo/genai-python-template/src/examples/agents/sequential_pipeline.py) | Sequential agent workflow |
| **multi_agent_systems.md** | Debate Pattern | [src/examples/agents/debate_pattern.py](file:///home/leo/genai-python-template/src/examples/agents/debate_pattern.py) | Multi-agent debate for consensus |
| **multi_agent_systems.md** | Agentic RAG | [src/examples/agents/agentic_rag.py](file:///home/leo/genai-python-template/src/examples/agents/agentic_rag.py) | Agent that dynamically decides when to retrieve |
| **api_streaming.md** | FastAPI SSE | [src/examples/supporting/streaming.py](file:///home/leo/genai-python-template/src/examples/supporting/streaming.py) | Server-Sent Events streaming |
| **api_streaming.md** | RAG Endpoints | [src/examples/api/rag_endpoints.py](file:///home/leo/genai-python-template/src/examples/api/rag_endpoints.py) | FastAPI endpoints for RAG |
| **databases.md** | Vector Stores | [src/examples/supporting/vector_stores.py](file:///home/leo/genai-python-template/src/examples/supporting/vector_stores.py) | Multiple vector store implementations |
| **testing_quality.md** | RAGAS Evaluation | [src/examples/supporting/ragas_evaluation.py](file:///home/leo/genai-python-template/src/examples/supporting/ragas_evaluation.py) | RAG evaluation with RAGAS |
| **observability_monitoring.md** | OpenTelemetry | [src/examples/supporting/observability.py](file:///home/leo/genai-python-template/src/examples/supporting/observability.py) | Tracing and metrics |
| **data_ml_engineering.md** | Hybrid ML+LLM | [src/examples/ml/hybrid_classifier_llm.py](file:///home/leo/genai-python-template/src/examples/ml/hybrid_classifier_llm.py) | Classical ML classifier + LLM generation |
| **automation.md** | RAG CLI | [src/examples/cli/rag_cli.py](file:///home/leo/genai-python-template/src/examples/cli/rag_cli.py) | Command-line interface for RAG |
| **event_driven_systems.md** | Celery Tasks | [src/examples/integration/celery_tasks.py](file:///home/leo/genai-python-template/src/examples/integration/celery_tasks.py) | Async task processing |
| **event_driven_systems.md** | Complete App | [src/examples/integration/complete_app.py](file:///home/leo/genai-python-template/src/examples/integration/complete_app.py) | Full application integration |

## Running Examples

All examples can be run with:

```bash
uv run python -m src.examples.<category>.<example_name>
```

### Examples by Category

#### Evaluation (`src/examples/evaluation/`)
```bash
uv run python -m src.examples.evaluation.deepeval_hallucination
uv run python -m src.examples.evaluation.ragas_evaluation
uv run python -m src.examples.evaluation.llm_as_judge
uv run python -m src.examples.evaluation.regression_testing
```

#### Context Engineering (`src/examples/context_engineering/`)
```bash
uv run python -m src.examples.context_engineering.semantic_blueprint
uv run python -m src.examples.context_engineering.context_compression
uv run python -m src.examples.context_engineering.memory_management
uv run python -m src.examples.context_engineering.context_router
```

#### Analytics (`src/examples/analytics/`)
```bash
uv run python -m src.examples.analytics.llm_cost_dashboard
```

#### Machine Learning (`src/examples/ml/`)
```bash
uv run python -m src.examples.ml.hybrid_classifier_llm
```

#### RAG (`src/examples/rag/`)
```bash
uv run python -m src.examples.rag.naive_rag
uv run python -m src.examples.rag.advanced_rag
uv run python -m src.examples.rag.modular_rag
uv run python -m src.examples.rag.graph_rag
uv run python -m src.examples.rag.grar_agent
uv run python -m src.examples.rag.semantic_cache
```

#### Agents (`src/examples/agents/`)
```bash
uv run python -m src.examples.agents.single_agent
uv run python -m src.examples.agents.supervisor_pattern
uv run python -m src.examples.agents.hierarchical_agents
uv run python -m src.examples.agents.sequential_pipeline
uv run python -m src.examples.agents.debate_pattern
uv run python -m src.examples.agents.agentic_rag
```

#### Supporting (`src/examples/supporting/`)
```bash
uv run python -m src.examples.supporting.streaming
uv run python -m src.examples.supporting.vector_stores
uv run python -m src.examples.supporting.ragas_evaluation
uv run python -m src.examples.supporting.observability
uv run python -m src.examples.supporting.prompt_templates
```

## Skills Documentation

All skills are located in `docs/skills/`:

- [hallucination_detection.md](file:///home/leo/genai-python-template/docs/skills/hallucination_detection.md) - Detecting and mitigating LLM hallucinations
- [prompt_engineering.md](file:///home/leo/genai-python-template/docs/skills/prompt_engineering.md) - Prompt versioning and testing
- [observability_monitoring.md](file:///home/leo/genai-python-template/docs/skills/observability_monitoring.md) - Production observability and alerting
- [multi_tenancy.md](file:///home/leo/genai-python-template/docs/skills/multi_tenancy.md) - Multi-tenant isolation patterns
- [context_engineering.md](file:///home/leo/genai-python-template/docs/skills/context_engineering.md) - Context management strategies
- [genai_rag.md](file:///home/leo/genai-python-template/docs/skills/genai_rag.md) - RAG patterns and best practices
- [multi_agent_systems.md](file:///home/leo/genai-python-template/docs/skills/multi_agent_systems.md) - Multi-agent coordination
- [api_streaming.md](file:///home/leo/genai-python-template/docs/skills/api_streaming.md) - Streaming and async patterns
- [databases.md](file:///home/leo/genai-python-template/docs/skills/databases.md) - Database and vector store patterns
- [testing_quality.md](file:///home/leo/genai-python-template/docs/skills/testing_quality.md) - Testing strategies
- [observability_monitoring.md](file:///home/leo/genai-python-template/docs/skills/observability_monitoring.md) - Observability best practices
- [data_ml_engineering.md](file:///home/leo/genai-python-template/docs/skills/data_ml_engineering.md) - Data and ML patterns
- [automation.md](file:///home/leo/genai-python-template/docs/skills/automation.md) - CLI and automation
- [event_driven_systems.md](file:///home/leo/genai-python-template/docs/skills/event_driven_systems.md) - Event-driven architecture
- [cloud_infrastructure.md](file:///home/leo/genai-python-template/docs/skills/cloud_infrastructure.md) - Cloud deployment patterns
- [security.md](file:///home/leo/genai-python-template/docs/skills/security.md) - Security best practices
- [software_architecture.md](file:///home/leo/genai-python-template/docs/skills/software_architecture.md) - Architecture patterns

## Contributing

When adding new examples:

1. Place in appropriate category under `src/examples/`
2. Add entry to this README mapping to relevant skill(s)
3. Include docstring with usage instructions
4. Add tests in `tests/examples/`
5. Update skill documentation if needed
