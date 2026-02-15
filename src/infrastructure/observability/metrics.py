"""OpenTelemetry metrics for LLM usage tracking.

Reference implementation for counters and histograms
specific to GenAI systems.
"""

from opentelemetry import metrics

meter = metrics.get_meter("genai-app")

# LLM metrics
llm_request_counter = meter.create_counter(
    "llm.requests.total",
    description="Total LLM requests",
)

llm_error_counter = meter.create_counter(
    "llm.requests.errors",
    description="Total failed LLM requests",
)

llm_latency_histogram = meter.create_histogram(
    "llm.latency",
    description="LLM request latency",
    unit="ms",
)

llm_token_counter = meter.create_counter(
    "llm.tokens.used",
    description="Total tokens consumed",
)

llm_cost_counter = meter.create_counter(
    "llm.cost.usd",
    description="Total LLM cost in USD",
)

# RAG metrics
rag_retrieval_latency = meter.create_histogram(
    "rag.retrieval.latency",
    description="Document retrieval latency",
    unit="ms",
)

rag_documents_retrieved = meter.create_histogram(
    "rag.retrieval.documents",
    description="Number of documents retrieved per query",
)

# Agent metrics
agent_steps_counter = meter.create_counter(
    "agent.steps.total",
    description="Total agent steps executed",
)
