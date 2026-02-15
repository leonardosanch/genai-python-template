import asyncio
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import Counter, Gauge, Histogram

# --- Configuration & Setup ---

# Initialize OpenTelemetry (Mocking exporter for example portability)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
# In production, use OTLPSpanExporter
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Prometheus Metrics
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total", "Total tokens consumed", ["model", "type", "endpoint"]
)
LLM_LATENCY_SECONDS = Histogram(
    "llm_latency_seconds",
    "LLM request latency",
    ["model", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
LLM_TTFT_SECONDS = Histogram(
    "llm_ttft_seconds", "Time to first token", ["model"], buckets=[0.1, 0.2, 0.5, 1.0, 2.0]
)
LLM_COST_USD = Counter("llm_cost_usd_total", "Total LLM cost in USD", ["model", "endpoint"])
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total", "Total LLM requests", ["model", "endpoint", "status"]
)
LLM_ACTIVE_REQUESTS = Gauge("llm_active_requests", "Currently active LLM requests", ["model"])

# --- Utilities ---


def calculate_cost(model: str, tokens: int) -> float:
    """Calculate cost based on model pricing (Example pricing)."""
    pricing = {
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
    }
    # Simplified: assume 50/50 split for total tokens if type not specified
    rate = pricing.get(model, {"input": 0.01 / 1000, "output": 0.01 / 1000})
    return tokens * (rate["input"] + rate["output"]) / 2


# --- Core Functionality ---


class MockOpenAIClient:
    """Mock OpenAI Client for demonstration purposes."""

    class Chat:
        class Completions:
            async def create(self, model: str, messages: list, stream: bool = False):
                class Choice:
                    class Delta:
                        def __init__(self, content):
                            self.content = content

                    def __init__(self, content):
                        self.delta = self.Delta(content)

                class Chunk:
                    def __init__(self, content):
                        self.choices = [Choice(content)]

                class Usage:
                    def __init__(self, p, c, t):
                        self.prompt_tokens = p
                        self.completion_tokens = c
                        self.total_tokens = t

                if stream:
                    # Yield chunks to simulate streaming
                    yield Chunk("Hello")
                    await asyncio.sleep(0.1)  # Simulate network delay
                    yield Chunk(" World")

                # Attaching usage to the last chunk-like object is tricky in pure mock
                # implementation without a wrapper, so we simplify for the example
                # function's usage logic.
                pass


client = MockOpenAIClient()


async def call_llm_monitored(
    prompt: str, model: str = "gpt-4", endpoint: str = "/generate", mock_client: Any = None
) -> str:
    """
    Fully instrumented LLM call with tracing and metrics.

    Args:
        prompt: User input
        model: Model identifier
        endpoint: API endpoint name for tagging
        mock_client: Optional injected client for testing
    """
    # local_client = mock_client if mock_client else client
    # Using client directly for this example since we mocked it globally

    with tracer.start_as_current_span("llm.generate") as span:
        # Span attributes
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.endpoint", endpoint)
        span.set_attribute("llm.prompt_length", len(prompt))

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()
        start_time = time.time()
        ttft = None

        try:
            # Simulate LLM call structure matching the example
            # Note: In a real app complexity, usage comes at the end.
            # We mock the response object structure needed by logic below.

            # For this example extraction, we adapt slightly to make it runnable
            # without a real OpenAI connection.

            # --- Simulation Block Start ---
            # Simulate network latency
            await asyncio.sleep(0.1)
            ttft = time.time() - start_time
            LLM_TTFT_SECONDS.labels(model=model).observe(ttft)
            span.set_attribute("llm.ttft_ms", ttft * 1000)

            full_response = "Mocked LLM Response"
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
            # --- Simulation Block End ---

            latency = time.time() - start_time

            # Calculate cost
            cost = calculate_cost(model, total_tokens)

            # Record metrics
            LLM_TOKENS_TOTAL.labels(model=model, type="prompt", endpoint=endpoint).inc(
                prompt_tokens
            )
            LLM_TOKENS_TOTAL.labels(model=model, type="completion", endpoint=endpoint).inc(
                completion_tokens
            )
            LLM_LATENCY_SECONDS.labels(model=model, endpoint=endpoint).observe(latency)
            LLM_COST_USD.labels(model=model, endpoint=endpoint).inc(cost)
            LLM_REQUESTS_TOTAL.labels(model=model, endpoint=endpoint, status="success").inc()

            # Span attributes
            span.set_attribute("llm.tokens.prompt", prompt_tokens)
            span.set_attribute("llm.tokens.completion", completion_tokens)
            span.set_attribute("llm.tokens.total", total_tokens)
            span.set_attribute("llm.latency_ms", latency * 1000)
            span.set_attribute("llm.cost_usd", cost)
            span.set_status(trace.Status(trace.StatusCode.OK))

            return full_response

        except Exception as e:
            LLM_REQUESTS_TOTAL.labels(model=model, endpoint=endpoint, status="error").inc()
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            LLM_ACTIVE_REQUESTS.labels(model=model).dec()
