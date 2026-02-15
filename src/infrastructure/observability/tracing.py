"""OpenTelemetry tracing setup.

Reference implementation for distributed tracing.
All LLM calls, agent steps, and tool invocations are traced.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

from src.infrastructure.config import Settings


def setup_tracing(settings: Settings) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        settings: The application settings object.
    """
    resource = Resource.create({"service.name": settings.APP_NAME})
    provider = TracerProvider(resource=resource)

    exporter: SpanExporter
    if settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer instance."""
    return trace.get_tracer(name)
