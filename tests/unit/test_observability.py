"""Tests for observability modules — logging, metrics, tracing."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.observability.logging import get_logger, setup_logging
from src.infrastructure.observability.metrics import (
    agent_steps_counter,
    llm_cost_counter,
    llm_error_counter,
    llm_latency_histogram,
    llm_request_counter,
    llm_token_counter,
    rag_documents_retrieved,
    rag_retrieval_latency,
)
from src.infrastructure.observability.tracing import get_tracer, setup_tracing

_has_ctypes = importlib.util.find_spec("_ctypes") is not None


class TestLogging:
    def test_setup_logging_runs(self) -> None:
        settings = MagicMock()
        settings.observability.LOG_LEVEL = "INFO"
        setup_logging(settings)

    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("test")
        assert logger is not None


class TestMetrics:
    def test_all_metrics_exist(self) -> None:
        assert llm_request_counter is not None
        assert llm_error_counter is not None
        assert llm_latency_histogram is not None
        assert llm_token_counter is not None
        assert llm_cost_counter is not None
        assert rag_retrieval_latency is not None
        assert rag_documents_retrieved is not None
        assert agent_steps_counter is not None


class TestTracing:
    def test_setup_tracing_console(self) -> None:
        settings = MagicMock()
        settings.APP_NAME = "test-app"
        settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = ""
        setup_tracing(settings)

    @pytest.mark.skipif(not _has_ctypes, reason="OTLP exporter requires _ctypes")
    @patch(
        "src.infrastructure.observability.tracing.OTLPSpanExporter",
        create=True,
    )
    def test_setup_tracing_otlp(self, mock_exporter: MagicMock) -> None:
        settings = MagicMock()
        settings.APP_NAME = "test-app"
        settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4317"
        setup_tracing(settings)

    def test_get_tracer_returns_tracer(self) -> None:
        tracer = get_tracer("test")
        assert tracer is not None
