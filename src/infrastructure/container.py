# src/infrastructure/container.py
import logging
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.application.services.prompt_registry import PromptRegistry
from src.domain.ports.audit_trail_port import AuditTrailPort
from src.domain.ports.cost_tracker_port import CostTrackerPort
from src.domain.ports.data_sink_port import DataSinkPort
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.ports.event_bus_port import EventBusPort
from src.domain.ports.hallucination_checker_port import HallucinationCheckerPort
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.mcp_client_port import MCPClientPort
from src.domain.ports.message_broker_port import MessageBrokerPort
from src.domain.ports.pii_masking_port import PIIMaskingPort
from src.domain.ports.reranker_port import RerankerPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.ports.storage_port import StoragePort
from src.domain.ports.vector_store_port import VectorStorePort
from src.infrastructure.cache.redis_cache import RedisCache
from src.infrastructure.config import Settings
from src.infrastructure.database.session import create_session_factory
from src.infrastructure.events.in_memory_event_bus import InMemoryEventBus
from src.infrastructure.llm.openai_adapter import OpenAIAdapter
from src.infrastructure.resilience.resilient_llm import ResilientLLMPort
from src.infrastructure.vectorstores.chromadb_adapter import ChromaDBAdapter

logger = logging.getLogger(__name__)

_IN_MEMORY_WARNING = (
    "Using in-memory %s — data will be lost on restart. Set BACKENDS__%s in production."
)


class DBAdapterProtocol(Protocol):
    """Protocol for database adapter management."""

    session_maker: async_sessionmaker[AsyncSession]


class Container:  # pragma: no cover
    """Dependency injection container for the application."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm_adapter: LLMPort | None = None
        self._chromadb_adapter: ChromaDBAdapter | None = None  # Single instance for both roles
        self._redis_cache: RedisCache | None = None
        self._db_session_factory: async_sessionmaker[AsyncSession] | None = None
        self._storage_adapter: StoragePort | None = None
        self._event_bus: EventBusPort | None = None
        self._data_source: DataSourcePort | None = None
        self._data_sink: DataSinkPort | None = None
        self._data_validator: DataValidatorPort | None = None
        self._message_broker: MessageBrokerPort | None = None
        self._mcp_client: MCPClientPort | None = None
        self._hallucination_checker: HallucinationCheckerPort | None = None
        self._prompt_registry: PromptRegistry | None = None
        self._reranker: RerankerPort | None = None
        self._audit_trail: AuditTrailPort | None = None
        self._pii_masker: PIIMaskingPort | None = None
        self._cost_tracker: CostTrackerPort | None = None

    @property
    def llm_adapter(self) -> LLMPort:
        if self._llm_adapter is None:
            raw: LLMPort
            if self._settings.llm.MODE == "openai":
                raw = OpenAIAdapter()
            else:
                from src.infrastructure.llm.litellm_adapter import LiteLLMAdapter

                model = (
                    f"{self._settings.llm.MODE}/{self._settings.llm.ANTHROPIC_MODEL}"
                    if self._settings.llm.MODE == "anthropic"
                    else self._settings.llm.MODE
                )
                raw = LiteLLMAdapter(
                    model=model,
                    api_key=self._settings.llm.ANTHROPIC_API_KEY,
                )
            self._llm_adapter = ResilientLLMPort(raw)
        assert self._llm_adapter is not None
        return self._llm_adapter

    @property
    def retriever_adapter(self) -> RetrieverPort:
        if self._chromadb_adapter is None:
            self._chromadb_adapter = ChromaDBAdapter()
        assert self._chromadb_adapter is not None
        return self._chromadb_adapter

    @property
    def vector_store_adapter(self) -> VectorStorePort:
        if self._chromadb_adapter is None:
            self._chromadb_adapter = ChromaDBAdapter()
        assert self._chromadb_adapter is not None
        return self._chromadb_adapter

    @property
    def redis_cache(self) -> RedisCache:
        if self._redis_cache is None:
            # We already modified RedisCache to use get_settings()
            self._redis_cache = RedisCache()
        return self._redis_cache

    @property
    def db_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._db_session_factory is None:
            # We already modified create_session_factory to use get_settings()
            self._db_session_factory = create_session_factory()
        return self._db_session_factory

    @property
    def storage_adapter(self) -> StoragePort:
        """Get storage adapter (LocalStorage or S3)."""
        if self._storage_adapter is None:
            if self._settings.cloud.STORAGE_BACKEND == "s3":
                from src.infrastructure.cloud.s3_storage import S3Storage

                self._storage_adapter = S3Storage(
                    bucket=self._settings.cloud.S3_BUCKET,
                    region=self._settings.cloud.S3_REGION,
                    endpoint_url=self._settings.cloud.S3_ENDPOINT_URL or None,
                )
            else:
                from src.infrastructure.storage.local_storage import LocalStorage

                storage_path = getattr(self._settings, "STORAGE_PATH", "./data")
                self._storage_adapter = LocalStorage(base_path=storage_path)
        return self._storage_adapter

    @property
    def event_bus(self) -> EventBusPort:
        """Get event bus adapter."""
        if self._event_bus is None:
            if self._settings.ENVIRONMENT not in ("development", "testing"):
                logger.warning(_IN_MEMORY_WARNING, "event bus", "EVENT_BUS")
            self._event_bus = InMemoryEventBus()
        return self._event_bus

    @property
    def data_source(self) -> DataSourcePort:
        """Get data source adapter (LocalFileSource)."""
        if self._data_source is None:
            from src.infrastructure.data.local_file_source import LocalFileSource

            self._data_source = LocalFileSource()
        return self._data_source

    @property
    def data_sink(self) -> DataSinkPort:
        """Get data sink adapter (LocalFileSink)."""
        if self._data_sink is None:
            from src.infrastructure.data.local_file_sink import LocalFileSink

            self._data_sink = LocalFileSink()
        return self._data_sink

    @property
    def data_validator(self) -> DataValidatorPort:
        """Get data validator adapter (PydanticDataValidator)."""
        if self._data_validator is None:
            from src.infrastructure.data.pydantic_data_validator import PydanticDataValidator

            self._data_validator = PydanticDataValidator()
        return self._data_validator

    @property
    def message_broker(self) -> MessageBrokerPort:
        """Get message broker adapter based on settings."""
        if self._message_broker is None:
            backend = self._settings.backends.MESSAGE_BROKER
            if backend == "redis":
                from src.infrastructure.events.redis_streams_broker import RedisStreamsBroker

                self._message_broker = RedisStreamsBroker(
                    redis_url=self._settings.redis.URL,
                )
            else:
                from src.infrastructure.data.in_memory_message_broker import InMemoryMessageBroker

                if self._settings.ENVIRONMENT not in ("development", "testing"):
                    logger.warning(_IN_MEMORY_WARNING, "message broker", "MESSAGE_BROKER=redis")
                self._message_broker = InMemoryMessageBroker()
        return self._message_broker

    @property
    def mcp_client(self) -> MCPClientPort:
        """Get MCP client adapter (StdioMCPClient)."""
        if self._mcp_client is None:
            from src.infrastructure.mcp.stdio_mcp_client import StdioMCPClient

            self._mcp_client = StdioMCPClient(command="mcp-server")
        return self._mcp_client

    @property
    def hallucination_checker(self) -> HallucinationCheckerPort:
        """Get hallucination checker adapter (LLM-based)."""
        if self._hallucination_checker is None:
            from src.infrastructure.hallucination.llm_checker import LLMHallucinationChecker

            self._hallucination_checker = LLMHallucinationChecker(llm=self.llm_adapter)
        return self._hallucination_checker

    @property
    def audit_trail(self) -> AuditTrailPort:
        """Get the audit trail implementation."""
        if self._audit_trail is None:
            if self._settings.ENVIRONMENT not in ("development", "testing"):
                logger.warning(_IN_MEMORY_WARNING, "audit trail", "AUDIT_TRAIL")
            from src.infrastructure.governance.in_memory_audit_trail import InMemoryAuditTrail

            self._audit_trail = InMemoryAuditTrail()
        return self._audit_trail

    @property
    def pii_masker(self) -> PIIMaskingPort:
        """Get the PII masker implementation."""
        if self._pii_masker is None:
            from src.infrastructure.governance.regex_pii_masker import RegexPIIMasker

            self._pii_masker = RegexPIIMasker()
        return self._pii_masker

    @property
    def cost_tracker(self) -> CostTrackerPort:
        """Get the cost tracker implementation."""
        if self._cost_tracker is None:
            if self._settings.ENVIRONMENT not in ("development", "testing"):
                logger.warning(_IN_MEMORY_WARNING, "cost tracker", "COST_TRACKER")
            from src.infrastructure.analytics.in_memory_cost_tracker import InMemoryCostTracker

            self._cost_tracker = InMemoryCostTracker()
        return self._cost_tracker

    @property
    def reranker(self) -> RerankerPort:
        """Get the LLM-based reranker."""
        if self._reranker is None:
            from src.infrastructure.rag.llm_reranker import LLMReranker

            self._reranker = LLMReranker(llm=self.llm_adapter)
        return self._reranker

    @property
    def prompt_registry(self) -> PromptRegistry:
        """Get the prompt template registry."""
        if self._prompt_registry is None:
            self._prompt_registry = PromptRegistry()
        return self._prompt_registry

    @property
    def spark_data_source(self) -> DataSourcePort:
        """Get Spark JDBC data source adapter."""
        from src.infrastructure.data.spark.session_manager import SparkSessionManager
        from src.infrastructure.data.spark.spark_jdbc_source import SparkJDBCSource

        spark = SparkSessionManager.get_or_create(self._settings.spark)
        return SparkJDBCSource(spark=spark, fetch_size=self._settings.spark.JDBC_FETCH_SIZE)

    @property
    def spark_file_source(self) -> DataSourcePort:
        """Get Spark file data source adapter."""
        from src.infrastructure.data.spark.session_manager import SparkSessionManager
        from src.infrastructure.data.spark.spark_file_source import SparkFileSource

        spark = SparkSessionManager.get_or_create(self._settings.spark)
        return SparkFileSource(spark=spark)

    @property
    def spark_data_sink(self) -> DataSinkPort:
        """Get Spark JDBC data sink adapter."""
        from src.infrastructure.data.spark.session_manager import SparkSessionManager
        from src.infrastructure.data.spark.spark_jdbc_sink import SparkJDBCSink

        spark = SparkSessionManager.get_or_create(self._settings.spark)
        return SparkJDBCSink(spark=spark)

    @property
    def spark_file_sink(self) -> DataSinkPort:
        """Get Spark file data sink adapter."""
        from src.infrastructure.data.spark.session_manager import SparkSessionManager
        from src.infrastructure.data.spark.spark_file_sink import SparkFileSink

        spark = SparkSessionManager.get_or_create(self._settings.spark)
        return SparkFileSink(spark=spark)

    async def close(self) -> None:
        """Close connections of managed adapters."""
        if self._redis_cache:
            await self._redis_cache.close()
        if self._message_broker:
            await self._message_broker.close()
        if self._mcp_client:
            await self._mcp_client.close()
        # Stop SparkSession if it was started
        try:
            from src.infrastructure.data.spark.session_manager import SparkSessionManager

            SparkSessionManager.stop()
        except ImportError:
            pass
